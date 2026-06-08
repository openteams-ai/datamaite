"""HTML validation report generator."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from databridge._formats.hmie.categories import (
    _CATEGORY_LABELS,
    SKIP_VIDEO_INTEGRITY,
    _categorize_findings,
    skipped_category_keys,
)
from databridge._types import (
    ValidationResult,
    _shorten_label,
)


def prepare_report_data(result: ValidationResult) -> dict[str, Any]:
    """Convert a ValidationResult into a JSON-serializable dict for the HTML template."""
    cats = _categorize_findings(result.finding_severity_counts)
    skipped_cats = skipped_category_keys(result.skipped_checks)
    root = result.dataset_path

    categories = []
    for key in ("structure", "video", "coverage", "scale_spec"):
        errs, warns = cats[key]
        # SKIPPED only when nothing fired: some video-category findings still
        # run with video checks off (e.g. multiple_videos_in_seq_mp4), and a
        # real FAIL/WARN must take precedence over the skip banner.
        if key in skipped_cats and errs == 0 and warns == 0:
            status = "skipped"
        elif errs > 0:
            status = "fail"
        elif warns > 0:
            status = "warn"
        else:
            status = "pass"
        categories.append(
            {
                "key": key,
                "label": _CATEGORY_LABELS[key],
                "status": status,
                "errors": errs,
                "warnings": warns,
            }
        )

    grouped: dict[str, list[dict[str, Any]]] = {}
    for f in result.findings:
        try:
            rel = str(f.path.relative_to(root))
        except ValueError:
            rel = str(f.path)
        grouped.setdefault(rel, []).append(
            {
                "severity": f.severity.value,
                "check": f.check,
                "message": f.message,
            }
        )

    finding_groups = sorted(
        [{"path": path, "findings": findings, "count": len(findings)} for path, findings in grouped.items()],
        key=lambda g: g["path"],
    )

    labels = [{"name": _shorten_label(label), "count": count} for label, count in result.label_histogram.most_common()]
    max_label_count = labels[0]["count"] if labels else 1

    finding_counts = [{"check": check, "count": count} for check, count in result.finding_counts.most_common()]

    return {
        "is_multi": False,
        "dataset_path": str(root),
        "dataset_name": root.name,
        "passed": result.passed,
        "snippet_count": result.snippet_count,
        "annotation_count": result.annotation_count,
        "cache_hits": result.cache_hits,
        "cache_misses": result.cache_misses,
        # Totals come from uncapped severity counts, not findings (which
        # may be truncated when max_findings_per_check is set).
        "error_count": sum(result.finding_severity_counts.get("error", Counter()).values()),
        "warning_count": sum(result.finding_severity_counts.get("warning", Counter()).values()),
        "categories": categories,
        "skipped_checks": sorted(result.skipped_checks),
        "video_checks_skipped": SKIP_VIDEO_INTEGRITY in result.skipped_checks,
        "finding_groups": finding_groups,
        "finding_counts": finding_counts,
        "labels": labels,
        "max_label_count": max_label_count,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
    }


def _aggregate_categories(batches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cat_totals: dict[str, list[int]] = {
        "structure": [0, 0],
        "video": [0, 0],
        "coverage": [0, 0],
        "scale_spec": [0, 0],
    }
    for b in batches:
        for cat in b["categories"]:
            cat_totals[cat["key"]][0] += cat["errors"]
            cat_totals[cat["key"]][1] += cat["warnings"]

    # Count, per category, how many batches skipped it. Per-batch status is
    # already "skipped" only with zero errors/warnings, so a batch with a real
    # video-category finding reads warn/fail and is not counted here.
    n_batches = len(batches)
    skipped_counts: dict[str, int] = dict.fromkeys(cat_totals, 0)
    for b in batches:
        for c in b["categories"]:
            if c["status"] == "skipped":
                skipped_counts[c["key"]] += 1

    categories = []
    for key in ("structure", "video", "coverage", "scale_spec"):
        errs, warns = cat_totals[key]
        categories.append(
            {
                "key": key,
                "label": _CATEGORY_LABELS[key],
                "status": _aggregate_status(errs, warns, skipped_counts[key], n_batches),
                "errors": errs,
                "warnings": warns,
            }
        )
    return categories


def _aggregate_status(errs: int, warns: int, skipped_n: int, n_batches: int) -> str:
    """Resolve one aggregate category status across batches.

    Precedence: real findings first, then all-skipped, then the mixed case. A
    green "all clear" aggregate must not hide that SOME batches never ran the
    check -- that reads as "all batches checked and clean". "partial" only
    arises when the category is otherwise clean, so it never masks a fail/warn.
    """
    if errs > 0:
        return "fail"
    if warns > 0:
        return "warn"
    if n_batches and skipped_n == n_batches:
        return "skipped"
    if skipped_n > 0:
        return "partial"
    return "pass"


def _aggregate_finding_groups(batches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for b in batches:
        bname = b["batch_name"]
        groups.extend(
            {"path": f"{bname}/{g['path']}", "findings": g["findings"], "count": g["count"]}
            for g in b["finding_groups"]
        )
    groups.sort(key=lambda g: g["path"])
    return groups


def _aggregate_counter(
    batches: list[dict[str, Any]],
    list_key: str,
    item_name_key: str,
    item_count_key: str,
    out_name_key: str,
) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    for b in batches:
        for item in b[list_key]:
            counter[item[item_name_key]] += item[item_count_key]
    return [{out_name_key: name, "count": count} for name, count in counter.most_common()]


def _aggregate_batches(batches: list[dict[str, Any]]) -> dict[str, Any]:
    """Build an aggregated 'view' across all batches.

    Same shape as ``prepare_report_data``'s output so the JS can render it
    with the same code path. Finding group paths are prefixed with the
    batch name so the source is visible in the unified findings list.
    """
    finding_counts = _aggregate_counter(batches, "finding_counts", "check", "count", "check")
    labels = _aggregate_counter(batches, "labels", "name", "count", "name")
    return {
        "categories": _aggregate_categories(batches),
        "finding_groups": _aggregate_finding_groups(batches),
        "finding_counts": finding_counts,
        "labels": labels,
        "max_label_count": labels[0]["count"] if labels else 1,
        "snippet_count": sum(b["snippet_count"] for b in batches),
        "annotation_count": sum(b["annotation_count"] for b in batches),
        "error_count": sum(b["error_count"] for b in batches),
        "warning_count": sum(b["warning_count"] for b in batches),
        "cache_hits": sum(b["cache_hits"] for b in batches),
        "cache_misses": sum(b["cache_misses"] for b in batches),
        "passed": all(b["passed"] for b in batches),
        "video_checks_skipped": any(b["video_checks_skipped"] for b in batches),
    }


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Databridge Validation Report</title>
<style>
/* ---- Color tokens ---- */
:root {
  --pass: #22c55e;
  --pass-bg: #dcfce7;
  --warn: #eab308;
  --warn-bg: #fef9c3;
  --fail: #ef4444;
  --fail-bg: #fee2e2;
  --bg-primary: #ffffff;
  --bg-secondary: #f8fafc;
  --bg-tertiary: #f1f5f9;
  --text-primary: #0f172a;
  --text-secondary: #475569;
  --text-muted: #94a3b8;
  --border: #e2e8f0;
  --border-strong: #cbd5e1;
  --link: #2563eb;
  --shadow: 0 1px 3px rgba(0,0,0,0.08);
  --radius: 8px;
}
[data-theme="dark"] {
  --pass-bg: #14532d;
  --warn-bg: #422006;
  --fail-bg: #450a0a;
  --bg-primary: #0f172a;
  --bg-secondary: #1e293b;
  --bg-tertiary: #334155;
  --text-primary: #f1f5f9;
  --text-secondary: #94a3b8;
  --text-muted: #64748b;
  --border: #334155;
  --border-strong: #475569;
  --link: #60a5fa;
  --shadow: 0 1px 3px rgba(0,0,0,0.3);
}
/* ---- Base ---- */
*, *::before, *::after { box-sizing: border-box; }
body {
  margin: 0; padding: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
    "Helvetica Neue", Arial, sans-serif;
  background: var(--bg-secondary);
  color: var(--text-primary);
  line-height: 1.5;
}
.container { max-width: 1080px; margin: 0 auto; padding: 24px 16px; }
code, .mono { font-family: "SF Mono", "Cascadia Code", "Consolas", monospace; font-size: 0.875em; }
/* ---- Header ---- */
.header {
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 24px;
  margin-bottom: 16px;
  box-shadow: var(--shadow);
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 12px;
}
.header h1 { margin: 0; font-size: 1.25rem; font-weight: 600; word-break: break-all; }
.header .path { color: var(--text-secondary); font-size: 0.8rem; margin-top: 4px; }
.verdict {
  font-size: 0.875rem; font-weight: 700;
  padding: 6px 20px; border-radius: 9999px;
  text-transform: uppercase; letter-spacing: 0.05em;
  flex-shrink: 0;
}
.verdict-pass { background: var(--pass); color: #fff; }
.verdict-fail { background: var(--fail); color: #fff; }
/* ---- Breadcrumb ---- */
.breadcrumb {
  font-size: 0.85rem;
  margin-bottom: 12px;
  color: var(--text-secondary);
  display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
}
.breadcrumb a {
  color: var(--link);
  text-decoration: none;
  cursor: pointer;
}
.breadcrumb a:hover { text-decoration: underline; }
.breadcrumb .scope { font-weight: 600; color: var(--text-primary); }
/* ---- Toolbar ---- */
.toolbar {
  display: flex; gap: 8px; align-items: center; flex-wrap: wrap;
  margin-bottom: 8px;
}
.toolbar button, .toolbar select, .toolbar input {
  font-size: 0.8rem; padding: 6px 12px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--bg-primary);
  color: var(--text-primary);
  cursor: pointer;
}
.toolbar input { flex: 1; min-width: 180px; }
.toolbar button:hover { background: var(--bg-tertiary); }
/* ---- Dashboard grid ---- */
.dashboard {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 12px;
  margin-bottom: 16px;
}
.check-card {
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-left: 4px solid var(--border-strong);
  border-radius: var(--radius);
  padding: 16px 20px;
  box-shadow: var(--shadow);
}
.check-card[data-status="pass"] { border-left-color: var(--pass); }
.check-card[data-status="warn"] { border-left-color: var(--warn); }
.check-card[data-status="fail"] { border-left-color: var(--fail); }
.check-card[data-status="skipped"] { border-left-color: var(--text-muted); }
.check-card[data-status="partial"] { border-left-color: var(--warn); }
.check-card .label { font-weight: 600; font-size: 0.95rem; margin-bottom: 4px; }
.check-card .detail { font-size: 0.8rem; color: var(--text-secondary); }
/* ---- Status pill ---- */
.status-pill {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 9999px;
  font-size: 0.7rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  vertical-align: middle;
}
.pill-pass { background: var(--pass-bg); color: var(--pass); }
.pill-warn { background: var(--warn-bg); color: var(--warn); }
.pill-fail { background: var(--fail-bg); color: var(--fail); }
.pill-skipped { background: var(--bg-tertiary); color: var(--text-secondary); }
.pill-partial { background: var(--warn-bg); color: var(--warn); }
.skip-banner {
  background: var(--warn-bg); color: var(--text-primary);
  border: 1px solid var(--warn); border-radius: var(--radius);
  padding: 10px 16px; margin-bottom: 16px; font-size: 0.85rem;
}
/* ---- Stats bar ---- */
.stats-bar {
  display: flex;
  gap: 0;
  margin-bottom: 16px;
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  overflow: hidden;
}
.stat-cell {
  flex: 1;
  padding: 14px 16px;
  text-align: center;
  border-right: 1px solid var(--border);
}
.stat-cell:last-child { border-right: none; }
.stat-cell .value { font-size: 1.4rem; font-weight: 700; }
.stat-cell .label {
  font-size: 0.7rem; color: var(--text-secondary);
  text-transform: uppercase; letter-spacing: 0.05em;
}
/* ---- Findings section ---- */
.findings-section {
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px;
  margin-bottom: 16px;
  box-shadow: var(--shadow);
}
.findings-section h2 { margin: 0 0 12px; font-size: 1rem; }
.finding-group {
  border: 1px solid var(--border);
  border-radius: 6px;
  margin-bottom: 6px;
  overflow: hidden;
}
.finding-group summary {
  padding: 8px 14px;
  cursor: pointer;
  font-size: 0.85rem;
  background: var(--bg-secondary);
  list-style: none;
  display: flex;
  align-items: center;
  gap: 8px;
}
.finding-group summary::-webkit-details-marker { display: none; }
.finding-group summary::before {
  content: "\\25B6"; font-size: 0.65rem;
  color: var(--text-muted); transition: transform 0.15s;
}
.finding-group[open] summary::before { transform: rotate(90deg); }
.finding-group .file-path { flex: 1; word-break: break-all; }
.finding-group .count-badge {
  background: var(--bg-tertiary);
  color: var(--text-secondary);
  padding: 1px 8px;
  border-radius: 9999px;
  font-size: 0.7rem;
  font-weight: 600;
}
.finding-row {
  display: flex; gap: 8px; align-items: baseline;
  padding: 6px 14px 6px 30px;
  font-size: 0.8rem;
  border-top: 1px solid var(--border);
}
.severity-badge {
  display: inline-block;
  padding: 1px 8px;
  border-radius: 4px;
  font-size: 0.7rem;
  font-weight: 700;
  text-transform: uppercase;
  flex-shrink: 0;
}
.badge-error { background: var(--fail); color: #fff; }
.badge-warn { background: var(--warn-bg); color: #92400e; }
[data-theme="dark"] .badge-warn { background: #fef3c7; color: #92400e; }
.check-name {
  padding: 1px 7px;
  background: var(--bg-tertiary);
  border-radius: 4px;
  font-size: 0.75rem;
  flex-shrink: 0;
}
.finding-msg { color: var(--text-secondary); word-break: break-word; }
.show-more-btn {
  display: block; margin: 12px auto 0;
  padding: 8px 24px; border: 1px solid var(--border);
  border-radius: 6px; background: var(--bg-primary);
  color: var(--text-primary); font-size: 0.8rem;
  cursor: pointer;
}
.show-more-btn:hover { background: var(--bg-tertiary); }
.empty-msg { color: var(--text-muted); font-size: 0.85rem; font-style: italic; padding: 12px 0; }
/* ---- Tables ---- */
.data-table { width: 100%; border-collapse: collapse; font-size: 0.8rem; margin-top: 8px; }
.data-table th {
  text-align: left; padding: 8px 12px;
  background: var(--bg-secondary); border-bottom: 2px solid var(--border);
  font-weight: 600; font-size: 0.7rem; text-transform: uppercase;
  letter-spacing: 0.04em; color: var(--text-secondary);
}
.data-table td { padding: 6px 12px; border-bottom: 1px solid var(--border); }
.data-table .num { text-align: right; font-variant-numeric: tabular-nums; }
.data-table tr.clickable { cursor: pointer; }
.data-table tr.clickable:hover { background: var(--bg-secondary); }
.data-table tr.clickable:focus { outline: 2px solid var(--link); outline-offset: -2px; }
.data-table tr.active-row { background: var(--bg-tertiary); }
.data-table tr.active-row:hover { background: var(--bg-tertiary); }
/* ---- Bar chart ---- */
.bar-chart { margin-top: 8px; }
.bar-row { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; font-size: 0.8rem; }
.bar-label { width: 200px; text-align: right; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.bar-track { flex: 1; height: 20px; background: var(--bg-tertiary); border-radius: 4px; overflow: hidden; }
.bar-fill {
  display: block; height: 100%; background: var(--pass);
  border-radius: 4px; min-width: 2px; transition: width 0.3s;
}
.bar-count { width: 60px; font-variant-numeric: tabular-nums; }
/* ---- Section card ---- */
.section-card {
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px;
  margin-bottom: 16px;
  box-shadow: var(--shadow);
}
.section-card h2 { margin: 0 0 12px; font-size: 1rem; }
/* ---- Batch table (multi-report) ---- */
.batch-section {
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px;
  margin-bottom: 16px;
  box-shadow: var(--shadow);
}
.batch-section h2 { margin: 0 0 12px; font-size: 1rem; }
.batch-summary { font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 12px; }
/* ---- Footer ---- */
.footer { text-align: center; color: var(--text-muted); font-size: 0.7rem; padding: 16px 0; }
/* ---- Print ---- */
@media print {
  :root, [data-theme="dark"] {
    --pass: #22c55e; --pass-bg: #dcfce7;
    --warn: #eab308; --warn-bg: #fef9c3;
    --fail: #ef4444; --fail-bg: #fee2e2;
    --bg-primary: #fff; --bg-secondary: #f8fafc; --bg-tertiary: #f1f5f9;
    --text-primary: #0f172a; --text-secondary: #475569; --text-muted: #94a3b8;
    --border: #e2e8f0; --border-strong: #cbd5e1;
    --shadow: none;
  }
  body { background: #fff; }
  .toolbar, .show-more-btn, #theme-toggle { display: none !important; }
  .finding-group[open] summary ~ * { display: block; }
  details { break-inside: avoid; }
  .badge-warn { background: #fef9c3; color: #92400e; }
}
/* ---- Responsive ---- */
@media (max-width: 768px) {
  .dashboard { grid-template-columns: 1fr; }
  .stats-bar { flex-wrap: wrap; }
  .stat-cell { flex-basis: 50%; border-bottom: 1px solid var(--border); }
  .header { flex-direction: column; align-items: flex-start; }
  .data-table { display: block; overflow-x: auto; }
  .bar-label { width: 120px; font-size: 0.75rem; }
}
</style>
</head>
<body>
<div class="container">
  <div class="header" id="report-header">
    <div>
      <h1 id="dataset-name"></h1>
      <div class="path mono" id="dataset-path"></div>
    </div>
    <div style="display:flex;gap:8px;align-items:center;">
      <div id="verdict"></div>
      <button id="theme-toggle" class="theme-btn"
        style="border:1px solid var(--border);border-radius:6px;padding:4px 10px;
               font-size:0.8rem;cursor:pointer;background:var(--bg-primary);
               color:var(--text-primary);"
        aria-label="Toggle theme">&#9790;</button>
    </div>
  </div>
  <div class="breadcrumb" id="breadcrumb" style="display:none;"></div>
  <div class="batch-section" id="batch-section" style="display:none;">
    <h2>Batch Results</h2>
    <div class="batch-summary" id="batch-summary"></div>
    <table class="data-table" id="batch-table">
      <thead><tr>
        <th>Batch</th><th class="num">Snippets</th><th class="num">Annotations</th>
        <th class="num">Errors</th><th class="num">Warnings</th><th>Status</th>
      </tr></thead>
      <tbody id="batch-body"></tbody>
    </table>
  </div>
  <div class="skip-banner" id="skip-banner" style="display:none;"></div>
  <div class="dashboard" id="dashboard"></div>
  <div class="stats-bar" id="stats-bar"></div>
  <div class="findings-section" id="findings-section">
    <h2>Findings</h2>
    <div class="toolbar" id="findings-toolbar">
      <input type="text" id="search-input" placeholder="Search by path or message..." aria-label="Search findings">
      <select id="severity-filter" aria-label="Filter by severity">
        <option value="all">All</option>
        <option value="error">Errors only</option>
        <option value="warning">Warnings only</option>
      </select>
      <button id="expand-btn">Expand all</button>
    </div>
    <div id="findings-list"></div>
    <button id="show-more-btn" class="show-more-btn" style="display:none;">Show more</button>
  </div>
  <div class="section-card" id="counts-section">
    <h2>Findings by Check</h2>
    <table class="data-table" id="counts-table">
      <thead><tr><th>Check</th><th class="num">Count</th></tr></thead>
      <tbody id="counts-body"></tbody>
    </table>
  </div>
  <div class="section-card" id="labels-section">
    <h2>Label Histogram</h2>
    <div class="bar-chart" id="label-chart"></div>
  </div>
  <div class="footer" id="footer"></div>
</div>

<script>
var REPORT_DATA = {{DATA_JSON}};
(function() {
  "use strict";
  var D = REPORT_DATA;

  /* ---- Helpers ---- */
  function esc(s) {
    var d = document.createElement("div");
    d.appendChild(document.createTextNode(s));
    return d.innerHTML.replace(/"/g, '&quot;');
  }
  function num(n) { return n == null ? "0" : n.toLocaleString(); }

  /* ---- Theme toggle ---- */
  function getTheme() {
    try { return localStorage.getItem("db-report-theme"); } catch(e) { return null; }
  }
  function setTheme(t) {
    try { localStorage.setItem("db-report-theme", t); } catch(e) {}
  }
  var root = document.documentElement;
  var saved = getTheme();
  if (saved) root.setAttribute("data-theme", saved);
  var toggle = document.getElementById("theme-toggle");
  toggle.addEventListener("click", function() {
    var next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    setTheme(next);
    toggle.textContent = next === "dark" ? "\\u2600" : "\\u263E";
  });
  toggle.textContent = root.getAttribute("data-theme") === "dark" ? "\\u2600" : "\\u263E";

  /* ---- Header ---- */
  document.getElementById("dataset-name").textContent = D.dataset_name;
  document.getElementById("dataset-path").textContent = D.dataset_path;

  /* ---- Render functions (take a "view" object) ---- */
  function renderVerdict(view, isAggregate) {
    var v = document.getElementById("verdict");
    v.className = "verdict " + (view.passed ? "verdict-pass" : "verdict-fail");
    if (D.is_multi && isAggregate) {
      v.textContent = view.passed ? "ALL PASS" : "FAIL";
    } else {
      v.textContent = view.passed ? "PASS" : "FAIL";
    }
  }

  function renderDashboard(view) {
    var dash = document.getElementById("dashboard");
    dash.innerHTML = "";
    view.categories.forEach(function(cat) {
      var card = document.createElement("div");
      card.className = "check-card";
      card.setAttribute("data-status", cat.status);
      var pillCls = "status-pill pill-" + cat.status;
      var detail = cat.status === "skipped" ? "checks not run"
        : cat.status === "partial" ? "partially checked; some batches skipped"
        : cat.errors === 0 && cat.warnings === 0 ? "all clear"
        : (cat.errors > 0 ? cat.errors + " error" + (cat.errors !== 1 ? "s" : "") : "")
        + (cat.errors > 0 && cat.warnings > 0 ? ", " : "")
        + (cat.warnings > 0 ? cat.warnings + " warning" + (cat.warnings !== 1 ? "s" : "") : "");
      card.innerHTML = '<div class="label">' + esc(cat.label)
        + ' <span class="' + pillCls + '">' + esc(cat.status.toUpperCase()) + '</span></div>'
        + '<div class="detail">' + esc(detail) + '</div>';
      dash.appendChild(card);
    });
  }

  function renderSkipBanner(view) {
    var banner = document.getElementById("skip-banner");
    if (view.video_checks_skipped) {
      banner.textContent = "\\u26A0 Video checks not run: FMV integrity and "
        + "video\\u2194annotation consistency were skipped (no verified-clean status).";
      banner.style.display = "block";
    } else {
      banner.style.display = "none";
    }
  }

  function renderStats(view) {
    var bar = document.getElementById("stats-bar");
    bar.innerHTML = "";
    var stats = [
      {v: view.snippet_count, l: "Snippets"},
      {v: view.annotation_count, l: "Annotations"},
      {v: view.error_count, l: "Errors"},
      {v: view.warning_count, l: "Warnings"},
      {v: view.cache_hits, l: "Cache Hits"},
      {v: view.cache_misses, l: "Cache Misses"}
    ];
    stats.forEach(function(s) {
      var cell = document.createElement("div");
      cell.className = "stat-cell";
      cell.innerHTML = '<div class="value">' + num(s.v) + '</div><div class="label">' + s.l + '</div>';
      bar.appendChild(cell);
    });
  }

  function renderCounts(view) {
    var section = document.getElementById("counts-section");
    var body = document.getElementById("counts-body");
    body.innerHTML = "";
    if (!view.finding_counts || view.finding_counts.length === 0) {
      section.style.display = "none";
      return;
    }
    section.style.display = "";
    view.finding_counts.forEach(function(fc) {
      var tr = document.createElement("tr");
      tr.innerHTML = '<td><span class="mono">' + esc(fc.check) + '</span></td>'
        + '<td class="num">' + num(fc.count) + '</td>';
      body.appendChild(tr);
    });
  }

  function renderLabels(view) {
    var section = document.getElementById("labels-section");
    var chart = document.getElementById("label-chart");
    chart.innerHTML = "";
    if (!view.labels || view.labels.length === 0) {
      section.style.display = "none";
      return;
    }
    section.style.display = "";
    var maxC = view.max_label_count;
    view.labels.forEach(function(lb) {
      var pct = Math.max(1, Math.round(lb.count / maxC * 100));
      var row = document.createElement("div");
      row.className = "bar-row";
      row.innerHTML = '<span class="bar-label mono" title="' + esc(lb.name) + '">' + esc(lb.name) + '</span>'
        + '<span class="bar-track"><span class="bar-fill" style="width:' + pct + '%"></span></span>'
        + '<span class="bar-count num">' + num(lb.count) + '</span>';
      chart.appendChild(row);
    });
  }

  /* ---- Findings (lazy render with filter) ---- */
  var allGroups = [];
  var filteredGroups = [];
  var rendered = 0;
  var BATCH = 50;
  var list = document.getElementById("findings-list");
  var moreBtn = document.getElementById("show-more-btn");

  function appendFindings(start, count) {
    var end = Math.min(start + count, filteredGroups.length);
    for (var i = start; i < end; i++) {
      var g = filteredGroups[i];
      var det = document.createElement("details");
      det.className = "finding-group";
      var sum = document.createElement("summary");
      sum.innerHTML = '<span class="file-path mono">' + esc(g.path) + '</span>'
        + '<span class="count-badge">' + g.count + '</span>';
      det.appendChild(sum);
      g.findings.forEach(function(f) {
        var row = document.createElement("div");
        row.className = "finding-row";
        row.setAttribute("data-severity", f.severity);
        var bcls = f.severity === "error" ? "badge-error" : "badge-warn";
        row.innerHTML = '<span class="severity-badge ' + bcls + '">' + esc(f.severity) + '</span>'
          + '<span class="check-name mono">' + esc(f.check) + '</span>'
          + '<span class="finding-msg">' + esc(f.message) + '</span>';
        det.appendChild(row);
      });
      list.appendChild(det);
    }
    rendered = end;
    moreBtn.style.display = rendered < filteredGroups.length ? "block" : "none";
  }

  function resetFindings() {
    list.innerHTML = "";
    rendered = 0;
    if (filteredGroups.length === 0) {
      var p = document.createElement("div");
      p.className = "empty-msg";
      p.textContent = allGroups.length === 0
        ? "No findings -- all checks passed."
        : "No findings match the current filter.";
      list.appendChild(p);
      moreBtn.style.display = "none";
      return;
    }
    appendFindings(0, BATCH);
  }

  function renderFindings(view) {
    allGroups = view.finding_groups || [];
    filteredGroups = allGroups;
    var searchInput = document.getElementById("search-input");
    var sevFilter = document.getElementById("severity-filter");
    if (searchInput) searchInput.value = "";
    if (sevFilter) sevFilter.value = "all";
    var expandBtn = document.getElementById("expand-btn");
    if (expandBtn) {
      expandBtn.textContent = "Expand all";
      expandedAll = false;
    }
    resetFindings();
  }

  moreBtn.addEventListener("click", function() { appendFindings(rendered, BATCH); });

  /* ---- Search ---- */
  var searchInput = document.getElementById("search-input");
  var sevFilter = document.getElementById("severity-filter");

  function applyFilter() {
    var q = searchInput.value.toLowerCase();
    var sev = sevFilter.value;
    filteredGroups = allGroups.filter(function(g) {
      var matchSev = sev === "all" || g.findings.some(function(f) { return f.severity === sev; });
      if (!matchSev) return false;
      if (!q) return true;
      if (g.path.toLowerCase().indexOf(q) !== -1) return true;
      return g.findings.some(function(f) {
        return f.message.toLowerCase().indexOf(q) !== -1
          || f.check.toLowerCase().indexOf(q) !== -1;
      });
    });
    resetFindings();
  }
  var filterTimer;
  searchInput.addEventListener("input", function() {
    clearTimeout(filterTimer);
    filterTimer = setTimeout(applyFilter, 200);
  });
  sevFilter.addEventListener("change", applyFilter);

  /* ---- Expand / Collapse ---- */
  var expandBtn = document.getElementById("expand-btn");
  var expandedAll = false;
  expandBtn.addEventListener("click", function() {
    expandedAll = !expandedAll;
    expandBtn.textContent = expandedAll ? "Collapse all" : "Expand all";
    var details = list.querySelectorAll("details");
    for (var i = 0; i < details.length; i++) details[i].open = expandedAll;
  });

  /* ---- Print: expand all before print, collapse after ---- */
  window.addEventListener("beforeprint", function() {
    if (rendered < filteredGroups.length) appendFindings(rendered, filteredGroups.length - rendered);
    var details = list.querySelectorAll("details");
    for (var i = 0; i < details.length; i++) details[i].open = true;
  });
  window.addEventListener("afterprint", function() {
    var dets = document.querySelectorAll("#findings-list details");
    for (var i = 0; i < dets.length; i++) dets[i].removeAttribute("open");
  });

  /* ---- Footer ---- */
  document.getElementById("footer").textContent = "Generated " + D.generated_at + " by Databridge";

  /* ---- Multi-batch routing ---- */
  function renderBatchTable() {
    var batchSection = document.getElementById("batch-section");
    batchSection.style.display = "block";
    document.getElementById("batch-summary").textContent =
      D.passed_count + " of " + D.total_count + " batches passed";
    var batchBody = document.getElementById("batch-body");
    batchBody.innerHTML = "";
    D.batches.forEach(function(b, idx) {
      var tr = document.createElement("tr");
      tr.className = "clickable";
      tr.setAttribute("tabindex", "0");
      tr.setAttribute("role", "button");
      tr.setAttribute("aria-label", "View details for " + b.batch_name);
      tr.setAttribute("data-batch-idx", idx);
      var pillCls = b.passed ? "pill-pass" : "pill-fail";
      var pillText = b.passed ? "PASS" : "FAIL";
      tr.innerHTML = '<td><span class="mono">' + esc(b.batch_name) + '</span></td>'
        + '<td class="num">' + num(b.snippet_count) + '</td>'
        + '<td class="num">' + num(b.annotation_count) + '</td>'
        + '<td class="num">' + num(b.error_count) + '</td>'
        + '<td class="num">' + num(b.warning_count) + '</td>'
        + '<td><span class="status-pill ' + pillCls + '">' + pillText + '</span></td>';
      tr.addEventListener("click", function() { navigate("batch", idx); });
      tr.addEventListener("keydown", function(e) {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          navigate("batch", idx);
        }
      });
      batchBody.appendChild(tr);
    });
  }

  function highlightActiveBatchRow(idx) {
    var rows = document.querySelectorAll("#batch-body tr");
    for (var i = 0; i < rows.length; i++) {
      if (i === idx) rows[i].classList.add("active-row");
      else rows[i].classList.remove("active-row");
    }
  }

  function renderBreadcrumb(viewType, batchName) {
    var bc = document.getElementById("breadcrumb");
    if (!D.is_multi) { bc.style.display = "none"; return; }
    bc.style.display = "flex";
    if (viewType === "batch") {
      bc.innerHTML = '<a id="bc-back" tabindex="0" role="button">&larr; All batches</a>'
        + '<span>/</span>'
        + '<span class="scope mono">' + esc(batchName) + '</span>';
      var back = document.getElementById("bc-back");
      back.addEventListener("click", function() { navigate("aggregate"); });
      back.addEventListener("keydown", function(e) {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); navigate("aggregate"); }
      });
    } else {
      bc.innerHTML = '<span class="scope">All batches</span>'
        + '<span>&middot; ' + D.total_count + ' total</span>'
        + '<span style="color:var(--text-muted);">&middot; click a batch row below for details</span>';
    }
  }

  function applyView(viewType, idx) {
    var view, isAggregate;
    if (viewType === "batch") {
      view = D.batches[idx];
      isAggregate = false;
    } else {
      view = D.is_multi ? D.aggregate : D;
      isAggregate = true;
    }
    renderVerdict(view, isAggregate);
    renderDashboard(view);
    renderSkipBanner(view);
    renderStats(view);
    renderFindings(view);
    renderCounts(view);
    renderLabels(view);
    if (D.is_multi) {
      renderBreadcrumb(viewType, viewType === "batch" ? D.batches[idx].batch_name : null);
      highlightActiveBatchRow(viewType === "batch" ? idx : -1);
      window.scrollTo(0, 0);
    }
  }

  function navigate(viewType, idx) {
    var hash;
    if (viewType === "batch") {
      hash = "#batch=" + encodeURIComponent(D.batches[idx].batch_name);
    } else {
      hash = "";
    }
    var newUrl = location.pathname + location.search + hash;
    if (history && history.pushState) {
      history.pushState({view: viewType, idx: idx}, "", newUrl);
    }
    applyView(viewType, idx);
  }

  function parseHash() {
    var m = location.hash.match(/^#batch=(.+)$/);
    if (m && D.is_multi && D.batches) {
      var name = decodeURIComponent(m[1]);
      for (var i = 0; i < D.batches.length; i++) {
        if (D.batches[i].batch_name === name) return {type: "batch", index: i};
      }
    }
    return {type: "aggregate", index: -1};
  }

  window.addEventListener("popstate", function() {
    var v = parseHash();
    applyView(v.type, v.index);
  });

  /* ---- Initial render ---- */
  if (D.is_multi) {
    renderBatchTable();
    var initial = parseHash();
    applyView(initial.type, initial.index);
  } else {
    applyView("aggregate", -1);
  }
})();
</script>
</body>
</html>
"""


def render_html_report(result: ValidationResult) -> str:
    """Render a single-dataset validation result as a self-contained HTML report."""
    data = prepare_report_data(result)
    data_json = json.dumps(data, default=str)
    safe_json = data_json.replace("</", "<\\/")
    return _HTML_TEMPLATE.replace("{{DATA_JSON}}", safe_json)


def render_html_report_multi(
    results: list[tuple[Path, ValidationResult]],
    root: Path,
) -> str:
    """Render a multi-batch validation report as a self-contained HTML page.

    Parameters
    ----------
    results:
        List of (batch_directory, ValidationResult) pairs.
    root:
        The root directory that contains all batches (used for display).
    """
    batches = []
    for batch_dir, result in results:
        batch_data = prepare_report_data(result)
        batch_data["batch_name"] = batch_dir.name
        batches.append(batch_data)

    aggregate = _aggregate_batches(batches)
    passed_count = sum(1 for _, r in results if r.passed)
    total_count = len(results)

    multi_data: dict[str, Any] = {
        "is_multi": True,
        "root_path": str(root),
        "root_name": root.name,
        "dataset_path": str(root),
        "dataset_name": root.name,
        "passed": passed_count == total_count,
        "passed_count": passed_count,
        "total_count": total_count,
        "snippet_count": aggregate["snippet_count"],
        "annotation_count": aggregate["annotation_count"],
        "error_count": aggregate["error_count"],
        "warning_count": aggregate["warning_count"],
        "cache_hits": aggregate["cache_hits"],
        "cache_misses": aggregate["cache_misses"],
        "aggregate": aggregate,
        "batches": batches,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
    }

    data_json = json.dumps(multi_data, default=str)
    safe_json = data_json.replace("</", "<\\/")
    return _HTML_TEMPLATE.replace("{{DATA_JSON}}", safe_json)
