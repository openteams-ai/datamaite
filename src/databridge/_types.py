"""Shared types for databridge."""

from __future__ import annotations

import enum
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Color support
# ---------------------------------------------------------------------------


def _use_color(force: bool | None = None) -> bool:
    """Return True if we should emit ANSI color codes.

    ``force=True`` or ``force=False`` overrides the auto-detected
    stdout-is-a-tty behavior. Call sites writing to a file (e.g. the
    CLI's ``-o <file>.txt`` path) pass ``force=False`` so the resulting
    file is plain ASCII instead of embedding ``\\033[...m`` escape
    sequences.
    """
    if force is not None:
        return force
    if os.environ.get("NO_COLOR"):
        return False
    if not hasattr(sys.stdout, "isatty"):
        return False
    return sys.stdout.isatty()


def _green(text: str, *, use_color: bool | None = None) -> str:
    return f"\033[32m{text}\033[0m" if _use_color(use_color) else text


def _yellow(text: str, *, use_color: bool | None = None) -> str:
    return f"\033[33m{text}\033[0m" if _use_color(use_color) else text


def _red(text: str, *, use_color: bool | None = None) -> str:
    return f"\033[31m{text}\033[0m" if _use_color(use_color) else text


def _bold(text: str, *, use_color: bool | None = None) -> str:
    return f"\033[1m{text}\033[0m" if _use_color(use_color) else text


def _dim(text: str, *, use_color: bool | None = None) -> str:
    return f"\033[2m{text}\033[0m" if _use_color(use_color) else text


def _status_indicator(errors: int, warnings: int, *, use_color: bool | None = None) -> str:
    """Return a colored status indicator for a check category."""
    if errors > 0:
        return _red("FAIL", use_color=use_color)
    if warnings > 0:
        return _yellow("WARN", use_color=use_color)
    return _green("PASS", use_color=use_color)


def _status_detail(errors: int, warnings: int, *, use_color: bool | None = None) -> str:
    """Return a detail string like '3 errors, 12 warnings' or 'all clear'."""
    if errors == 0 and warnings == 0:
        return _dim("all clear", use_color=use_color)
    parts = []
    if errors > 0:
        parts.append(f"{errors:,} errors")
    if warnings > 0:
        parts.append(f"{warnings:,} warnings")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_findings(
    findings: list[Finding],
    root: Path,
    max_findings: int | None,
    lines: list[str],
) -> None:
    """Append grouped findings to lines, optionally capped."""
    grouped: dict[Path, list[Finding]] = {}
    for f in findings:
        grouped.setdefault(f.path, []).append(f)

    cap = max_findings if max_findings is not None else len(findings)
    shown = 0
    for path, file_findings in sorted(grouped.items()):
        if shown >= cap:
            break
        try:
            rel = path.relative_to(root)
        except ValueError:
            rel = path
        lines.append("")
        lines.append(f"  {rel}")
        for f in file_findings:
            if shown >= cap:
                break
            tag = "error" if f.severity == Severity.ERROR else "warn"
            lines.append(f"    {tag}[{f.check}]: {f.message}")
            shown += 1

    total = len(findings)
    if total > cap:
        lines.append(f"\n  ... and {total - cap:,} more finding(s)")


def _format_finding_counts(finding_counts: Counter[str], lines: list[str]) -> None:
    """Append per-check finding counts."""
    lines.append("")
    lines.append("  Findings by check:")
    for check, count in finding_counts.most_common():
        lines.append(f"    {check:<40s} {count:>6,}")


def _format_labels(
    label_histogram: Counter[str],
    max_labels: int,
    lines: list[str],
) -> None:
    """Append label histogram with shortened URIs."""
    total_labels = sum(label_histogram.values())
    num_types = len(label_histogram)
    lines.append("")
    lines.append(f"  Labels ({num_types:,} types, {total_labels:,} total):")
    for label, count in label_histogram.most_common(max_labels):
        short = _shorten_label(label)
        lines.append(f"    {short:<50s} {count:>6,}")
    if num_types > max_labels:
        lines.append(f"    ... and {num_types - max_labels} more label types")


def _shorten_label(label: str) -> str:
    """Shorten ontology URIs to their final identifier.

    ``http://example.com/ontology/a/FOO_000`` -> ``FOO_000``
    ``https://example.com/ontology/bar-001`` -> ``bar-001``
    Plain strings pass through unchanged.
    """
    if "/" in label:
        return label.rsplit("/", 1)[-1]
    return label


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------


class DatasetFormat(enum.Enum):
    """Supported dataset formats."""

    HMIE = "hmie"
    MOTCHALLENGE = "motchallenge"


class Severity(enum.Enum):
    """Severity level for a validation finding."""

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class Finding:
    """A single validation finding."""

    severity: Severity
    path: Path
    check: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON output."""
        return {
            "severity": self.severity.value,
            "path": str(self.path),
            "check": self.check,
            "message": self.message,
        }


@dataclass
class ValidationResult:
    """Outcome of validating one dataset.

    ``findings`` is the primary per-item detail list. At 60K datasets with
    many findings each, the list can grow large in memory, so callers can
    cap it via ``max_findings_per_check`` at the validate() level; when
    capped, ``finding_counts`` still reflects the true total count per
    check name so summary reporting stays accurate even under caps.
    """

    dataset_path: Path
    dataset_format: DatasetFormat
    passed: bool = True
    findings: list[Finding] = field(default_factory=list)
    label_histogram: Counter[str] = field(default_factory=Counter)
    finding_counts: Counter[str] = field(default_factory=Counter)
    # Per-(severity, check) counts — always uncapped. Used to derive
    # accurate summary / report totals when ``findings`` is capped by
    # max_findings_per_check.
    finding_severity_counts: dict[str, Counter[str]] = field(
        default_factory=lambda: {"error": Counter(), "warning": Counter()}
    )
    snippet_count: int = 0
    annotation_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    def __post_init__(self) -> None:
        # Backfill severity counts from findings when callers construct a
        # ValidationResult directly (tests, ad-hoc users) and haven't
        # populated finding_severity_counts. The happy path via validate()
        # populates this from the accumulator and does not need backfill.
        err = self.finding_severity_counts.get("error", Counter())
        warn = self.finding_severity_counts.get("warning", Counter())
        if not err and not warn and self.findings:
            derived: dict[str, Counter[str]] = {"error": Counter(), "warning": Counter()}
            for f in self.findings:
                derived[f.severity.value][f.check] += 1
            self.finding_severity_counts = derived

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == Severity.WARNING]

    def summary(
        self,
        *,
        show_findings: bool = True,
        max_findings: int | None = 20,
        max_labels: int = 15,
        use_color: bool | None = None,
    ) -> str:
        """Human-readable validation report.

        Parameters
        ----------
        show_findings
            If True, show individual findings grouped by file.
            If False, show only the summary counts (compact mode).
        max_findings
            Cap on individual findings shown. None = show all.
        max_labels
            Cap on label histogram entries.
        use_color
            If True, emit ANSI color codes. If False, emit plain text
            (no escapes). If None (default), auto-detect from stdout.
            Callers writing the summary to a file should pass ``False``
            so the saved report is plain ASCII, not peppered with
            ``\\033[...m`` sequences.
        """
        lines: list[str] = []
        # Totals come from the uncapped severity counts, not the
        # (possibly-capped) findings list. When max_findings_per_check
        # is set, self.findings only carries examples while
        # finding_severity_counts retains the real totals.
        error_count = sum(self.finding_severity_counts.get("error", Counter()).values())
        warn_count = sum(self.finding_severity_counts.get("warning", Counter()).values())

        # --- Header ---
        lines.append("")
        lines.append(f"  {_bold(str(self.dataset_path), use_color=use_color)}")
        lines.append("  " + "=" * 68)

        # --- Dataset overview ---
        if self.snippet_count or self.annotation_count:
            lines.append("")
            lines.append(f"  Snippets: {self.snippet_count:,}    Annotations: {self.annotation_count:,}")

        # --- Cache utilization ---
        if self.cache_hits or self.cache_misses:
            total = self.cache_hits + self.cache_misses
            lines.append(f"  Validated {total:,} pairs ({self.cache_hits:,} cached, {self.cache_misses:,} new)")

        # --- 4 requirement checks ---
        # Category mapping is HMIE-specific; lazy import keeps _types.py
        # neutral (no compile-time dependency on a particular format).
        from databridge._formats.hmie.categories import _CATEGORY_LABELS, _categorize_findings

        cats = _categorize_findings(self.finding_severity_counts)
        s_errs, _ = cats["structure"]
        c_errs, _ = cats["coverage"]
        struct_failed = s_errs > 0
        no_annotations = self.annotation_count == 0 and c_errs > 0
        lines.append("")
        for cat_key in ("structure", "video", "coverage", "scale_spec"):
            errs, warns = cats[cat_key]
            label = _CATEGORY_LABELS[cat_key]
            if struct_failed and cat_key != "structure":
                indicator = _dim("N/A", use_color=use_color)
                detail = _dim("requires passing structure check", use_color=use_color)
            elif no_annotations and cat_key in ("video", "scale_spec"):
                indicator = _dim("N/A", use_color=use_color)
                detail = _dim("no annotations to validate", use_color=use_color)
            else:
                indicator = _status_indicator(errs, warns, use_color=use_color)
                detail = _status_detail(errs, warns, use_color=use_color)
            lines.append(f"  {indicator:<18s} {label:<28s} {detail}")

        # --- Individual findings (verbose mode) ---
        if show_findings and self.findings:
            _format_findings(self.findings, self.dataset_path, max_findings, lines)

        # --- Finding summary by check ---
        if self.finding_counts:
            _format_finding_counts(self.finding_counts, lines)

        # --- Label histogram ---
        if self.label_histogram:
            _format_labels(self.label_histogram, max_labels, lines)

        # --- Result line (always last) ---
        status = _green("PASS", use_color=use_color) if self.passed else _red("FAIL", use_color=use_color)
        lines.append("")
        lines.append(f"  Result: {status}  ({error_count:,} errors, {warn_count:,} warnings)")
        lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full result to a plain dict for JSON output."""
        return {
            "dataset_path": str(self.dataset_path),
            "dataset_format": self.dataset_format.value,
            "passed": self.passed,
            "snippet_count": self.snippet_count,
            "annotation_count": self.annotation_count,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "finding_counts": dict(self.finding_counts),
            "finding_severity_counts": {sev: dict(counter) for sev, counter in self.finding_severity_counts.items()},
            "label_histogram": dict(self.label_histogram),
            "findings": [f.to_dict() for f in self.findings],
        }

    def to_jsonl(self) -> str:
        """One finding per line, newline-separated JSONL format.

        Convenient for streaming large results to a file or piping to
        `jq`/`grep`. Dataset-level metadata (pass/fail, counts) is
        emitted on the first line as a summary record.
        """
        import json

        summary_line = json.dumps(
            {
                "type": "summary",
                "dataset_path": str(self.dataset_path),
                "passed": self.passed,
                "snippet_count": self.snippet_count,
                "annotation_count": self.annotation_count,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "finding_counts": dict(self.finding_counts),
                "finding_severity_counts": {
                    sev: dict(counter) for sev, counter in self.finding_severity_counts.items()
                },
                "label_histogram": dict(self.label_histogram),
            }
        )
        finding_lines = [json.dumps({"type": "finding", **f.to_dict()}) for f in self.findings]
        return "\n".join([summary_line, *finding_lines])
