"""Tests for HTML report generation."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from databridge._report import (
    _aggregate_batches,
    prepare_report_data,
    render_html_report,
    render_html_report_multi,
)
from databridge._types import DatasetFormat, Finding, Severity, ValidationResult


class TestPrepareReportData:
    def test_basic_structure(self, tmp_path: Path) -> None:
        result = ValidationResult(
            dataset_path=tmp_path / "dataset",
            dataset_format=DatasetFormat.HMIE,
            passed=True,
            snippet_count=10,
            annotation_count=8,
        )
        data = prepare_report_data(result)
        assert data["passed"] is True
        assert data["dataset_path"] == str(tmp_path / "dataset")
        assert data["snippet_count"] == 10
        assert data["annotation_count"] == 8
        assert "categories" in data
        assert len(data["categories"]) == 4

    def test_findings_grouped_by_file(self, tmp_path: Path) -> None:
        root = tmp_path / "dataset"
        result = ValidationResult(
            dataset_path=root,
            dataset_format=DatasetFormat.HMIE,
            passed=False,
            findings=[
                Finding(Severity.ERROR, root / "a/scale/ann.json", "annotation_schema", "bad field"),
                Finding(Severity.WARNING, root / "a/scale/ann.json", "annotation_missing_afr", "no AFR"),
                Finding(Severity.ERROR, root / "b/scale/ann.json", "annotation_schema", "bad field"),
            ],
            finding_counts=Counter({"annotation_schema": 2, "annotation_missing_afr": 1}),
        )
        data = prepare_report_data(result)
        groups = data["finding_groups"]
        assert len(groups) == 2
        assert groups[0]["count"] == 2
        assert groups[1]["count"] == 1

    def test_labels_shortened(self, tmp_path: Path) -> None:
        result = ValidationResult(
            dataset_path=tmp_path,
            dataset_format=DatasetFormat.HMIE,
            label_histogram=Counter(
                {
                    "http://example.com/ontology/a/FOO_000": 100,
                    "widget": 50,
                }
            ),
        )
        data = prepare_report_data(result)
        labels = data["labels"]
        assert labels[0]["name"] == "FOO_000"
        assert labels[0]["count"] == 100
        assert labels[1]["name"] == "widget"

    def test_category_statuses(self, tmp_path: Path) -> None:
        result = ValidationResult(
            dataset_path=tmp_path,
            dataset_format=DatasetFormat.HMIE,
            findings=[
                Finding(Severity.ERROR, tmp_path / "x.json", "annotation_schema", "err"),
            ],
            finding_counts=Counter({"annotation_schema": 1}),
        )
        data = prepare_report_data(result)
        cats = {c["key"]: c for c in data["categories"]}
        assert cats["scale_spec"]["status"] == "fail"
        assert cats["structure"]["status"] == "pass"


class TestRenderHtmlReport:
    def test_produces_valid_html(self, tmp_path: Path) -> None:
        result = ValidationResult(
            dataset_path=tmp_path / "test-dataset",
            dataset_format=DatasetFormat.HMIE,
            passed=True,
            snippet_count=5,
            annotation_count=5,
            label_histogram=Counter({"boat": 10}),
        )
        html_str = render_html_report(result)
        assert "<!DOCTYPE html>" in html_str
        assert "test-dataset" in html_str
        assert "PASS" in html_str
        assert "boat" in html_str

    def test_escapes_html_in_paths(self, tmp_path: Path) -> None:
        result = ValidationResult(
            dataset_path=tmp_path / "<script>alert(1)</script>",
            dataset_format=DatasetFormat.HMIE,
        )
        html_str = render_html_report(result)
        assert "<script>alert(1)</script>" not in html_str

    def test_fail_result(self, tmp_path: Path) -> None:
        result = ValidationResult(
            dataset_path=tmp_path,
            dataset_format=DatasetFormat.HMIE,
            passed=False,
            findings=[
                Finding(Severity.ERROR, tmp_path / "bad.json", "annotation_schema", "missing task_id"),
            ],
            finding_counts=Counter({"annotation_schema": 1}),
        )
        html_str = render_html_report(result)
        assert "FAIL" in html_str
        assert "annotation_schema" in html_str

    def test_data_json_embedded(self, tmp_path: Path) -> None:
        findings = [
            Finding(Severity.WARNING, tmp_path / f"file_{i}.json", "annotation_schema", f"msg {i}") for i in range(100)
        ]
        result = ValidationResult(
            dataset_path=tmp_path,
            dataset_format=DatasetFormat.HMIE,
            findings=findings,
            finding_counts=Counter({"annotation_schema": 100}),
        )
        html_str = render_html_report(result)
        assert "REPORT_DATA" in html_str
        assert "msg 99" in html_str

    def test_script_breakout_in_finding_message(self, tmp_path: Path) -> None:
        result = ValidationResult(
            dataset_path=tmp_path,
            dataset_format=DatasetFormat.HMIE,
            findings=[
                Finding(
                    Severity.ERROR,
                    tmp_path / "x.json",
                    "annotation_schema",
                    "</script><script>alert(1)</script>",
                ),
            ],
            finding_counts=Counter({"annotation_schema": 1}),
        )
        html_str = render_html_report(result)
        assert "</script><script>" not in html_str

    def test_double_quote_in_label_name(self, tmp_path: Path) -> None:
        result = ValidationResult(
            dataset_path=tmp_path,
            dataset_format=DatasetFormat.HMIE,
            label_histogram=Counter({'" onmouseover="alert(1)': 5}),
        )
        html_str = render_html_report(result)
        assert 'onmouseover="alert(1)"' not in html_str


class TestRenderMultiBatch:
    def test_multi_batch_produces_html(self, tmp_path: Path) -> None:
        results: list[tuple[Path, ValidationResult]] = [
            (
                tmp_path / "batch-1",
                ValidationResult(
                    dataset_path=tmp_path / "batch-1",
                    dataset_format=DatasetFormat.HMIE,
                    passed=True,
                    snippet_count=10,
                    annotation_count=10,
                ),
            ),
            (
                tmp_path / "batch-2",
                ValidationResult(
                    dataset_path=tmp_path / "batch-2",
                    dataset_format=DatasetFormat.HMIE,
                    passed=False,
                    snippet_count=5,
                    annotation_count=3,
                    findings=[
                        Finding(Severity.ERROR, tmp_path / "batch-2/x.json", "annotation_schema", "err"),
                    ],
                    finding_counts=Counter({"annotation_schema": 1}),
                ),
            ),
        ]
        html_str = render_html_report_multi(results, tmp_path)
        assert "<!DOCTYPE html>" in html_str
        assert "batch-1" in html_str
        assert "batch-2" in html_str

    def test_multi_batch_aggregates_counts(self, tmp_path: Path) -> None:
        results: list[tuple[Path, ValidationResult]] = [
            (
                tmp_path / "a",
                ValidationResult(
                    dataset_path=tmp_path / "a",
                    dataset_format=DatasetFormat.HMIE,
                    passed=True,
                    snippet_count=10,
                    annotation_count=8,
                ),
            ),
            (
                tmp_path / "b",
                ValidationResult(
                    dataset_path=tmp_path / "b",
                    dataset_format=DatasetFormat.HMIE,
                    passed=True,
                    snippet_count=5,
                    annotation_count=3,
                ),
            ),
        ]
        html_str = render_html_report_multi(results, tmp_path)
        assert "REPORT_DATA" in html_str
        assert '"is_multi": true' in html_str
        assert '"snippet_count": 15' in html_str
        assert '"annotation_count": 11' in html_str

    def test_multi_batch_aggregate_block_present(self, tmp_path: Path) -> None:
        results: list[tuple[Path, ValidationResult]] = [
            (
                tmp_path / "a",
                ValidationResult(
                    dataset_path=tmp_path / "a",
                    dataset_format=DatasetFormat.HMIE,
                    passed=False,
                    findings=[
                        Finding(Severity.WARNING, tmp_path / "a/x.json", "annotation_missing_afr", "no AFR"),
                    ],
                    finding_counts=Counter({"annotation_missing_afr": 1}),
                ),
            ),
            (
                tmp_path / "b",
                ValidationResult(
                    dataset_path=tmp_path / "b",
                    dataset_format=DatasetFormat.HMIE,
                    passed=False,
                    findings=[
                        Finding(Severity.ERROR, tmp_path / "b/y.json", "annotation_schema", "missing field"),
                    ],
                    finding_counts=Counter({"annotation_schema": 1}),
                ),
            ),
        ]
        html_str = render_html_report_multi(results, tmp_path)
        # aggregate block must be present in the JSON payload
        assert '"aggregate":' in html_str
        # both findings should be reachable from aggregate
        assert "annotation_missing_afr" in html_str
        assert "annotation_schema" in html_str
        # batch_name attached
        assert '"batch_name": "a"' in html_str
        assert '"batch_name": "b"' in html_str

    def test_multi_batch_all_pass(self, tmp_path: Path) -> None:
        results: list[tuple[Path, ValidationResult]] = [
            (
                tmp_path / "x",
                ValidationResult(
                    dataset_path=tmp_path / "x",
                    dataset_format=DatasetFormat.HMIE,
                    passed=True,
                ),
            ),
        ]
        html_str = render_html_report_multi(results, tmp_path)
        assert '"passed": true' in html_str
        assert '"passed_count": 1' in html_str


class TestAggregateBatches:
    def _batch(
        self,
        tmp_path: Path,
        name: str,
        *,
        passed: bool = True,
        findings: list[Finding] | None = None,
        finding_counts: Counter[str] | None = None,
        labels: Counter[str] | None = None,
        snippets: int = 0,
        annotations: int = 0,
        cache_hits: int = 0,
        cache_misses: int = 0,
    ) -> dict:
        result = ValidationResult(
            dataset_path=tmp_path / name,
            dataset_format=DatasetFormat.HMIE,
            passed=passed,
            findings=findings or [],
            finding_counts=finding_counts or Counter(),
            label_histogram=labels or Counter(),
            snippet_count=snippets,
            annotation_count=annotations,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
        )
        data = prepare_report_data(result)
        data["batch_name"] = name
        return data

    def test_sums_categories(self, tmp_path: Path) -> None:
        b1 = self._batch(
            tmp_path,
            "a",
            findings=[Finding(Severity.ERROR, tmp_path / "a/x.json", "annotation_schema", "err")],
            finding_counts=Counter({"annotation_schema": 1}),
        )
        b2 = self._batch(
            tmp_path,
            "b",
            findings=[Finding(Severity.WARNING, tmp_path / "b/y.json", "annotation_missing_afr", "warn")],
            finding_counts=Counter({"annotation_missing_afr": 1}),
        )
        agg = _aggregate_batches([b1, b2])
        cats = {c["key"]: c for c in agg["categories"]}
        assert cats["scale_spec"]["errors"] == 1
        assert cats["scale_spec"]["warnings"] == 1
        assert cats["scale_spec"]["status"] == "fail"

    def test_prefixes_finding_paths_with_batch_name(self, tmp_path: Path) -> None:
        b1 = self._batch(
            tmp_path,
            "alpha",
            findings=[Finding(Severity.ERROR, tmp_path / "alpha/x.json", "annotation_schema", "err")],
            finding_counts=Counter({"annotation_schema": 1}),
        )
        b2 = self._batch(
            tmp_path,
            "beta",
            findings=[Finding(Severity.ERROR, tmp_path / "beta/y.json", "annotation_schema", "err")],
            finding_counts=Counter({"annotation_schema": 1}),
        )
        agg = _aggregate_batches([b1, b2])
        paths = [g["path"] for g in agg["finding_groups"]]
        assert all(p.startswith(("alpha/", "beta/")) for p in paths)
        assert "alpha/x.json" in paths
        assert "beta/y.json" in paths

    def test_merges_label_histograms(self, tmp_path: Path) -> None:
        b1 = self._batch(tmp_path, "a", labels=Counter({"boat": 5, "car": 3}))
        b2 = self._batch(tmp_path, "b", labels=Counter({"boat": 7, "plane": 2}))
        agg = _aggregate_batches([b1, b2])
        labels = {lb["name"]: lb["count"] for lb in agg["labels"]}
        assert labels == {"boat": 12, "car": 3, "plane": 2}
        assert agg["max_label_count"] == 12

    def test_merges_finding_counts(self, tmp_path: Path) -> None:
        b1 = self._batch(
            tmp_path,
            "a",
            finding_counts=Counter({"annotation_schema": 3, "video_open": 1}),
        )
        b2 = self._batch(
            tmp_path,
            "b",
            finding_counts=Counter({"annotation_schema": 2}),
        )
        agg = _aggregate_batches([b1, b2])
        counts = {fc["check"]: fc["count"] for fc in agg["finding_counts"]}
        assert counts == {"annotation_schema": 5, "video_open": 1}

    def test_sums_stats(self, tmp_path: Path) -> None:
        b1 = self._batch(tmp_path, "a", snippets=10, annotations=8, cache_hits=2, cache_misses=8)
        b2 = self._batch(tmp_path, "b", snippets=5, annotations=3, cache_hits=1, cache_misses=4)
        agg = _aggregate_batches([b1, b2])
        assert agg["snippet_count"] == 15
        assert agg["annotation_count"] == 11
        assert agg["cache_hits"] == 3
        assert agg["cache_misses"] == 12

    def test_passed_only_when_all_pass(self, tmp_path: Path) -> None:
        b1 = self._batch(tmp_path, "a", passed=True)
        b2 = self._batch(tmp_path, "b", passed=False)
        assert _aggregate_batches([b1, b2])["passed"] is False
        assert _aggregate_batches([b1])["passed"] is True


class TestReportE2E:
    def test_html_report_from_validation(self, tmp_path: Path) -> None:
        from databridge.validation import validate
        from tests._hmie_factory import FullVideoSpec, SnippetSpec, make_hmie_dataset

        root = make_hmie_dataset(
            tmp_path / "hmie",
            [FullVideoSpec(name="v_000000", snippets=[SnippetSpec(name="v_000001")])],
        )
        result = validate(root, check_video_integrity=False, workers=1)
        html_str = render_html_report(result)
        assert "<!DOCTYPE html>" in html_str
        assert "REPORT_DATA" in html_str
        assert "<html" in html_str
        assert "</html>" in html_str

    def test_html_report_writes_to_file(self, tmp_path: Path) -> None:
        from databridge.validation import validate
        from tests._hmie_factory import FullVideoSpec, SnippetSpec, make_hmie_dataset

        root = make_hmie_dataset(
            tmp_path / "hmie",
            [FullVideoSpec(name="v_000000", snippets=[SnippetSpec(name="v_000001")])],
        )
        result = validate(root, check_video_integrity=False, workers=1)
        output = tmp_path / "report.html"
        output.write_text(render_html_report(result), encoding="utf-8")
        assert output.exists()
        assert output.stat().st_size > 1000
