"""Tests for shared types."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from databridge._types import DatasetFormat, Finding, Severity, ValidationResult


class TestValidationResult:
    def test_default_passes(self) -> None:
        result = ValidationResult(dataset_path=Path("/data"), dataset_format=DatasetFormat.HMIE)
        assert result.passed is True
        assert result.errors == []
        assert result.warnings == []

    def test_with_error(self) -> None:
        finding = Finding(
            severity=Severity.ERROR,
            path=Path("/data/video.mp4"),
            check="video_integrity",
            message="Cannot open file",
        )
        result = ValidationResult(
            dataset_path=Path("/data"),
            dataset_format=DatasetFormat.HMIE,
            passed=False,
            findings=[finding],
        )
        assert result.passed is False
        assert len(result.errors) == 1
        assert len(result.warnings) == 0

    def test_summary_pass(self) -> None:
        result = ValidationResult(dataset_path=Path("/data"), dataset_format=DatasetFormat.HMIE)
        assert "Result: PASS" in result.summary()

    def test_default_label_histogram_empty(self) -> None:
        result = ValidationResult(dataset_path=Path("/data"), dataset_format=DatasetFormat.HMIE)
        assert result.label_histogram == Counter()

    def test_summary_without_labels_hides_section(self) -> None:
        result = ValidationResult(dataset_path=Path("/data"), dataset_format=DatasetFormat.HMIE)
        assert "Labels" not in result.summary()

    def test_summary_with_labels_shows_histogram(self) -> None:
        result = ValidationResult(
            dataset_path=Path("/data"),
            dataset_format=DatasetFormat.HMIE,
            label_histogram=Counter({"car": 10, "truck": 3}),
        )
        summary = result.summary()
        assert "Labels (" in summary
        assert "car" in summary
        assert "truck" in summary

    def test_finding_to_dict(self) -> None:
        finding = Finding(
            severity=Severity.ERROR,
            path=Path("/data/x.mp4"),
            check="video_open",
            message="bad",
        )
        d = finding.to_dict()
        assert d["severity"] == "error"
        assert d["path"] == "/data/x.mp4"
        assert d["check"] == "video_open"
        assert d["message"] == "bad"

    def test_result_to_dict(self) -> None:
        finding = Finding(
            severity=Severity.ERROR,
            path=Path("/data/bad.json"),
            check="annotation_schema",
            message="Missing task_id",
        )
        result = ValidationResult(
            dataset_path=Path("/data"),
            dataset_format=DatasetFormat.HMIE,
            passed=False,
            findings=[finding],
            label_histogram=Counter({"car": 3}),
            finding_counts=Counter({"annotation_schema": 1}),
        )
        d = result.to_dict()
        assert d["passed"] is False
        assert d["dataset_format"] == "hmie"
        assert d["finding_counts"] == {"annotation_schema": 1}
        assert d["label_histogram"] == {"car": 3}
        assert len(d["findings"]) == 1

    def test_result_to_jsonl(self) -> None:
        import json

        finding = Finding(
            severity=Severity.WARNING,
            path=Path("/data/x"),
            check="orphan_video",
            message="no ann",
        )
        result = ValidationResult(
            dataset_path=Path("/data"),
            dataset_format=DatasetFormat.HMIE,
            findings=[finding],
        )
        jsonl = result.to_jsonl()
        lines = jsonl.splitlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["type"] == "summary"
        second = json.loads(lines[1])
        assert second["type"] == "finding"
        assert second["check"] == "orphan_video"

    def test_summary_fail(self) -> None:
        finding = Finding(
            severity=Severity.ERROR,
            path=Path("/data/bad.json"),
            check="annotation_schema",
            message="Missing task_id",
        )
        result = ValidationResult(
            dataset_path=Path("/data"),
            dataset_format=DatasetFormat.HMIE,
            passed=False,
            findings=[finding],
        )
        summary = result.summary()
        assert "Result: FAIL" in summary
        assert "Missing task_id" in summary
        assert "error[annotation_schema]" in summary
