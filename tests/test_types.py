"""Tests for shared types."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from datamaite._types import DatasetFormat, Finding, Severity, ValidationResult


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
        assert d["skipped_checks"] == []

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

    def test_skipped_checks_default_empty(self) -> None:
        result = ValidationResult(dataset_path=Path("/data"), dataset_format=DatasetFormat.HMIE)
        assert result.skipped_checks == set()
        assert result.passed is True  # skipped state never affects passed

    def test_skipped_checks_in_to_dict_sorted(self) -> None:
        result = ValidationResult(
            dataset_path=Path("/data"),
            dataset_format=DatasetFormat.HMIE,
            skipped_checks={"video_integrity", "video_annotation_consistency"},
        )
        d = result.to_dict()
        assert d["skipped_checks"] == ["video_annotation_consistency", "video_integrity"]
        assert d["passed"] is True

    def test_skipped_checks_in_to_jsonl_summary(self) -> None:
        import json

        result = ValidationResult(
            dataset_path=Path("/data"),
            dataset_format=DatasetFormat.HMIE,
            skipped_checks={"video_integrity"},
        )
        summary = json.loads(result.to_jsonl().splitlines()[0])
        assert summary["type"] == "summary"
        assert summary["skipped_checks"] == ["video_integrity"]

    def test_summary_shows_skipped_row_and_banner(self) -> None:
        result = ValidationResult(
            dataset_path=Path("/data"),
            dataset_format=DatasetFormat.HMIE,
            annotation_count=1,
            skipped_checks={"video_integrity", "video_annotation_consistency"},
        )
        summary = result.summary(use_color=False)
        assert "SKIPPED" in summary
        assert "FMV integrity" in summary
        assert "Video checks not run" in summary
        assert "Result: PASS" in summary

    def test_summary_no_skip_has_no_skipped_markers(self) -> None:
        result = ValidationResult(
            dataset_path=Path("/data"),
            dataset_format=DatasetFormat.HMIE,
            annotation_count=1,
        )
        summary = result.summary(use_color=False)
        assert "SKIPPED" not in summary
        assert "Video checks not run" not in summary

    def test_summary_skip_does_not_mask_video_warning(self) -> None:
        # A real video-category finding (e.g. multiple_videos_in_seq_mp4) still
        # runs with video checks off. The FMV row must read WARN, not SKIPPED,
        # so the finding is not hidden -- though the banner still flags that
        # integrity/consistency did not run.
        video_warn = Finding(
            severity=Severity.WARNING,
            path=Path("/data/snippet/seq_mp4"),
            check="multiple_videos_in_seq_mp4",
            message="2 video files; only the first is validated",
        )
        result = ValidationResult(
            dataset_path=Path("/data"),
            dataset_format=DatasetFormat.HMIE,
            annotation_count=1,
            findings=[video_warn],
            skipped_checks={"video_integrity", "video_annotation_consistency"},
        )
        summary = result.summary(use_color=False)
        fmv_line = next(line for line in summary.splitlines() if "FMV integrity" in line)
        assert "WARN" in fmv_line
        assert "SKIPPED" not in fmv_line
        # The banner still explains that integrity/consistency were not run.
        assert "Video checks not run" in summary

    def test_summary_skipped_beats_na_when_structure_fails(self) -> None:
        # When structure fails AND video is skipped, the FMV row must read
        # SKIPPED (by request), not N/A (structure-failed) -- proves precedence.
        structure_error = Finding(
            severity=Severity.ERROR,
            path=Path("/data"),
            check="discovery",
            message="no snippet dirs",
        )
        result = ValidationResult(
            dataset_path=Path("/data"),
            dataset_format=DatasetFormat.HMIE,
            findings=[structure_error],
            skipped_checks={"video_integrity", "video_annotation_consistency"},
        )
        summary = result.summary(use_color=False)
        # The FMV-integrity row shows SKIPPED, not N/A.
        assert any("SKIPPED" in line and "FMV integrity" in line for line in summary.splitlines())
        assert not any("N/A" in line and "FMV integrity" in line for line in summary.splitlines())

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
