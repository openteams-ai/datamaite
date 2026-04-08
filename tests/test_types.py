"""Tests for shared types."""

from __future__ import annotations

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
        assert "[PASS]" in result.summary()

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
        assert "[FAIL]" in summary
        assert "Missing task_id" in summary
