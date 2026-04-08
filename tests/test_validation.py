"""Tests for the validation module."""

from __future__ import annotations

from pathlib import Path

import pytest

from databridge._types import DatasetFormat
from databridge.validation import validate


class TestValidate:
    def test_returns_result(self, tmp_path: Path) -> None:
        result = validate(tmp_path, format=DatasetFormat.HMIE)
        assert result.dataset_path == tmp_path
        assert result.dataset_format == DatasetFormat.HMIE

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        result = validate(str(tmp_path), format="hmie")
        assert result.dataset_path == tmp_path

    def test_accepts_string_format(self, tmp_path: Path) -> None:
        result = validate(tmp_path, format="hmie")
        assert result.dataset_format == DatasetFormat.HMIE

    def test_unsupported_format(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not a valid"):
            validate(tmp_path, format="unsupported")

    def test_skip_video_check(self, tmp_path: Path) -> None:
        result = validate(tmp_path, check_video_integrity=False)
        assert result is not None
