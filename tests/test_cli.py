"""Tests for the CLI entrypoint."""

from __future__ import annotations

from pathlib import Path

from databridge._cli import main


class TestCLI:
    def test_no_command_returns_1(self) -> None:
        assert main([]) == 1

    def test_validate_command(self, tmp_path: Path) -> None:
        result = main(["validate", str(tmp_path)])
        assert result == 0

    def test_validate_skip_video(self, tmp_path: Path) -> None:
        result = main(["validate", str(tmp_path), "--skip-video-check"])
        assert result == 0

    def test_validate_format_flag(self, tmp_path: Path) -> None:
        result = main(["validate", str(tmp_path), "--format", "hmie"])
        assert result == 0
