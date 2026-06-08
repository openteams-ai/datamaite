"""Tests for the CLI entrypoint."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from databridge._cli import main


@pytest.fixture
def multi_batch_hmie(tmp_path: Path, valid_annotation: Path) -> Path:
    """Two sibling batch directories, each with one snippet.

    Shape:
        tmp_path/multi/
            batch-a/
                snippet-1/
                    labeler_a/CDAO_a.json
                    seq_mp4/clip.mp4
            batch-b/
                snippet-2/
                    labeler_a/CDAO_b.json
                    seq_mp4/clip.mp4
    """
    root = tmp_path / "multi"
    for batch_name, snippet_name, ann_name in [
        ("batch-a", "snippet-1", "CDAO_a.json"),
        ("batch-b", "snippet-2", "CDAO_b.json"),
    ]:
        snippet = root / batch_name / snippet_name
        labeler = snippet / "labeler_a"
        labeler.mkdir(parents=True)
        (labeler / ann_name).write_text(valid_annotation.read_text())
        (snippet / "seq_mp4").mkdir()
        (snippet / "seq_mp4" / "clip.mp4").write_bytes(b"fake mp4")
    return root


class TestCLI:
    def test_no_command_returns_1(self) -> None:
        assert main([]) == 1

    def test_validate_command_pass(self, single_snippet_hmie: Path) -> None:
        """Valid dataset with no findings -> exit 0."""
        result = main(["validate", str(single_snippet_hmie), "--skip-video-check"])
        assert result == 0

    def test_validate_skip_video(self, single_snippet_hmie: Path) -> None:
        result = main(["validate", str(single_snippet_hmie), "--skip-video-check"])
        assert result == 0

    def test_validate_format_flag(self, single_snippet_hmie: Path) -> None:
        result = main(["validate", str(single_snippet_hmie), "--format", "hmie", "--skip-video-check"])
        assert result == 0

    def test_validate_invalid_format_clean_error(self, single_snippet_hmie: Path, capsys) -> None:
        """--format xyz must argparse-exit (2) with a friendly error, not a traceback."""
        with pytest.raises(SystemExit) as exc:
            main(["validate", str(single_snippet_hmie), "--format", "not-a-real-format"])
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert "invalid choice" in err
        assert "not-a-real-format" in err
        # The underlying ValueError ("'not-a-real-format' is not a valid DatasetFormat")
        # must NOT leak to the user.
        assert "Traceback" not in err
        assert "DatasetFormat" not in err

    def test_validate_empty_dir_errors(self, tmp_path: Path) -> None:
        """Empty directory has an ERROR finding -> exit 2."""
        result = main(["validate", str(tmp_path)])
        assert result == 2

    def test_validate_json_output(self, single_snippet_hmie: Path, capsys) -> None:
        result = main(["validate", str(single_snippet_hmie), "--skip-video-check", "--json"])
        assert result == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["passed"] is True
        assert data["dataset_format"] == "hmie"
        assert "finding_counts" in data
        assert "findings" in data
        assert "label_histogram" in data

    def test_validate_jsonl_output(self, single_snippet_hmie: Path, capsys) -> None:
        result = main(["validate", str(single_snippet_hmie), "--skip-video-check", "--jsonl"])
        assert result == 0
        captured = capsys.readouterr()
        lines = captured.out.strip().splitlines()
        # First line is the summary record, followed by zero or more findings
        summary = json.loads(lines[0])
        assert summary["type"] == "summary"
        assert summary["passed"] is True

    def test_validate_json_and_jsonl_mutually_exclusive(self, single_snippet_hmie: Path) -> None:
        with pytest.raises(SystemExit):
            main(["validate", str(single_snippet_hmie), "--json", "--jsonl"])

    def test_json_with_output_stdout_is_pipe_clean(self, single_snippet_hmie: Path, tmp_path: Path, capsys) -> None:
        """--json + -o must leave stdout as pure JSON (status goes to stderr)."""
        out = tmp_path / "report.json"
        rc = main(["validate", str(single_snippet_hmie), "--skip-video-check", "--json", "-o", str(out)])
        assert rc == 0
        captured = capsys.readouterr()
        # stdout is parseable as a single JSON document with no trailing noise.
        data = json.loads(captured.out)
        assert data["dataset_format"] == "hmie"
        # The "Report written to ..." status line lives on stderr, not stdout.
        assert "Report written" not in captured.out
        assert "Report written" in captured.err

    def test_jsonl_with_output_stdout_is_pipe_clean(self, single_snippet_hmie: Path, tmp_path: Path, capsys) -> None:
        """--jsonl + -o: every stdout line must parse as JSON."""
        out = tmp_path / "report.txt"
        rc = main(["validate", str(single_snippet_hmie), "--skip-video-check", "--jsonl", "-o", str(out)])
        assert rc == 0
        captured = capsys.readouterr()
        for line in captured.out.strip().splitlines():
            json.loads(line)  # raises if any line is non-JSON
        assert "Report written" not in captured.out
        assert "Report written" in captured.err

    def test_output_txt_has_no_ansi(self, single_snippet_hmie: Path, tmp_path: Path, monkeypatch) -> None:
        """A .txt report written by -o must not contain ANSI escape codes.

        Previously, ``ValidationResult.summary()`` embedded ``\\033[...m``
        sequences whenever sys.stdout.isatty() was True, and the CLI
        wrote that same string to file. Regression: the CLI must strip
        color for file output.
        """
        # Simulate interactive terminal so summary() would normally use color.
        import sys

        class _FakeTTY:
            def isatty(self) -> bool:
                return True

            def __getattr__(self, name):
                return getattr(sys.__stdout__, name)

        monkeypatch.setattr(sys, "stdout", _FakeTTY())

        out = tmp_path / "report.txt"
        # Force some finding content so color helpers are exercised.
        rc = main(["validate", str(single_snippet_hmie), "--skip-video-check", "-o", str(out)])
        assert rc in (0, 1)
        content = out.read_bytes()
        assert b"\033" not in content
        assert b"\x1b" not in content

    def test_output_json_extension(self, single_snippet_hmie: Path, tmp_path: Path) -> None:
        """-o <file>.json writes a JSON dump."""
        out = tmp_path / "report.json"
        rc = main(["validate", str(single_snippet_hmie), "--skip-video-check", "-o", str(out)])
        assert rc == 0
        data = json.loads(out.read_text())
        assert data["passed"] is True
        assert data["dataset_format"] == "hmie"

    def test_output_html_extension(self, single_snippet_hmie: Path, tmp_path: Path) -> None:
        """-o <file>.html renders the HTML report."""
        out = tmp_path / "report.html"
        rc = main(["validate", str(single_snippet_hmie), "--skip-video-check", "-o", str(out)])
        assert rc == 0
        html = out.read_text()
        assert "<!DOCTYPE html>" in html
        assert "REPORT_DATA" in html

    def test_output_auto_html_naming(self, single_snippet_hmie: Path, tmp_path: Path, monkeypatch) -> None:
        """-o with no argument defaults to a timestamped HTML filename."""
        monkeypatch.chdir(tmp_path)
        rc = main(["validate", str(single_snippet_hmie), "--skip-video-check", "-o"])
        assert rc == 0
        html_reports = list(tmp_path.glob("databridge-report-*.html"))
        assert len(html_reports) == 1

    def test_no_cache_flag(self, single_snippet_hmie: Path) -> None:
        """--no-cache disables the cache entirely."""
        rc = main(["validate", str(single_snippet_hmie), "--skip-video-check", "--no-cache"])
        assert rc == 0

    def test_clean_flag(self, single_snippet_hmie: Path, tmp_path: Path, monkeypatch) -> None:
        """--clean wipes the cache before running."""
        # Redirect default cache location to a tmp path so the test doesn't
        # touch the user's home-directory cache.
        fake_home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(fake_home))
        rc = main(["validate", str(single_snippet_hmie), "--skip-video-check", "--clean"])
        assert rc == 0

    def test_workers_invalid_value_errors(self, single_snippet_hmie: Path) -> None:
        """--workers 0 is rejected by the parser."""
        with pytest.raises(SystemExit):
            main(["validate", str(single_snippet_hmie), "--workers", "0"])

    def test_workers_capped_to_64(self, single_snippet_hmie: Path) -> None:
        """--workers 9999 is silently clamped to 64 (no error)."""
        rc = main(["validate", str(single_snippet_hmie), "--skip-video-check", "--workers", "9999"])
        assert rc == 0

    def test_verbose_shows_findings(self, tmp_path: Path, capsys) -> None:
        """With -v the individual findings are shown."""
        # Empty dir produces at least one discovery ERROR -> exit 2 + findings.
        rc = main(["-v", "validate", str(tmp_path)])
        assert rc == 2
        out = capsys.readouterr().out
        # Verbose should include the per-finding error prefix
        assert "error[" in out or "Result: FAIL" in out

    def test_quiet_suppresses_progress(self, single_snippet_hmie: Path) -> None:
        """-q suppresses progress output but still prints the summary."""
        rc = main(["-q", "validate", str(single_snippet_hmie), "--skip-video-check"])
        assert rc == 0

    def test_debug_flag(self, single_snippet_hmie: Path) -> None:
        """--debug enables debug logging without affecting exit code."""
        rc = main(["--debug", "validate", str(single_snippet_hmie), "--skip-video-check"])
        assert rc == 0

    def test_max_findings_per_check_flag(self, single_snippet_hmie: Path) -> None:
        rc = main(["validate", str(single_snippet_hmie), "--skip-video-check", "--max-findings-per-check", "5"])
        assert rc == 0


class TestMultiBatchCLI:
    def test_multi_batch_discovery(self, multi_batch_hmie: Path) -> None:
        """Two sibling batch dirs -> the multi-batch code path is taken."""
        rc = main(["validate", str(multi_batch_hmie), "--skip-video-check"])
        # Valid fixtures -> exit 0
        assert rc == 0

    def test_multi_batch_json(self, multi_batch_hmie: Path, capsys) -> None:
        """Multi-batch + --json emits a JSON array of per-batch results."""
        rc = main(["validate", str(multi_batch_hmie), "--skip-video-check", "--json"])
        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert isinstance(data, list)
        assert len(data) == 2

    def test_multi_batch_jsonl(self, multi_batch_hmie: Path, capsys) -> None:
        """Multi-batch + --jsonl emits per-batch JSONL."""
        rc = main(["validate", str(multi_batch_hmie), "--skip-video-check", "--jsonl"])
        assert rc == 0
        out = capsys.readouterr().out
        assert out.count('"type": "summary"') == 2

    def test_multi_batch_html_output(self, multi_batch_hmie: Path, tmp_path: Path) -> None:
        out = tmp_path / "multi.html"
        rc = main(["validate", str(multi_batch_hmie), "--skip-video-check", "-o", str(out)])
        assert rc == 0
        html = out.read_text()
        assert "<!DOCTYPE html>" in html
        assert '"is_multi": true' in html

    def test_multi_batch_json_output(self, multi_batch_hmie: Path, tmp_path: Path) -> None:
        out = tmp_path / "multi.json"
        rc = main(["validate", str(multi_batch_hmie), "--skip-video-check", "-o", str(out)])
        assert rc == 0
        data = json.loads(out.read_text())
        assert isinstance(data, list)
        assert len(data) == 2

    def test_multi_batch_txt_output_no_ansi(self, multi_batch_hmie: Path, tmp_path: Path, monkeypatch) -> None:
        """Multi-batch .txt report must also strip ANSI escapes."""
        import sys

        class _FakeTTY:
            def isatty(self) -> bool:
                return True

            def __getattr__(self, name):
                return getattr(sys.__stdout__, name)

        monkeypatch.setattr(sys, "stdout", _FakeTTY())
        out = tmp_path / "multi.txt"
        rc = main(["validate", str(multi_batch_hmie), "--skip-video-check", "-o", str(out)])
        assert rc == 0
        assert b"\033" not in out.read_bytes()

    def test_multi_batch_failure_exit_code(self, multi_batch_hmie: Path) -> None:
        """If any batch fails, exit code is 2."""
        # Make one batch invalid: remove seq_mp4 -> no snippet detected
        # Actually a safer way: corrupt an annotation
        bad = multi_batch_hmie / "batch-a" / "snippet-1" / "labeler_a" / "CDAO_a.json"
        bad.write_text("not json")
        rc = main(["validate", str(multi_batch_hmie), "--skip-video-check"])
        assert rc == 2

    def test_multi_batch_one_crashing_validate_does_not_kill_run(
        self, multi_batch_hmie: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        """A per-batch crash in validate() must surface as a validate_crash
        finding on that batch (exit 2) and must not prevent the sibling
        batch from being validated.

        Without crash isolation in the multi-batch CLI path, a single
        blown-up batch aborts the entire run -- the same pathology the
        notebook's try/except was working around. Pinning this here so
        the CLI stays consistent with validate_batches().
        """
        from databridge import validation as validation_module

        bad_batch = multi_batch_hmie / "batch-a"
        original = validation_module.validate

        def flaky(path, *args, **kwargs):  # type: ignore[no-untyped-def]
            if Path(path) == bad_batch:
                msg = "boom"
                raise RuntimeError(msg)
            return original(path, *args, **kwargs)

        monkeypatch.setattr(validation_module, "validate", flaky)

        rc = main(["validate", str(multi_batch_hmie), "--skip-video-check", "--json"])

        assert rc == 2, "crash in one batch must propagate as exit 2, not raise"
        data = json.loads(capsys.readouterr().out)
        assert isinstance(data, list)
        assert len(data) == 2, "both batches must appear in the report"

        by_name = {Path(d["dataset_path"]).name: d for d in data}
        crash_batch = by_name["batch-a"]
        assert any(f["check"] == "validate_crash" for f in crash_batch["findings"])
        assert crash_batch["passed"] is False

        # Sibling batch must have been validated normally.
        clean_batch = by_name["batch-b"]
        assert not any(f["check"] == "validate_crash" for f in clean_batch["findings"])


class TestFindBatchDirs:
    def test_single_batch_returns_self(self, single_snippet_hmie: Path) -> None:
        from databridge._cli import _find_batch_dirs

        result = _find_batch_dirs(single_snippet_hmie)
        assert result == [single_snippet_hmie]

    def test_multi_batch_returns_children(self, multi_batch_hmie: Path) -> None:
        from databridge._cli import _find_batch_dirs

        result = _find_batch_dirs(multi_batch_hmie)
        names = sorted(d.name for d in result)
        assert names == ["batch-a", "batch-b"]

    def test_nonexistent_returns_empty(self, tmp_path: Path) -> None:
        from databridge._cli import _find_batch_dirs

        result = _find_batch_dirs(tmp_path / "missing")
        assert result == []

    def test_empty_directory_returns_empty(self, tmp_path: Path) -> None:
        from databridge._cli import _find_batch_dirs

        empty = tmp_path / "empty"
        empty.mkdir()
        result = _find_batch_dirs(empty)
        assert result == []


class TestHelpers:
    def test_rpad_strips_ansi_for_width(self) -> None:
        from databridge._cli import _rpad

        # ANSI-wrapped text should be padded to visible width, not byte length
        text = "\033[32mPASS\033[0m"  # visible length 4
        padded = _rpad(text, 10)
        # 10 - 4 = 6 leading spaces, then the raw ansi text
        assert padded.startswith("      ")
        assert padded.endswith(text)

    def test_show_progress_quiet_returns_false(self) -> None:
        import argparse

        from databridge._cli import _show_progress

        args = argparse.Namespace(quiet=True)
        assert _show_progress(args) is False

    def test_make_status_callback_writes_when_show(self, capsys) -> None:
        from databridge._cli import _make_status_callback

        cb = _make_status_callback(show=True)
        cb("hello")
        err = capsys.readouterr().err
        assert "hello" in err

    def test_make_status_callback_noop_when_not_show(self, capsys) -> None:
        from databridge._cli import _make_status_callback

        cb = _make_status_callback(show=False)
        cb("should be silent")
        err = capsys.readouterr().err
        assert err == ""

    def test_multi_exit_code_all_passed(self, tmp_path: Path) -> None:
        from databridge._cli import _multi_exit_code
        from databridge._types import DatasetFormat, ValidationResult

        results = [
            (
                tmp_path / "a",
                ValidationResult(dataset_path=tmp_path / "a", dataset_format=DatasetFormat.HMIE, passed=True),
            ),
        ]
        assert _multi_exit_code(results) == 0

    def test_multi_exit_code_one_failed(self, tmp_path: Path) -> None:
        from databridge._cli import _multi_exit_code
        from databridge._types import DatasetFormat, ValidationResult

        results = [
            (
                tmp_path / "a",
                ValidationResult(dataset_path=tmp_path / "a", dataset_format=DatasetFormat.HMIE, passed=True),
            ),
            (
                tmp_path / "b",
                ValidationResult(dataset_path=tmp_path / "b", dataset_format=DatasetFormat.HMIE, passed=False),
            ),
        ]
        assert _multi_exit_code(results) == 2

    def test_multi_exit_code_warnings_only(self, tmp_path: Path) -> None:
        """Warnings without errors must return 1, matching single-batch semantics."""
        from databridge._cli import _multi_exit_code
        from databridge._types import DatasetFormat, Finding, Severity, ValidationResult

        warning_finding = Finding(
            severity=Severity.WARNING,
            path=tmp_path / "b" / "snippet.json",
            check="some_warning",
            message="not fatal",
        )
        results = [
            (
                tmp_path / "a",
                ValidationResult(dataset_path=tmp_path / "a", dataset_format=DatasetFormat.HMIE, passed=True),
            ),
            (
                tmp_path / "b",
                ValidationResult(
                    dataset_path=tmp_path / "b",
                    dataset_format=DatasetFormat.HMIE,
                    passed=True,
                    findings=[warning_finding],
                ),
            ),
        ]
        assert _multi_exit_code(results) == 1

    def test_multi_batch_status_line_format(self, multi_batch_hmie: Path, capsys) -> None:
        """Status line reports clean / warnings / failed counts, not 'fully compliant'."""
        rc = main(["validate", str(multi_batch_hmie), "--skip-video-check"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "clean" in out
        assert "warnings" in out
        assert "failed" in out
        assert "fully compliant" not in out

    def test_multi_batch_table_video_column_skipped(self, multi_batch_hmie: Path, capsys) -> None:
        """--skip-video-check: the batch table Video column shows SKIP, not a misleading PASS."""
        rc = main(["validate", str(multi_batch_hmie), "--skip-video-check"])
        assert rc == 0
        out = capsys.readouterr().out
        # One SKIP per batch row in the Video column (the fixture has 2 batches).
        assert out.count("SKIP") >= 2

    def test_batch_table_skip_does_not_mask_video_warning(self, tmp_path: Path, capsys) -> None:
        """A video-category WARNING that still fires with video checks off must
        read WARN in the table, not SKIP -- SKIP must not mask real findings."""
        from collections import Counter

        from databridge import DatasetFormat, ValidationResult
        from databridge._cli import _print_batch_table

        result = ValidationResult(
            dataset_path=tmp_path / "batch-a",
            dataset_format=DatasetFormat.HMIE,
            snippet_count=1,
            annotation_count=1,
            finding_severity_counts={"error": Counter(), "warning": Counter({"multiple_videos_in_seq_mp4": 1})},
            skipped_checks={"video_integrity", "video_annotation_consistency"},
        )
        _print_batch_table([(tmp_path / "batch-a", result)])
        out = capsys.readouterr().out
        assert "WARN" in out
        assert "SKIP" not in out

    def test_make_batch_progress_increments(self, capsys) -> None:
        from databridge._cli import _make_batch_progress

        cb = _make_batch_progress(1, 3, "batch-a", show=True)
        cb()
        cb()
        err = capsys.readouterr().err
        # The callback writes a progress string that includes pair count
        assert "2" in err

    def test_print_batch_table_smoke(self, tmp_path: Path, capsys) -> None:
        """_print_batch_table must not crash on any row shape (PASS/struct_fail/no_annotations)."""
        from collections import Counter

        from databridge._cli import _print_batch_table
        from databridge._types import DatasetFormat, Finding, Severity, ValidationResult

        passing = ValidationResult(
            dataset_path=tmp_path / "a",
            dataset_format=DatasetFormat.HMIE,
            passed=True,
            snippet_count=5,
            annotation_count=5,
        )
        struct_fail = ValidationResult(
            dataset_path=tmp_path / "b",
            dataset_format=DatasetFormat.HMIE,
            passed=False,
            findings=[
                Finding(Severity.ERROR, tmp_path / "b", "discovery", "no snippet dirs"),
            ],
            finding_counts=Counter({"discovery": 1}),
        )
        no_anns = ValidationResult(
            dataset_path=tmp_path / "c",
            dataset_format=DatasetFormat.HMIE,
            passed=False,
            findings=[
                Finding(Severity.ERROR, tmp_path / "c", "no_annotations", "no ann"),
            ],
            finding_counts=Counter({"no_annotations": 1}),
            snippet_count=2,
            annotation_count=0,
        )
        _print_batch_table(
            [
                (tmp_path / "a", passing),
                (tmp_path / "b", struct_fail),
                (tmp_path / "c", no_anns),
            ]
        )
        out = capsys.readouterr().out
        assert "Batch" in out
        assert "a" in out
        assert "b" in out
        assert "c" in out
