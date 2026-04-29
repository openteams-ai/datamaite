"""Tests for the validation module."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from databridge._types import DatasetFormat, Severity
from databridge.validation import validate, validate_annotation


def _raise_in_worker(pair, *, check_video):  # type: ignore[no-untyped-def]
    """Module-level function that always raises.

    Must live at module scope so it's picklable -- ProcessPoolExecutor
    workers need to reconstruct the callable in the child process. A
    closure or nested function would fail with PicklingError and never
    exercise the real parallel worker-crash path.
    """
    msg = "deterministic worker-side failure"
    raise RuntimeError(msg)


class TestValidate:
    def test_returns_result(self, single_snippet_hmie: Path) -> None:
        result = validate(single_snippet_hmie, dataset_format=DatasetFormat.HMIE, check_video_integrity=False)
        assert result.dataset_path == single_snippet_hmie
        assert result.dataset_format == DatasetFormat.HMIE

    def test_accepts_string_path(self, single_snippet_hmie: Path) -> None:
        result = validate(str(single_snippet_hmie), dataset_format="hmie", check_video_integrity=False)
        assert result.dataset_path == single_snippet_hmie

    def test_accepts_string_format(self, single_snippet_hmie: Path) -> None:
        result = validate(single_snippet_hmie, dataset_format="hmie", check_video_integrity=False)
        assert result.dataset_format == DatasetFormat.HMIE

    def test_unsupported_format(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not a valid"):
            validate(tmp_path, dataset_format="unsupported")

    def test_nonexistent_path_fails(self, tmp_path: Path) -> None:
        result = validate(tmp_path / "nope")
        assert result.passed is False
        assert any(f.check == "path_exists" for f in result.findings)

    def test_file_not_dir_fails(self, tmp_path: Path) -> None:
        f = tmp_path / "file.txt"
        f.write_text("hi")
        result = validate(f)
        assert result.passed is False
        assert any(f.check == "path_is_dir" for f in result.findings)

    def test_empty_dir_fails(self, tmp_path: Path) -> None:
        result = validate(tmp_path)
        assert result.passed is False
        assert any(f.check == "discovery" for f in result.findings)

    def test_discovers_and_validates_snippets(self, single_snippet_hmie: Path) -> None:
        result = validate(single_snippet_hmie, check_video_integrity=False)
        # Should find and validate the annotation without errors
        errors = [f for f in result.findings if f.severity == Severity.ERROR]
        assert len(errors) == 0
        assert result.passed is True


class TestValidateAnnotation:
    def test_valid_annotation(self, valid_annotation: Path) -> None:
        result = validate_annotation(valid_annotation)
        assert result.passed is True
        errors = [f for f in result.findings if f.severity == Severity.ERROR]
        assert len(errors) == 0

    def test_invalid_annotation(self, invalid_annotation: Path) -> None:
        result = validate_annotation(invalid_annotation)
        assert result.passed is False

    def test_bad_json(self, bad_json: Path) -> None:
        result = validate_annotation(bad_json)
        assert result.passed is False

    def test_missing_video(self, valid_annotation: Path, tmp_path: Path) -> None:
        result = validate_annotation(valid_annotation, video_path=tmp_path / "missing.mp4")
        assert result.passed is False
        assert any(f.check == "video_missing" for f in result.findings)

    def test_skip_video(self, valid_annotation: Path) -> None:
        result = validate_annotation(valid_annotation, video_path=None, check_video_integrity=False)
        assert result.passed is True


class TestParallelWorkerCrash:
    def test_worker_side_exception_surfaces_as_finding(self, single_snippet_hmie: Path, monkeypatch) -> None:
        """A real ProcessPoolExecutor worker crash must become a worker_crash
        finding on the correct pair, not raise and abort the whole run.

        The existing monkeypatch-based test only covers the serial path:
        replacing ``_validate_pair`` in the parent process has no effect
        on child processes spawned by ProcessPoolExecutor. This test
        swaps the top-level ``_safe_validate_pair`` symbol (picklable
        module-level callable) so the children actually import the
        raising function and exercise the parallel crash handler.
        """
        # Build a dataset with 2+ pairs so parallelism is meaningful.
        from tests._hmie_factory import FullVideoSpec, SnippetSpec, make_hmie_dataset

        root = make_hmie_dataset(
            single_snippet_hmie.parent / "parallel_crash",
            [
                FullVideoSpec(
                    name="v_000000",
                    snippets=[
                        SnippetSpec(name="v_000001"),
                        SnippetSpec(name="v_000002"),
                    ],
                )
            ],
        )

        from databridge import validation as validation_module

        monkeypatch.setattr(validation_module, "_safe_validate_pair", _raise_in_worker)

        result = validate(root, check_video_integrity=False, workers=2)

        # Run must not raise. The crash must surface as a worker_crash
        # finding with the correct pair path (NOT the "<unknown>"
        # fallback used when the pool-level path can't identify the pair).
        crash_findings = [f for f in result.findings if f.check == "worker_crash"]
        assert len(crash_findings) >= 1
        assert all("RuntimeError" in f.message for f in crash_findings)
        assert all("deterministic worker-side failure" in f.message for f in crash_findings)

        # The path on each crash finding must be one of the real
        # annotation paths we discovered, not "<unknown>".
        crash_paths = {f.path for f in crash_findings}
        assert Path("<unknown>") not in crash_paths
        for cp in crash_paths:
            assert cp.name.startswith("CDAO_")

        # Overall passed must be False -- worker crashes are ERROR-level.
        assert result.passed is False

    def test_worker_side_exception_labels_stay_empty(self, single_snippet_hmie: Path, monkeypatch) -> None:
        """When a worker crashes, the label histogram must not inherit
        stale/partial data from the failed pair. Counter() is the correct
        placeholder."""
        from tests._hmie_factory import FullVideoSpec, SnippetSpec, make_hmie_dataset

        # Need at least 2 pairs so the parallel branch is actually taken
        # (workers is capped to min(N, len(pairs)), so len(pairs)=1 falls
        # back to the serial for-loop).
        root = make_hmie_dataset(
            single_snippet_hmie.parent / "parallel_crash_labels",
            [
                FullVideoSpec(
                    name="v_000000",
                    snippets=[
                        SnippetSpec(name="v_000001"),
                        SnippetSpec(name="v_000002"),
                    ],
                )
            ],
        )

        from databridge import validation as validation_module

        monkeypatch.setattr(validation_module, "_safe_validate_pair", _raise_in_worker)

        result = validate(root, check_video_integrity=False, workers=2)
        # All labels come from successfully-validated pairs. If every
        # pair crashed, histogram must stay empty (not partial).
        assert result.label_histogram == Counter()
