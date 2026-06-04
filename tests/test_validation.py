"""Tests for the validation module."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from databridge._types import DatasetFormat, Severity
from databridge.validation import validate, validate_annotation, validate_batches


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

    def test_supported_but_unimplemented_format_raises(self, valid_annotation: Path) -> None:
        with pytest.raises(NotImplementedError, match="motchallenge"):
            validate_annotation(valid_annotation, dataset_format="motchallenge")


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


def _make_two_batch_root(tmp_path: Path, valid_annotation_text: str) -> Path:
    """Create a root with two sibling batch directories.

    Mirrors the ``single_snippet_hmie`` fixture layout so ``find_batch_roots``
    recognises each child as a batch (snippet child with ``seq_*/`` container).
    """
    root = tmp_path / "root_with_batches"
    root.mkdir()
    for batch_name in ("batch_a", "batch_b"):
        batch = root / batch_name
        batch.mkdir()
        snippet = batch / f"{batch_name}_000001"
        snippet.mkdir()
        labeler = snippet / "labeler_a"
        labeler.mkdir()
        (labeler / "CDAO_test.json").write_text(valid_annotation_text)
        (snippet / "seq_mp4").mkdir()
        (snippet / "seq_mp4" / f"{batch_name}_000001.mp4").write_bytes(b"fake mp4")
    return root


class TestValidateBatches:
    def test_discovers_under_root_and_yields_tuples(self, tmp_path: Path, valid_annotation: Path) -> None:
        root = _make_two_batch_root(tmp_path, valid_annotation.read_text())

        results = list(validate_batches(root, check_video_integrity=False))

        assert len(results) == 2
        paths = [p for p, _ in results]
        assert paths == sorted(paths), "order should match find_batch_roots (sorted)"
        for batch_path, result in results:
            assert batch_path.is_dir()
            assert result.dataset_path == batch_path
            assert result.dataset_format == DatasetFormat.HMIE

    def test_accepts_iterable_of_batch_paths(self, tmp_path: Path, valid_annotation: Path) -> None:
        root = _make_two_batch_root(tmp_path, valid_annotation.read_text())
        batches = [root / "batch_b", root / "batch_a"]  # caller-chosen order

        results = list(validate_batches(batches, check_video_integrity=False))

        assert [p for p, _ in results] == batches, "iterable order must be preserved"

    def test_returns_iterator_not_list(self, tmp_path: Path, valid_annotation: Path) -> None:
        """Streaming matters at 60K batches -- caller shouldn't have to wait
        for every batch before iterating."""
        import collections.abc

        root = _make_two_batch_root(tmp_path, valid_annotation.read_text())
        it = validate_batches(root, check_video_integrity=False)
        assert isinstance(it, collections.abc.Iterator)
        assert not isinstance(it, list)

    def test_raises_on_empty_root_discovery(self, tmp_path: Path) -> None:
        """A typo'd root path is the common error; raise loudly rather than
        silently returning an empty iterator."""
        empty = tmp_path / "does_not_contain_batches"
        empty.mkdir()

        with pytest.raises(ValueError, match=r"[Nn]o batches"):
            list(validate_batches(empty))

    def test_empty_iterable_yields_nothing(self) -> None:
        """An explicit empty iterable is a legitimate input (caller did their
        own discovery and got nothing). Don't raise -- just yield nothing."""
        out = list(validate_batches([], check_video_integrity=False))
        assert out == []

    def test_crash_becomes_validate_crash_finding(
        self, tmp_path: Path, valid_annotation: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If validate() raises for one batch, the helper must yield a
        ValidationResult carrying a validate_crash ERROR finding, not
        propagate the exception and kill the loop."""
        root = _make_two_batch_root(tmp_path, valid_annotation.read_text())

        from databridge import validation as validation_module

        original = validation_module.validate
        bad_batch = root / "batch_a"

        def flaky_validate(path, *args, **kwargs):  # type: ignore[no-untyped-def]
            if Path(path) == bad_batch:
                msg = "boom"
                raise RuntimeError(msg)
            return original(path, *args, **kwargs)

        monkeypatch.setattr(validation_module, "validate", flaky_validate)

        results = list(validate_batches(root, check_video_integrity=False))

        assert len(results) == 2
        by_path = dict(results)
        crash_result = by_path[bad_batch]
        assert crash_result.passed is False
        crash_findings = [f for f in crash_result.findings if f.check == "validate_crash"]
        assert len(crash_findings) == 1
        assert crash_findings[0].severity == Severity.ERROR
        assert "RuntimeError" in crash_findings[0].message
        assert "boom" in crash_findings[0].message
        # The other batch must still have been validated normally.
        other = by_path[root / "batch_b"]
        assert not any(f.check == "validate_crash" for f in other.findings)

    def test_crash_result_summary_does_not_raise(
        self, tmp_path: Path, valid_annotation: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Rendering a validate_crash result via summary()/HTML/CLI text
        path must not blow up. _categorize_findings raises KeyError for
        check names missing from _CHECK_CATEGORIES; if validate_crash is
        not registered, the entire text/HTML output path explodes."""
        root = _make_two_batch_root(tmp_path, valid_annotation.read_text())

        from databridge import validation as validation_module

        bad_batch = root / "batch_a"

        def always_crash(path, *args, **kwargs):  # type: ignore[no-untyped-def]
            msg = "boom"
            raise RuntimeError(msg)

        monkeypatch.setattr(validation_module, "validate", always_crash)

        crash_result = next(r for _, r in validate_batches([bad_batch], check_video_integrity=False))
        # The bug surfaces here -- _categorize_findings via summary().
        text = crash_result.summary(use_color=False)
        assert "FAIL" in text

    def test_forwards_check_video_integrity_kwarg(
        self, tmp_path: Path, valid_annotation: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Kwargs must reach validate(); otherwise the helper silently
        strips caller intent (e.g. CHECK_VIDEO=False gets ignored)."""
        root = _make_two_batch_root(tmp_path, valid_annotation.read_text())

        seen: list[bool] = []
        from databridge import validation as validation_module

        original = validation_module.validate

        def spy(path, *args, **kwargs):  # type: ignore[no-untyped-def]
            seen.append(kwargs.get("check_video_integrity"))
            return original(path, *args, **kwargs)

        monkeypatch.setattr(validation_module, "validate", spy)

        list(validate_batches(root, check_video_integrity=False))
        assert seen == [False, False]
