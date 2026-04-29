"""End-to-end tests for the HMIE validator.

Uses the HMIE factory to build realistic dataset trees in tmp_path and
exercises the full discover -> validate -> report pipeline.
"""

from __future__ import annotations

from pathlib import Path

from databridge._types import Severity
from databridge.validation import validate
from tests._hmie_factory import (
    AnnotationSpec,
    FullVideoSpec,
    SnippetSpec,
    TrackSpec,
    VideoSpec,
    default_happy_dataset,
    make_hmie_dataset,
    single_video_dataset,
)
from tests._scale_factory import default_frame, one_track_annotation


class TestHappyPath:
    def test_full_dataset_passes(self, tmp_path: Path) -> None:
        root = default_happy_dataset(tmp_path / "hmie")
        result = validate(root)
        errors = [f for f in result.findings if f.severity == Severity.ERROR]
        assert errors == [], f"Unexpected errors: {[f.message for f in errors]}"
        assert result.passed is True

    def test_summary_format(self, tmp_path: Path) -> None:
        root = default_happy_dataset(tmp_path / "hmie")
        result = validate(root)
        summary = result.summary()
        assert "Result: PASS" in summary
        assert str(root) in summary

    def test_multiple_snippets_discovered(self, tmp_path: Path) -> None:
        """Default happy dataset has 2 full videos with 2 snippets each = 4 pairs."""
        root = default_happy_dataset(tmp_path / "hmie")
        result = validate(root, check_video_integrity=False)
        # No errors, no orphan findings -> all 4 pairs matched
        orphans = [f for f in result.findings if "orphan" in f.check]
        assert orphans == []


class TestFindingCaps:
    def test_finding_counts_always_populated(self, tmp_path: Path) -> None:
        """finding_counts should be a complete histogram even when cap is off."""
        root = single_video_dataset(
            tmp_path / "hmie",
            [
                SnippetSpec(name="video_001_000001", include_video=False),
                SnippetSpec(name="video_001_000002", include_video=False),
                SnippetSpec(name="video_001_000003", include_video=False),
            ],
        )
        result = validate(root, check_video_integrity=False)
        # 3 orphan annotations, counts should equal 3
        assert result.finding_counts["orphan_annotation"] == 3

    def test_cap_keeps_counts_accurate_but_trims_findings(self, tmp_path: Path) -> None:
        """With max_findings_per_check=1, the findings list holds only the
        first of each check, but finding_counts still reflects the full total."""
        root = single_video_dataset(
            tmp_path / "hmie",
            [SnippetSpec(name=f"video_001_{i:06d}", include_video=False) for i in range(1, 6)],
        )
        result = validate(root, check_video_integrity=False, max_findings_per_check=1)
        # 5 orphan annotations total
        assert result.finding_counts["orphan_annotation"] == 5
        # But findings list only holds 1 of them
        orphan_findings_in_list = [f for f in result.findings if f.check == "orphan_annotation"]
        assert len(orphan_findings_in_list) == 1

    def test_cap_does_not_mask_pass_fail(self, tmp_path: Path) -> None:
        """Even when ERROR findings are capped out of .findings, passed=False."""
        root = single_video_dataset(
            tmp_path / "hmie",
            [SnippetSpec(name=f"video_001_{i:06d}", video=VideoSpec(corrupt=True)) for i in range(1, 6)],
        )
        result = validate(root, max_findings_per_check=1)
        assert result.passed is False
        assert result.finding_counts["video_open"] == 5

    def test_cap_keeps_severity_totals_accurate(self, tmp_path: Path) -> None:
        """finding_severity_counts must reflect uncapped totals, not the capped list."""
        root = single_video_dataset(
            tmp_path / "hmie",
            [SnippetSpec(name=f"video_001_{i:06d}", include_video=False) for i in range(1, 6)],
        )
        result = validate(root, check_video_integrity=False, max_findings_per_check=1)
        # findings list is capped to 1, but severity counts show the full 5 errors
        assert len(result.findings) == 1
        assert result.finding_severity_counts["error"]["orphan_annotation"] == 5
        # Summary text must report the true 5 errors, not the capped 1
        summary = result.summary(show_findings=False, use_color=False)
        assert "5 errors" in summary

    def test_cap_keeps_html_report_totals_accurate(self, tmp_path: Path) -> None:
        """prepare_report_data must report uncapped error/warning totals."""
        from databridge._report import prepare_report_data

        root = single_video_dataset(
            tmp_path / "hmie",
            [SnippetSpec(name=f"video_001_{i:06d}", include_video=False) for i in range(1, 6)],
        )
        result = validate(root, check_video_integrity=False, max_findings_per_check=1)
        report = prepare_report_data(result)
        assert report["error_count"] == 5, f"expected 5 uncapped errors, got {report['error_count']}"
        # Per-category (coverage) must also reflect the 5 uncapped errors
        coverage = next(c for c in report["categories"] if c["key"] == "coverage")
        assert coverage["errors"] == 5


class TestParallelValidation:
    def test_parallel_workers_produce_same_result_as_serial(self, tmp_path: Path) -> None:
        """Parallel validation must produce the same findings and histogram as serial."""
        root = default_happy_dataset(tmp_path / "hmie")
        serial = validate(root, check_video_integrity=False, workers=1)
        parallel = validate(root, check_video_integrity=False, workers=4)

        assert serial.passed == parallel.passed
        assert serial.label_histogram == parallel.label_histogram
        # Findings may not be in identical order in the parallel case, but
        # the set of (check, path) pairs must match.
        serial_set = {(f.check, str(f.path)) for f in serial.findings}
        parallel_set = {(f.check, str(f.path)) for f in parallel.findings}
        assert serial_set == parallel_set

    def test_parallel_detects_broken_video_among_good_ones(self, tmp_path: Path) -> None:
        """When one pair is bad, parallel validation must still catch it."""
        root = single_video_dataset(
            tmp_path / "hmie",
            [
                SnippetSpec(name="video_001_000001"),
                SnippetSpec(name="video_001_000002"),
                SnippetSpec(name="video_001_000003", video=VideoSpec(corrupt=True)),
                SnippetSpec(name="video_001_000004"),
            ],
        )
        result = validate(root, workers=4)
        assert result.passed is False
        assert any(f.check == "video_open" for f in result.findings)


class TestLabelHistogram:
    def test_histogram_aggregates_across_pairs(self, tmp_path: Path) -> None:
        """validate() should sum label counts from all snippets in the tree."""
        root = default_happy_dataset(tmp_path / "hmie")
        result = validate(root, check_video_integrity=False)
        # default_happy_dataset: 2 full-length videos x 2 snippets each = 4 pairs x 1 track = 4 "vehicle" labels
        assert result.label_histogram["vehicle"] == 4

    def test_summary_includes_labels(self, tmp_path: Path) -> None:
        root = default_happy_dataset(tmp_path / "hmie")
        result = validate(root, check_video_integrity=False)
        assert "Labels (" in result.summary()
        assert "vehicle" in result.summary()


class TestBrokenVideos:
    def test_corrupt_video_fails(self, tmp_path: Path) -> None:
        root = single_video_dataset(
            tmp_path / "hmie",
            [SnippetSpec(name="video_001_000001", video=VideoSpec(corrupt=True))],
        )
        result = validate(root)
        assert result.passed is False
        errors = [f for f in result.findings if f.severity == Severity.ERROR]
        assert any(f.check == "video_open" for f in errors)

    def test_one_bad_one_good(self, tmp_path: Path) -> None:
        root = single_video_dataset(
            tmp_path / "hmie",
            [
                SnippetSpec(name="video_001_000001"),  # good
                SnippetSpec(name="video_001_000002", video=VideoSpec(corrupt=True)),
            ],
        )
        result = validate(root)
        assert result.passed is False
        # Exactly one video_open error (from the corrupt one)
        video_errors = [f for f in result.findings if f.check == "video_open"]
        assert len(video_errors) == 1


class TestOrphans:
    def test_orphan_annotation_errors(self, tmp_path: Path) -> None:
        """Annotation with no matching video in seq_mp4/ -> ERROR.

        An annotation without a video is not ML-usable (you can't extract
        frames without the video), so the dataset fails. This matches
        validate_annotation()'s video_missing ERROR behavior.
        """
        root = single_video_dataset(
            tmp_path / "hmie",
            [SnippetSpec(name="video_001_000001", include_video=False)],
        )
        result = validate(root)
        orphans = [f for f in result.findings if f.check == "orphan_annotation"]
        assert len(orphans) == 1
        assert orphans[0].severity == Severity.ERROR
        assert result.passed is False

    def test_orphan_video_warns(self, tmp_path: Path) -> None:
        """Video with no matching CDAO annotation."""
        root = single_video_dataset(
            tmp_path / "hmie",
            [
                SnippetSpec(name="video_001_000001"),  # good pair
                SnippetSpec(name="video_001_000002", include_annotation=False),
            ],
        )
        result = validate(root)
        warnings = [f for f in result.findings if f.check == "orphan_video"]
        assert len(warnings) == 1


class TestBadAnnotations:
    def test_invalid_json_fails(self, tmp_path: Path) -> None:
        root = single_video_dataset(
            tmp_path / "hmie",
            [SnippetSpec(name="video_001_000001", annotation=AnnotationSpec(valid_json=False))],
        )
        result = validate(root)
        assert result.passed is False
        assert any(f.check == "annotation_json" for f in result.findings)

    def test_missing_task_id_fails(self, tmp_path: Path) -> None:
        root = single_video_dataset(
            tmp_path / "hmie",
            [SnippetSpec(name="video_001_000001", annotation=AnnotationSpec(include_task_id=False))],
        )
        result = validate(root)
        assert result.passed is False
        assert any(f.check == "annotation_schema" for f in result.findings)


def _write_hand_rolled_snippet(
    root: Path,
    annotation_data: dict,
    video_fps: float = 30.0,
    video_width: int = 320,
    video_height: int = 240,
    num_frames: int = 30,
    flat: bool = False,
) -> Path:
    """Build a minimal HMIE snippet with a hand-rolled annotation dict and a
    synthetic video. Intentionally bypasses _hmie_factory.make_annotation_dict
    so the test exercises the real Scale schema, not the factory's
    interpretation of it.

    If ``flat=True``, the video frames are all solid black -- useful for
    tests that need to trigger video_flat_frames.
    """
    import json

    import cv2
    import numpy as np

    snippet = root / "video_000000" / "video_000001"
    snippet.mkdir(parents=True)
    labeler = snippet / "labeler_a"
    labeler.mkdir()
    (labeler / "CDAO_test.json").write_text(json.dumps(annotation_data))
    (snippet / "seq_mp4").mkdir()
    video_path = snippet / "seq_mp4" / "clip.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (video_width, video_height))
    for i in range(num_frames):
        if flat:
            frame = np.zeros((video_height, video_width, 3), dtype=np.uint8)
        else:
            gradient = np.linspace(0, 255, video_width, dtype=np.uint8)
            row = np.roll(gradient, i * 5)
            single = np.tile(row, (video_height, 1))
            frame = np.stack([single, single, single], axis=-1)
        writer.write(frame)
    writer.release()
    return root


class TestConsistency:
    """These tests hand-roll the annotation JSON rather than going through
    _hmie_factory.make_annotation_dict. The factory and the validator share
    the ``key * fps / afr`` formula, so e2e tests that build fixtures via
    AnnotationSpec are effectively testing the factory against itself. Direct
    JSON construction exercises the real Scale schema and catches drift
    between the two.
    """

    def test_fps_mismatch_warns(self, tmp_path: Path) -> None:
        """Video FPS differs from annotation's declared FPS."""
        # Annotation declares video fps = 60; real video will be 30
        data = one_track_annotation(task_id="fps-mismatch", fps=60)
        _write_hand_rolled_snippet(tmp_path / "hmie", data, video_fps=30.0)
        result = validate(tmp_path / "hmie")
        assert any(f.check == "consistency_fps" for f in result.findings)

    def test_bbox_out_of_bounds_warns(self, tmp_path: Path) -> None:
        """A bbox extending well past the video frame must warn."""
        # Video is 100x100; bbox at (50, 50) 200x200 clips way past
        data = one_track_annotation(
            task_id="bbox-oob",
            frames=[default_frame(left=50, top=50, height=200, width=200)],
        )
        _write_hand_rolled_snippet(tmp_path / "hmie", data, video_fps=30.0, video_width=100, video_height=100)
        result = validate(tmp_path / "hmie")
        assert any(f.check == "consistency_bbox_bounds" for f in result.findings)


class TestWorkerCrashResilience:
    def test_worker_crash_becomes_finding_not_exception(self, tmp_path: Path, monkeypatch) -> None:
        """One bad pair must not kill the entire validation run.

        Monkeypatch _validate_pair to raise on a specific path. Build a
        dataset with multiple pairs. validate() must still return a
        ValidationResult (not raise) and must surface the crash as a
        worker_crash finding for the bad pair while still processing
        the others.
        """
        # Build a happy dataset with 3 snippets
        root = default_happy_dataset(tmp_path / "hmie")

        # Monkeypatch _validate_pair to raise on the first pair only
        from databridge import validation as validation_module

        original_validate_pair = validation_module._validate_pair
        crashed = {"count": 0}

        def flaky_validate_pair(ann_path, video_path, *, check_video):  # type: ignore[no-untyped-def]
            if crashed["count"] == 0:
                crashed["count"] += 1
                msg = "simulated worker crash"
                raise RuntimeError(msg)
            return original_validate_pair(ann_path, video_path, check_video=check_video)

        monkeypatch.setattr(validation_module, "_validate_pair", flaky_validate_pair)

        # Serial path (workers=1) -- must not raise
        result = validate(root, check_video_integrity=False, workers=1)

        # worker_crash finding must be present for the bad pair
        assert any(f.check == "worker_crash" for f in result.findings)
        crash = next(f for f in result.findings if f.check == "worker_crash")
        assert "RuntimeError" in crash.message
        assert "simulated worker crash" in crash.message

        # The surviving pairs must still have been validated
        # (default_happy_dataset produces 4 pairs across 2 full-length videos,
        # so 3 should complete cleanly despite the first crashing)
        assert crashed["count"] == 1
        # The dataset has a crash so overall validation failed, but that's
        # because a worker crashed, not because validation aborted.
        assert result.passed is False


class TestConsistencyNotSkippedOnDecodeErrors:
    def test_consistency_still_runs_when_flat_frames_error(self, tmp_path: Path) -> None:
        """A video with valid metadata but flat frames triggers video_flat_frames
        ERROR. We must still run the consistency cross-check -- the metadata is
        authoritative, and skipping it hides real annotation/video mismatches.

        Prior to the fix, any video ERROR would skip the consistency check and
        the consistency_fps warning below would never fire.
        """
        # Annotation reports fps=60, but the video is written at 30 -- the
        # consistency_fps warning should still fire even though the all-black
        # video triggers video_flat_frames ERROR.
        data = one_track_annotation(task_id="decode-err-consistency", fps=60)
        _write_hand_rolled_snippet(tmp_path / "hmie", data, video_fps=30.0, flat=True)

        result = validate(tmp_path / "hmie")
        # Video error IS present
        assert any(f.check == "video_flat_frames" for f in result.findings)
        # AND the consistency warning must still fire (previously it would not)
        assert any(f.check == "consistency_fps" for f in result.findings)


class TestFactoryMultiTrack:
    def test_factory_clamps_each_track_independently(self) -> None:
        """Multi-track AnnotationSpec with differing num_frames per track must
        produce valid fixtures where each track's keys stay within the video
        frame bounds based on that track's length -- not based on tracks[0].

        The previous buggy clamp used tracks[0].num_frames for all tracks,
        which could silently truncate later tracks or leave fixtures in an
        inconsistent state.
        """
        from tests._hmie_factory import AnnotationSpec, make_annotation_dict

        spec = AnnotationSpec(
            task_id="multi",
            afr=5.0,
            video_fps=30.0,
            tracks=[
                TrackSpec(label="car", num_frames=3),
                TrackSpec(label="truck", num_frames=4),
            ],
        )
        data = make_annotation_dict(spec, VideoSpec(num_frames=30, fps=30.0))
        tracks = data["response"]["annotations"]
        assert len(tracks) == 2
        assert len(tracks["track-uuid-000"]["frames"]) == 3
        assert len(tracks["track-uuid-001"]["frames"]) == 4


class TestMultiLabeler:
    def test_same_snippet_multiple_labelers(self, tmp_path: Path) -> None:
        """Two labelers on the same snippet produces two annotation files.

        The discovery should find both, and since they share the same seq_mp4/
        video, one will match and the other will be reported as having no
        matching video in its own labeler subdir structure.
        """
        root = tmp_path / "hmie"
        full_video = FullVideoSpec(
            name="video_001_000000",
            snippets=[
                SnippetSpec(
                    name="video_001_000001",
                    labeler="labeler_alpha",
                    source_designator="SRC1",
                    hash_suffix="aaa111",
                ),
            ],
        )
        make_hmie_dataset(root, [full_video])
        # Add a second labeler's annotation in the same snippet dir
        single_video_dataset(
            root,
            [
                SnippetSpec(
                    name="video_001_000001",
                    labeler="labeler_beta",
                    source_designator="SRC1",
                    hash_suffix="bbb222",
                    include_video=False,  # video already exists from alpha
                ),
            ],
        )
        result = validate(root)
        # Both labelers' annotations should be discovered
        # Both resolve to the same seq_mp4/*.mp4, so neither is orphaned
        orphan_ann = [f for f in result.findings if f.check == "orphan_annotation"]
        assert orphan_ann == []


class TestEmptyAndMissing:
    def test_empty_root_fails(self, tmp_path: Path) -> None:
        root = tmp_path / "empty"
        root.mkdir()
        result = validate(root)
        assert result.passed is False
        assert any(f.check == "discovery" for f in result.findings)

    def test_nonexistent_root_fails(self, tmp_path: Path) -> None:
        result = validate(tmp_path / "does_not_exist")
        assert result.passed is False
        assert any(f.check == "path_exists" for f in result.findings)
