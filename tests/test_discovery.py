"""Tests for HMIE folder discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from datamaite._formats.hmie.discovery import discover_hmie_pairs


@pytest.fixture
def hmie_tree_labeler(tmp_path: Path) -> Path:
    """Old-style layout: annotation in labeler subfolder.

    video_001_000000/
        video_001_000001/
            labeler_a/
                CDAO_SRC_video_001_000001.mp4_abc123.json
            seq_mp4/
                video_001_000001.mp4
        video_001_000002/
            labeler_a/
                CDAO_SRC_video_001_000002.mp4_def456.json
            seq_mp4/
                video_001_000002.mp4
    """
    root = tmp_path / "video_001_000000"
    root.mkdir()

    s1 = root / "video_001_000001"
    s1.mkdir()
    (s1 / "labeler_a").mkdir()
    (s1 / "labeler_a" / "CDAO_SRC_video_001_000001.mp4_abc123.json").write_text(
        '{"task_id": "t1", "response": {"annotations": {}}}'
    )
    (s1 / "seq_mp4").mkdir()
    (s1 / "seq_mp4" / "video_001_000001.mp4").write_bytes(b"fake mp4")

    s2 = root / "video_001_000002"
    s2.mkdir()
    (s2 / "labeler_a").mkdir()
    (s2 / "labeler_a" / "CDAO_SRC_video_001_000002.mp4_def456.json").write_text(
        '{"task_id": "t2", "response": {"annotations": {}}}'
    )
    (s2 / "seq_mp4").mkdir()
    (s2 / "seq_mp4" / "video_001_000002.mp4").write_bytes(b"fake mp4")

    return root


@pytest.fixture
def hmie_tree_scale(tmp_path: Path) -> Path:
    """Real-world layout: annotation in scale/ subdir, metadata at snippet level.

    CDAO_HMIE_BATCH-A/
        SRC1_100001_000002/
            SRC1_100001_000002.json           (video metadata, ignored)
            mapp_metadata/
                pipeline.json
            scale/
                HMI_TASK__abc123__SRC1_100001_000002.json
            seq_mp4/
                SRC1_000002.mp4
        SRC1_100001_000003/
            SRC1_100001_000003.json           (video metadata, ignored)
            scale/
                HMI_TASK__def456__SRC1_100001_000003.json
            seq_mp4/
                SRC1_000003.mp4
    """
    root = tmp_path / "CDAO_HMIE_BATCH-A"
    root.mkdir()

    s1 = root / "SRC1_100001_000002"
    s1.mkdir()
    (s1 / "SRC1_100001_000002.json").write_text('{"derivation_type": "fmv_sequence"}')
    (s1 / "mapp_metadata").mkdir()
    (s1 / "mapp_metadata" / "pipeline.json").write_text('{"meta": true}')
    (s1 / "scale").mkdir()
    (s1 / "scale" / "HMI_TASK__abc123__SRC1_100001_000002.json").write_text(
        '{"task_id": "t1", "response": {"annotations": {}}}'
    )
    (s1 / "seq_mp4").mkdir()
    (s1 / "seq_mp4" / "SRC1_000002.mp4").write_bytes(b"fake mp4")

    s2 = root / "SRC1_100001_000003"
    s2.mkdir()
    (s2 / "SRC1_100001_000003.json").write_text('{"derivation_type": "fmv_sequence"}')
    (s2 / "scale").mkdir()
    (s2 / "scale" / "HMI_TASK__def456__SRC1_100001_000003.json").write_text(
        '{"task_id": "t2", "response": {"annotations": {}}}'
    )
    (s2 / "seq_mp4").mkdir()
    (s2 / "seq_mp4" / "SRC1_000003.mp4").write_bytes(b"fake mp4")

    return root


@pytest.fixture
def hmie_tree_unannotated(tmp_path: Path) -> Path:
    """Unannotated dataset: video metadata at snippet level, no scale/ subdir.

    CDAO_HMIE_BATCH-B/
        SRC2_100002_000003/
            SRC2_100002_000003.json      (video metadata only)
            mapp_metadata/
                pipeline.json
            seq_mp4/
                SRC2_000003.mp4
    """
    root = tmp_path / "CDAO_HMIE_BATCH-B"
    root.mkdir()

    s1 = root / "SRC2_100002_000003"
    s1.mkdir()
    (s1 / "SRC2_100002_000003.json").write_text('{"derivation_type": "fmv_sequence"}')
    (s1 / "mapp_metadata").mkdir()
    (s1 / "mapp_metadata" / "pipeline.json").write_text('{"meta": true}')
    (s1 / "seq_mp4").mkdir()
    (s1 / "seq_mp4" / "SRC2_000003.mp4").write_bytes(b"fake mp4")

    return root


@pytest.fixture
def hmie_tree_seq_ts(tmp_path: Path) -> Path:
    """Layout with both seq_mp4/ and seq_ts/ variants.

    CDAO_HMIE_BATCH-C/
        SRC3_100003_000000/
            SRC3_100003_000000.json          (video metadata, ignored)
            scale/
                CDAO_HMIE_ANN.json
            seq_mp4/
                SRC3_000000.mp4
            seq_ts/
                SRC3_000000.ts
    """
    root = tmp_path / "CDAO_HMIE_BATCH-C"
    root.mkdir()

    s1 = root / "SRC3_100003_000000"
    s1.mkdir()
    (s1 / "SRC3_100003_000000.json").write_text('{"derivation_type": "fmv_sequence"}')
    (s1 / "scale").mkdir()
    (s1 / "scale" / "CDAO_HMIE_ANN.json").write_text('{"task_id": "t1", "response": {"annotations": {}}}')
    (s1 / "seq_mp4").mkdir()
    (s1 / "seq_mp4" / "SRC3_000000.mp4").write_bytes(b"fake mp4")
    (s1 / "seq_ts").mkdir()
    (s1 / "seq_ts" / "SRC3_000000.ts").write_bytes(b"fake ts")

    return root


class TestDiscoverLabelerLayout:
    """Tests for the old labeler-subfolder layout."""

    def test_finds_pairs(self, hmie_tree_labeler: Path) -> None:
        result = discover_hmie_pairs(hmie_tree_labeler)
        assert len(result.errors) == 0
        assert len(result.pairs) == 2
        assert all(p.video_path is not None for p in result.pairs)
        assert all(p.video_path.suffix == ".mp4" for p in result.pairs if p.video_path)

    def test_annotation_paths(self, hmie_tree_labeler: Path) -> None:
        result = discover_hmie_pairs(hmie_tree_labeler)
        ann_names = sorted(p.annotation_path.name for p in result.pairs)
        assert ann_names == [
            "CDAO_SRC_video_001_000001.mp4_abc123.json",
            "CDAO_SRC_video_001_000002.mp4_def456.json",
        ]

    def test_no_orphans(self, hmie_tree_labeler: Path) -> None:
        result = discover_hmie_pairs(hmie_tree_labeler)
        assert len(result.orphan_annotations) == 0
        assert len(result.orphan_videos) == 0

    def test_orphan_annotation_when_no_video(self, hmie_tree_labeler: Path) -> None:
        (hmie_tree_labeler / "video_001_000001" / "seq_mp4" / "video_001_000001.mp4").unlink()
        result = discover_hmie_pairs(hmie_tree_labeler)
        assert len(result.orphan_annotations) == 1
        assert "000001" in result.orphan_annotations[0].name

    def test_orphan_video_when_no_annotation(self, hmie_tree_labeler: Path) -> None:
        extra_snippet = hmie_tree_labeler / "video_001_000099"
        extra_snippet.mkdir()
        (extra_snippet / "seq_mp4").mkdir()
        (extra_snippet / "seq_mp4" / "video_001_000099.mp4").write_bytes(b"fake")
        result = discover_hmie_pairs(hmie_tree_labeler)
        assert len(result.orphan_videos) == 1
        assert "000099" in result.orphan_videos[0].name


class TestDiscoverScaleLayout:
    """Tests for the real-world scale/ subdirectory layout."""

    def test_finds_pairs(self, hmie_tree_scale: Path) -> None:
        result = discover_hmie_pairs(hmie_tree_scale)
        assert len(result.errors) == 0
        assert len(result.pairs) == 2
        assert all(p.video_path is not None for p in result.pairs)

    def test_annotations_from_scale_subdir(self, hmie_tree_scale: Path) -> None:
        result = discover_hmie_pairs(hmie_tree_scale)
        for pair in result.pairs:
            assert pair.annotation_path.parent.name == "scale"

    def test_snippet_level_metadata_ignored(self, hmie_tree_scale: Path) -> None:
        """Snippet-level JSONs (video metadata) must not be treated as annotations."""
        result = discover_hmie_pairs(hmie_tree_scale)
        ann_names = {p.annotation_path.name for p in result.pairs}
        assert "SRC1_100001_000002.json" not in ann_names
        assert "SRC1_100001_000003.json" not in ann_names

    def test_metadata_dir_json_ignored(self, hmie_tree_scale: Path) -> None:
        """JSONs in *_metadata/ dirs must not be discovered."""
        result = discover_hmie_pairs(hmie_tree_scale)
        ann_parents = {p.annotation_path.parent.name for p in result.pairs}
        assert "mapp_metadata" not in ann_parents

    def test_no_orphans(self, hmie_tree_scale: Path) -> None:
        result = discover_hmie_pairs(hmie_tree_scale)
        assert len(result.orphan_annotations) == 0
        assert len(result.orphan_videos) == 0


class TestDiscoverUnannotated:
    """Tests for datasets with no annotations (only video metadata)."""

    def test_reports_no_annotations(self, hmie_tree_unannotated: Path) -> None:
        result = discover_hmie_pairs(hmie_tree_unannotated)
        assert len(result.pairs) == 0
        assert len(result.errors) == 1
        assert "No annotation files found" in result.errors[0]

    def test_snippet_level_json_not_treated_as_annotation(self, hmie_tree_unannotated: Path) -> None:
        result = discover_hmie_pairs(hmie_tree_unannotated)
        ann_names = {p.annotation_path.name for p in result.pairs}
        assert "SRC2_100002_000003.json" not in ann_names


class TestDiscoverBatchLevelScale:
    """Datasets that place annotations in <batch>/scale/ instead of
    per-snippet subdirectories (the prototype's headline layout). These
    are paired by matching the video filename embedded in each Scale
    annotation name against the videos discovered in the snippet dirs."""

    def test_batch_level_scale_pairs_by_embedded_filename(self, tmp_path: Path) -> None:
        import json

        root = tmp_path / "batch"
        root.mkdir()
        snippet = root / "snippet_001"
        (snippet / "seq_mp4").mkdir(parents=True)
        (snippet / "seq_mp4" / "clip_a.mp4").write_bytes(b"fake")
        (root / "scale").mkdir()
        # Annotation name embeds the video filename clip_a.mp4.
        ann = root / "scale" / "CDAO_SRC1_clip_a.mp4_hash.json"
        ann.write_text(json.dumps({"task_id": "t", "response": {"annotations": {}}}))

        result = discover_hmie_pairs(root)
        assert len(result.pairs) == 1
        assert result.pairs[0].annotation_path == ann
        assert result.pairs[0].video_path is not None
        assert result.pairs[0].video_path.name == "clip_a.mp4"
        assert not any("not yet supported" in e for e in result.errors)

    def test_batch_level_scale_annotation_without_video_is_orphan(self, tmp_path: Path) -> None:
        import json

        root = tmp_path / "batch"
        root.mkdir()
        snippet = root / "snippet_001"
        (snippet / "seq_mp4").mkdir(parents=True)
        (snippet / "seq_mp4" / "other.mp4").write_bytes(b"fake")
        (root / "scale").mkdir()
        ann = root / "scale" / "CDAO_SRC1_clip_a.mp4_hash.json"
        ann.write_text(json.dumps({"task_id": "t", "response": {"annotations": {}}}))

        result = discover_hmie_pairs(root)
        assert len(result.pairs) == 1
        assert result.pairs[0].video_path is None
        assert ann in result.orphan_annotations


class TestDiscoverSeqTs:
    """Tests for datasets with seq_ts/ alongside seq_mp4/."""

    def test_prefers_mp4_over_ts(self, hmie_tree_seq_ts: Path) -> None:
        result = discover_hmie_pairs(hmie_tree_seq_ts)
        assert len(result.pairs) == 1
        assert result.pairs[0].video_path is not None
        assert result.pairs[0].video_path.suffix == ".mp4"

    def test_falls_back_to_ts_when_no_mp4(self, hmie_tree_seq_ts: Path) -> None:
        import shutil

        shutil.rmtree(hmie_tree_seq_ts / "SRC3_100003_000000" / "seq_mp4")
        result = discover_hmie_pairs(hmie_tree_seq_ts)
        assert len(result.pairs) == 1
        assert result.pairs[0].video_path is not None
        assert result.pairs[0].video_path.suffix == ".ts"


class TestDiscoverMixed:
    """Tests for mixed layouts and edge cases."""

    def test_empty_directory(self, tmp_path: Path) -> None:
        root = tmp_path / "empty"
        root.mkdir()
        result = discover_hmie_pairs(root)
        assert len(result.pairs) == 0
        assert len(result.errors) == 1
        assert "No snippet directories" in result.errors[0]

    def test_not_a_directory(self, tmp_path: Path) -> None:
        f = tmp_path / "file.txt"
        f.write_text("hi")
        result = discover_hmie_pairs(f)
        assert len(result.errors) == 1
        assert "not a directory" in result.errors[0]

    def test_multi_video_in_seq_mp4_picks_first(self, hmie_tree_scale: Path) -> None:
        """seq_mp4/ with multiple mp4s should pick lexicographic first and flag extras."""
        s1_mp4 = hmie_tree_scale / "SRC1_100001_000002" / "seq_mp4"
        (s1_mp4 / "zzz_last.mp4").write_bytes(b"fake")

        result = discover_hmie_pairs(hmie_tree_scale)
        multi_paths = [p for p, _ in result.multi_video_dirs]
        assert s1_mp4 in multi_paths
        count = next(c for p, c in result.multi_video_dirs if p == s1_mp4)
        assert count == 2  # original + 1 extra

        # Lexicographic first: 'SRC1_000002.mp4' < 'zzz_last.mp4'
        pair_for_s1 = next(p for p in result.pairs if "000002" in str(p.annotation_path))
        assert pair_for_s1.video_path is not None
        assert pair_for_s1.video_path.name == "SRC1_000002.mp4"

    def test_ignores_video_outside_seq_dir(self, tmp_path: Path) -> None:
        """Video files not in a seq_* directory should not be discovered."""
        root = tmp_path / "dataset"
        root.mkdir()
        snippet = root / "snippet_001"
        snippet.mkdir()
        (snippet / "scale").mkdir()
        (snippet / "scale" / "ann.json").write_text('{"task_id": "t1"}')
        (snippet / "stray.mp4").write_bytes(b"fake")
        (snippet / "seq_mp4").mkdir()
        (snippet / "seq_mp4" / "clip.mp4").write_bytes(b"fake")

        result = discover_hmie_pairs(root)
        orphan_names = {p.name for p in result.orphan_videos}
        assert "stray.mp4" not in orphan_names

    def test_empty_seq_dir_still_identifies_snippet(self, tmp_path: Path) -> None:
        """seq_mp4/ exists but is empty -- snippet identified, annotation orphaned."""
        root = tmp_path / "dataset"
        root.mkdir()
        snippet = root / "snippet_001"
        snippet.mkdir()
        (snippet / "scale").mkdir()
        (snippet / "scale" / "ann.json").write_text('{"task_id": "t1"}')
        (snippet / "seq_mp4").mkdir()
        result = discover_hmie_pairs(root)
        assert len(result.pairs) == 1
        assert result.pairs[0].video_path is None
        assert len(result.orphan_annotations) == 1

    def test_deeply_nested_batches(self, tmp_path: Path) -> None:
        """Discovery works regardless of how deep snippets are nested."""
        root = tmp_path / "DATASET" / "BATCHES" / "BATCH-1"
        snippet = root / "ship_123_000000"
        snippet.mkdir(parents=True)
        (snippet / "scale").mkdir()
        (snippet / "scale" / "ann.json").write_text('{"task_id": "t1"}')
        (snippet / "seq_mp4").mkdir()
        (snippet / "seq_mp4" / "ship.mp4").write_bytes(b"fake")

        result = discover_hmie_pairs(tmp_path / "DATASET")
        assert len(result.pairs) == 1
        assert result.pairs[0].video_path is not None

    def test_ts_only_dataset(self, tmp_path: Path) -> None:
        """Dataset with only seq_ts/ (no seq_mp4/) still discovers pairs."""
        root = tmp_path / "ts_only"
        root.mkdir()
        snippet = root / "clip_000000"
        snippet.mkdir()
        (snippet / "scale").mkdir()
        (snippet / "scale" / "ann.json").write_text('{"task_id": "t1"}')
        (snippet / "seq_ts").mkdir()
        (snippet / "seq_ts" / "clip.ts").write_bytes(b"fake")

        result = discover_hmie_pairs(root)
        assert len(result.pairs) == 1
        assert result.pairs[0].video_path is not None
        assert result.pairs[0].video_path.suffix == ".ts"

    def test_multiple_labelers_same_snippet(self, tmp_path: Path) -> None:
        """Two annotation subdirs in same snippet both get paired."""
        root = tmp_path / "dataset"
        root.mkdir()
        snippet = root / "snippet_001"
        snippet.mkdir()
        (snippet / "labeler_alpha").mkdir()
        (snippet / "labeler_alpha" / "ann_alpha.json").write_text('{"task_id": "t1"}')
        (snippet / "labeler_beta").mkdir()
        (snippet / "labeler_beta" / "ann_beta.json").write_text('{"task_id": "t2"}')
        (snippet / "seq_mp4").mkdir()
        (snippet / "seq_mp4" / "clip.mp4").write_bytes(b"fake")

        result = discover_hmie_pairs(root)
        assert len(result.pairs) == 2
        ann_names = sorted(p.annotation_path.name for p in result.pairs)
        assert ann_names == ["ann_alpha.json", "ann_beta.json"]
        assert all(p.video_path is not None for p in result.pairs)

    def test_non_json_files_in_annotation_dir_ignored(self, tmp_path: Path) -> None:
        """Only .json files in annotation subdirs are collected."""
        root = tmp_path / "dataset"
        root.mkdir()
        snippet = root / "snippet_001"
        snippet.mkdir()
        (snippet / "scale").mkdir()
        (snippet / "scale" / "ann.json").write_text('{"task_id": "t1"}')
        (snippet / "scale" / "readme.txt").write_text("not json")
        (snippet / "scale" / "thumb.png").write_bytes(b"fake")
        (snippet / "seq_mp4").mkdir()
        (snippet / "seq_mp4" / "clip.mp4").write_bytes(b"fake")

        result = discover_hmie_pairs(root)
        assert len(result.pairs) == 1
        assert result.pairs[0].annotation_path.name == "ann.json"


class TestBatchScaleMerge:
    """Per-batch scale/ discovery that MERGES with per-snippet discovery.

    Review fix: batch-level scale/ used to be an all-or-nothing root-only
    fallback, so per-batch scale/ under a multi-batch parent (and trees mixing
    both layouts) were silently missed.
    """

    def test_per_batch_scale_found_under_multi_batch_parent(self, tmp_path: Path) -> None:
        import json

        # dataset_root/batch_a/scale/*.json + batch_a/snippet_1/seq_mp4/*.mp4
        root = tmp_path / "dataset_root"
        batch = root / "batch_a"
        snip = batch / "snippet_1"
        (snip / "seq_mp4").mkdir(parents=True)
        (snip / "seq_mp4" / "clip_a.mp4").write_bytes(b"x")
        (batch / "scale").mkdir(parents=True)
        ann = batch / "scale" / "CDAO_SRC1_clip_a.mp4_h.json"
        ann.write_text(json.dumps({"task_id": "t", "response": {"annotations": {}}}))

        # Point at the PARENT of the batch (the case the old fallback missed).
        result = discover_hmie_pairs(root)
        assert len(result.pairs) == 1
        assert result.pairs[0].video_path is not None
        assert result.pairs[0].video_path.name == "clip_a.mp4"

    def test_mixed_batch_scale_and_per_snippet_both_found(self, tmp_path: Path) -> None:
        import json

        batch = tmp_path / "batch"
        # snippet_1: annotation centralised in batch-level scale/
        s1 = batch / "snippet_1"
        (s1 / "seq_mp4").mkdir(parents=True)
        (s1 / "seq_mp4" / "clip_one.mp4").write_bytes(b"x")
        (batch / "scale").mkdir(parents=True)
        (batch / "scale" / "CDAO_clip_one.mp4_h.json").write_text(
            json.dumps({"task_id": "t1", "response": {"annotations": {}}})
        )
        # snippet_2: annotation in its own labeler subdir (per-snippet)
        s2 = batch / "snippet_2"
        (s2 / "seq_mp4").mkdir(parents=True)
        (s2 / "seq_mp4" / "clip_two.mp4").write_bytes(b"x")
        (s2 / "labeler").mkdir()
        (s2 / "labeler" / "CDAO_clip_two.mp4_h.json").write_text(
            json.dumps({"task_id": "t2", "response": {"annotations": {}}})
        )

        result = discover_hmie_pairs(batch)
        names = sorted(p.video_path.name for p in result.pairs if p.video_path)
        assert names == ["clip_one.mp4", "clip_two.mp4"]

    def test_non_annotation_json_in_scale_skipped(self, tmp_path: Path) -> None:
        import json

        batch = tmp_path / "batch"
        s1 = batch / "snippet_1"
        (s1 / "seq_mp4").mkdir(parents=True)
        (s1 / "seq_mp4" / "clip_a.mp4").write_bytes(b"x")
        (batch / "scale").mkdir(parents=True)
        (batch / "scale" / "metadata.json").write_text(json.dumps({"some": "metadata"}))
        (batch / "scale" / "CDAO_clip_a.mp4_h.json").write_text(
            json.dumps({"task_id": "t", "response": {"annotations": {}}})
        )

        result = discover_hmie_pairs(batch)
        assert len(result.pairs) == 1
        assert result.pairs[0].annotation_path.name == "CDAO_clip_a.mp4_h.json"

    def test_two_snippets_via_batch_scale_have_distinct_snippet_dirs(self, tmp_path: Path) -> None:
        import json

        batch = tmp_path / "batch"
        for n, clip in ((1, "clip_one"), (2, "clip_two")):
            s = batch / f"snippet_{n}"
            (s / "seq_mp4").mkdir(parents=True)
            (s / "seq_mp4" / f"{clip}.mp4").write_bytes(b"x")
        (batch / "scale").mkdir(parents=True)
        for clip, t in (("clip_one", "t1"), ("clip_two", "t2")):
            (batch / "scale" / f"CDAO_{clip}.mp4_h.json").write_text(
                json.dumps({"task_id": t, "response": {"annotations": {}}})
            )

        result = discover_hmie_pairs(batch)
        # snippet_count (validation) dedupes on snippet_dir -> must be 2, not 1.
        assert len({p.snippet_dir for p in result.pairs}) == 2


class TestMatchAnnotationAmbiguity:
    def test_duplicate_basename_is_ambiguous_orphan(self) -> None:
        from datamaite._formats.hmie.discovery import match_annotation_to_video

        videos = [Path("/a/snippet_1/seq_mp4/clip.mp4"), Path("/a/snippet_2/seq_mp4/clip.mp4")]
        # two distinct videos share the basename -> ambiguous -> None (orphan)
        assert match_annotation_to_video("CDAO_clip.mp4_h.json", videos) is None

    def test_longer_embedded_name_still_wins_uniquely(self) -> None:
        from datamaite._formats.hmie.discovery import match_annotation_to_video

        videos = [Path("/a/clip_a.mp4"), Path("/b/a.mp4")]
        # 'a.mp4' is an anchored suffix inside the embedded 'clip_a.mp4'; longest wins
        match = match_annotation_to_video("CDAO_clip_a.mp4_h.json", videos)
        assert match is not None
        assert match.name == "clip_a.mp4"
