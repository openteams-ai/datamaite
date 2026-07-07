"""Tests for the HMIE dataloader (datamaite.load_hmie)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datamaite import BoxAnnotation, BoxTrackDataset, VideoSequence
from datamaite._formats.hmie.loader import load_hmie

from ._hmie_factory import (
    AnnotationSpec,
    SnippetSpec,
    TrackSpec,
    default_happy_dataset,
    make_annotation_dict,
    make_video,
    single_video_dataset,
)
from ._hmie_factory import (
    VideoSpec as FactoryVideoSpec,
)


def _write_annotation(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


class TestDefaultLoad:
    def test_loads_all_snippets(self, tmp_path: Path) -> None:
        default_happy_dataset(tmp_path)
        ds = load_hmie(tmp_path)

        assert isinstance(ds, BoxTrackDataset)
        # 2 full-length videos x 2 snippets each
        assert len(ds.sequences) == 4
        assert all(isinstance(s, VideoSequence) for s in ds.sequences)
        # Each happy snippet: 1 track x 5 labeled frames.
        assert ds.num_boxes == 20

    def test_video_ids_are_sequential(self, tmp_path: Path) -> None:
        default_happy_dataset(tmp_path)
        ds = load_hmie(tmp_path)
        assert sorted(s.video_id for s in ds.sequences) == [0, 1, 2, 3]

    def test_sequence_carries_paths_and_metadata(self, tmp_path: Path) -> None:
        default_happy_dataset(tmp_path)
        ds = load_hmie(tmp_path)

        seq = ds.sequences[0]
        assert seq.video_path is not None
        assert seq.video_path.endswith(".mp4")
        assert seq.annotation_path.endswith(".json")
        # fps comes from params.videoMetadata.video.fps when not probing.
        assert seq.fps == 30.0
        # afr=5, fps=30 -> keys 0..4 map to video frame indices 0,6,12,18,24;
        # num_frames is the max video-frame index + 1, not the label count.
        assert seq.num_frames == 25
        assert seq.duration is not None

    def test_frame_index_is_video_space_not_key_space(self, tmp_path: Path) -> None:
        # afr=5, video fps=30 -> ratio 6: annotation keys 0..4 must be stored
        # as video frame indices 0,6,12,18,24, not the raw keys 0..4.
        default_happy_dataset(tmp_path)
        ds = load_hmie(tmp_path)
        indices = sorted(b.frame_index for b in ds.sequences[0].boxes)
        assert indices == [0, 6, 12, 18, 24]

    def test_box_fields(self, tmp_path: Path) -> None:
        default_happy_dataset(tmp_path)
        ds = load_hmie(tmp_path)

        box = ds.sequences[0].boxes[0]
        assert isinstance(box, BoxAnnotation)
        assert box.bbox == (10.0, 10.0, 50.0, 40.0)  # left, top, width, height
        assert box.category_name == "vehicle"
        assert box.attributes  # is_truncated / is_occluded carried through


class TestMultiTrack:
    def test_tracks_get_distinct_ids_uuids_and_categories(self, tmp_path: Path) -> None:
        # The common real-data case: several tracks per snippet. Verify
        # positional track_id, track_uuid preservation, and per-track
        # category assignment end-to-end.
        single_video_dataset(
            tmp_path,
            [
                SnippetSpec(
                    name="video_001_000001",
                    annotation=AnnotationSpec(
                        task_id="t-multi",
                        tracks=[
                            TrackSpec(label="boat", num_frames=3),
                            TrackSpec(label="plane", num_frames=2),
                            TrackSpec(label="boat", num_frames=1),
                        ],
                    ),
                )
            ],
        )
        ds = load_hmie(tmp_path)
        seq = ds.sequences[0]

        by_uuid: dict[str, list[BoxAnnotation]] = {}
        for box in seq.boxes:
            by_uuid.setdefault(box.track_uuid, []).append(box)

        # Three tracks -> three distinct uuids, positional ids 1/2/3. The ids
        # are 1-based on purpose: writers that reserve non-positive ids for
        # "no usable track" (MOTChallenge GT) must keep every HMIE track.
        assert set(by_uuid) == {"track-uuid-000", "track-uuid-001", "track-uuid-002"}
        assert {b.track_id for b in by_uuid["track-uuid-000"]} == {1}
        assert {b.track_id for b in by_uuid["track-uuid-001"]} == {2}
        assert {b.track_id for b in by_uuid["track-uuid-002"]} == {3}

        # Per-track category assignment; "boat" reuses one id across tracks.
        assert {b.category_name for b in by_uuid["track-uuid-000"]} == {"boat"}
        assert {b.category_name for b in by_uuid["track-uuid-001"]} == {"plane"}
        assert {b.category_name for b in by_uuid["track-uuid-002"]} == {"boat"}
        assert ds.categories == {"boat": 1, "plane": 2}
        boat_id = ds.categories["boat"]
        assert {b.category_id for b in by_uuid["track-uuid-000"]} == {boat_id}
        assert {b.category_id for b in by_uuid["track-uuid-002"]} == {boat_id}

        # Frame counts per track are preserved (3 + 2 + 1).
        assert len(by_uuid["track-uuid-000"]) == 3
        assert len(by_uuid["track-uuid-001"]) == 2
        assert len(by_uuid["track-uuid-002"]) == 1


class TestPrototypeParity:
    """Coverage of reader outputs the ported prototype produced (checkmaite#635)."""

    def _ann(self, **top: Any) -> dict[str, Any]:
        data: dict[str, Any] = {
            "task_id": "t-parity",
            "status": "completed",
            "params": {"annotation_frame_rate": 5.0, "videoMetadata": {"video": {"fps": 30.0}}},
            "response": {
                "annotations": {
                    "track-0": {
                        "label": "widget",
                        "geometry": "box",
                        "frames": [{"key": 0, "left": 1.0, "top": 2.0, "width": 3.0, "height": 4.0}],
                    }
                }
            },
        }
        data.update(top)
        return data

    def test_global_attributes_from_events_captured(self, tmp_path: Path) -> None:
        # The prototype harvested response.events[].attributes into
        # video_meta["global_attributes"] (level-3 sequence metadata).
        ann_dir = tmp_path / "anns"
        data = self._ann()
        data["response"]["events"] = [
            {"attributes": {"sea_state": "calm"}},
            {"attributes": {"time_of_day": "day"}},
        ]
        _write_annotation(ann_dir / "CDAO_SRC1_clip.mp4_h.json", data)
        ds = load_hmie(tmp_path, annotation_dir=ann_dir)
        assert ds.sequences[0].video_meta["global_attributes"] == {"sea_state": "calm", "time_of_day": "day"}

    def test_duration_read_from_annotation_scalar(self, tmp_path: Path) -> None:
        ann_dir = tmp_path / "anns"
        _write_annotation(ann_dir / "CDAO_SRC1_clip.mp4_h.json", self._ann(duration=12.5))
        ds = load_hmie(tmp_path, annotation_dir=ann_dir)
        # Real declared duration is preferred over the num_frames/fps estimate.
        assert ds.sequences[0].duration == 12.5

    def test_duration_read_from_annotation_dict_seconds(self, tmp_path: Path) -> None:
        ann_dir = tmp_path / "anns"
        _write_annotation(ann_dir / "CDAO_SRC1_clip.mp4_h.json", self._ann(duration={"seconds": 9.0}))
        ds = load_hmie(tmp_path, annotation_dir=ann_dir)
        assert ds.sequences[0].duration == 9.0

    def test_duration_falls_back_to_estimate_when_absent(self, tmp_path: Path) -> None:
        ann_dir = tmp_path / "anns"
        _write_annotation(ann_dir / "CDAO_SRC1_clip.mp4_h.json", self._ann())
        ds = load_hmie(tmp_path, annotation_dir=ann_dir)
        # No declared duration -> estimate from the single mapped frame (index 0).
        seq = ds.sequences[0]
        assert seq.num_frames == 1
        assert seq.duration == 1 / 30.0

    def test_fps_falls_back_to_top_level_fps_extra(self, tmp_path: Path) -> None:
        # No params.videoMetadata.video.fps and no seq_fps; only a top-level
        # "fps" extra -- the prototype used it, so it must not regress to 0.0.
        ann_dir = tmp_path / "anns"
        data = {
            "task_id": "t-fps",
            "status": "completed",
            "params": {"annotation_frame_rate": 5.0},
            "fps": "23.976",
            "response": {
                "annotations": {
                    "track-0": {
                        "label": "widget",
                        "geometry": "box",
                        "frames": [{"key": 0, "left": 1.0, "top": 2.0, "width": 3.0, "height": 4.0}],
                    }
                }
            },
        }
        _write_annotation(ann_dir / "CDAO_SRC1_clip.mp4_h.json", data)
        ds = load_hmie(tmp_path, annotation_dir=ann_dir)
        assert ds.sequences[0].fps == 23.976

    def test_non_box_track_dropped_and_logged(self, tmp_path: Path, caplog: Any) -> None:
        import logging

        ann_dir = tmp_path / "anns"
        data = self._ann()
        data["response"]["annotations"]["track-1"] = {
            "label": "zone",
            "geometry": "polygon",
            "frames": [{"key": 0, "vertices": [[0, 0], [1, 1], [2, 0]]}],
        }
        _write_annotation(ann_dir / "CDAO_SRC1_clip.mp4_h.json", data)

        with caplog.at_level(logging.WARNING, logger="datamaite._formats.hmie.loader"):
            ds = load_hmie(tmp_path, annotation_dir=ann_dir)

        seq = ds.sequences[0]
        # Only the box track yields boxes; the polygon track is dropped...
        assert {b.track_uuid for b in seq.boxes} == {"track-0"}
        assert ds.categories == {"widget": 1}
        # ...and the drop is logged, not silent.
        assert any("non-box" in r.getMessage() for r in caplog.records)


class TestExtendedReaderFields:
    """Fields the loader surfaces beyond the prototype, for downstream consumers."""

    def test_keyframe_fields_carried_onto_boxes(self, tmp_path: Path) -> None:
        # keyframeType / isInferredKeyframe are provenance a MAITE OD adapter
        # needs to tell human-labeled keyframes from interpolated ones.
        default_happy_dataset(tmp_path)
        ds = load_hmie(tmp_path)
        boxes = sorted(ds.sequences[0].boxes, key=lambda b: b.frame_index)
        assert boxes[0].keyframe_type == "start"
        assert all(b.is_inferred is False for b in boxes)

    def test_status_exposed_on_sequence(self, tmp_path: Path) -> None:
        default_happy_dataset(tmp_path)
        ds = load_hmie(tmp_path)
        assert ds.sequences[0].status == "completed"

    def test_scale_metadata_exposed_on_sequence(self, tmp_path: Path) -> None:
        ann_dir = tmp_path / "anns"
        data = {
            "task_id": "t-meta",
            "status": "completed",
            "metadata": {"original_filename": "SRC1_100001.mp4"},
            "params": {"annotation_frame_rate": 5.0, "videoMetadata": {"video": {"fps": 30.0}}},
            "response": {
                "annotations": {
                    "track-0": {
                        "label": "widget",
                        "geometry": "box",
                        "frames": [{"key": 0, "left": 1.0, "top": 2.0, "width": 3.0, "height": 4.0}],
                    }
                }
            },
        }
        _write_annotation(ann_dir / "CDAO_SRC1_clip.mp4_h.json", data)
        ds = load_hmie(tmp_path, annotation_dir=ann_dir)
        assert ds.sequences[0].metadata == {"original_filename": "SRC1_100001.mp4"}


class TestBatchLevelScaleLayout:
    """Default-mode load of the prototype's batch-root scale/ layout."""

    def test_loads_batch_root_scale_dir(self, tmp_path: Path) -> None:
        # root/scale/<CDAO...clip_a.mp4...>.json + root/snippet/seq_mp4/clip_a.mp4
        scale_dir = tmp_path / "scale"
        scale_dir.mkdir()
        data = make_annotation_dict(AnnotationSpec(task_id="t-batch"), FactoryVideoSpec())
        _write_annotation(scale_dir / "CDAO_SRC1_clip_a.mp4_hash.json", data)

        snippet = tmp_path / "snippet_001"
        (snippet / "seq_mp4").mkdir(parents=True)
        make_video(snippet / "seq_mp4" / "clip_a.mp4", FactoryVideoSpec())

        ds = load_hmie(tmp_path)

        assert len(ds.sequences) == 1
        seq = ds.sequences[0]
        assert seq.annotation_path.endswith("CDAO_SRC1_clip_a.mp4_hash.json")
        assert seq.video_path is not None
        assert seq.video_path.endswith("clip_a.mp4")
        assert len(seq.boxes) == 5


class TestCategoryMap:
    def test_single_label_one_category(self, tmp_path: Path) -> None:
        default_happy_dataset(tmp_path)
        ds = load_hmie(tmp_path)
        assert ds.categories == {"vehicle": 1}
        assert {b.category_id for s in ds.sequences for b in s.boxes} == {1}

    def test_ids_stable_across_sequences(self, tmp_path: Path) -> None:
        single_video_dataset(
            tmp_path,
            [
                SnippetSpec(
                    name="video_001_000001",
                    hash_suffix="a",
                    annotation=AnnotationSpec(task_id="t1", tracks=[TrackSpec(label="boat")]),
                ),
                SnippetSpec(
                    name="video_001_000002",
                    hash_suffix="b",
                    annotation=AnnotationSpec(task_id="t2", tracks=[TrackSpec(label="plane")]),
                ),
                SnippetSpec(
                    name="video_001_000003",
                    hash_suffix="c",
                    annotation=AnnotationSpec(task_id="t3", tracks=[TrackSpec(label="boat")]),
                ),
            ],
        )
        ds = load_hmie(tmp_path)

        # Two distinct labels -> two ids; "boat" reuses its id in both snippets.
        assert set(ds.categories) == {"boat", "plane"}
        boat_id = ds.categories["boat"]
        boat_boxes = [b for s in ds.sequences for b in s.boxes if b.category_uri == "boat"]
        assert {b.category_id for b in boat_boxes} == {boat_id}


class TestRequireVideo:
    def test_uses_video_frame_count(self, tmp_path: Path) -> None:
        # Video reports fps=24/24 frames; the annotation deliberately claims a
        # different fps (30) so the assertions can only pass if the probe
        # values -- not the annotation fallback -- are used.
        single_video_dataset(
            tmp_path,
            [
                SnippetSpec(
                    name="video_001_000001",
                    video=FactoryVideoSpec(num_frames=24, fps=24.0),
                    annotation=AnnotationSpec(video_fps=30.0, afr=5.0),
                )
            ],
        )
        ds = load_hmie(tmp_path, require_video=True)

        assert len(ds.sequences) == 1
        seq = ds.sequences[0]
        # Exact probed frame count, not the annotation-derived estimate.
        assert seq.num_frames == 24
        # Exact probed fps, not the annotation's claimed 30.0.
        assert seq.fps == 24.0

    def test_skips_snippet_without_video(self, tmp_path: Path) -> None:
        single_video_dataset(
            tmp_path,
            [SnippetSpec(name="video_001_000001", include_video=False)],
        )
        # Default mode: loaded with no video.
        default_ds = load_hmie(tmp_path)
        assert len(default_ds.sequences) == 1
        assert default_ds.sequences[0].video_path is None

        # require_video mode: snippet with no video is skipped.
        strict_ds = load_hmie(tmp_path, require_video=True)
        assert len(strict_ds.sequences) == 0


class TestTolerance:
    def test_unparseable_annotation_skipped(self, tmp_path: Path) -> None:
        single_video_dataset(
            tmp_path,
            [
                SnippetSpec(name="video_001_000001", hash_suffix="ok"),
                SnippetSpec(
                    name="video_001_000002",
                    hash_suffix="bad",
                    annotation=AnnotationSpec(task_id="t-bad", valid_json=False),
                ),
            ],
        )
        ds = load_hmie(tmp_path)
        # Only the parseable snippet loads.
        assert len(ds.sequences) == 1

    def test_empty_root_returns_empty_dataset(self, tmp_path: Path) -> None:
        ds = load_hmie(tmp_path)
        assert len(ds.sequences) == 0
        assert ds.num_boxes == 0
        assert ds.categories == {}

    def test_box_with_missing_field_is_dropped(self, tmp_path: Path) -> None:
        ann_dir = tmp_path / "anns"
        # One track, two frames; the second frame is missing 'width'.
        data = {
            "task_id": "t-missing",
            "status": "completed",
            "params": {"annotation_frame_rate": 5.0, "videoMetadata": {"video": {"fps": 30.0}}},
            "response": {
                "annotations": {
                    "track-0": {
                        "label": "widget",
                        "geometry": "box",
                        "frames": [
                            {"key": 0, "left": 1.0, "top": 2.0, "width": 3.0, "height": 4.0},
                            {"key": 1, "left": 1.0, "top": 2.0, "height": 4.0},  # no width
                        ],
                    }
                }
            },
        }
        _write_annotation(ann_dir / "CDAO_SRC1_clip.mp4_h.json", data)
        ds = load_hmie(tmp_path, annotation_dir=ann_dir)

        assert len(ds.sequences) == 1
        # Only the complete frame survives as a box.
        assert len(ds.sequences[0].boxes) == 1
        assert ds.sequences[0].boxes[0].frame_index == 0

    def test_unmappable_frame_keys_fall_back_to_label_space_and_log(self, tmp_path: Path, caplog: Any) -> None:
        # No annotation_frame_rate and no video fps -> the key->index mapping
        # is unusable, so frame_index stays in label space. That degrade must
        # be logged (skip-and-log contract), not silent, so a consumer is not
        # left silently mixing label-space and video-space frame indices.
        import logging

        ann_dir = tmp_path / "anns"
        data = {
            "task_id": "t-no-afr",
            "status": "completed",
            "params": {},  # no annotation_frame_rate, no videoMetadata.video.fps
            "response": {
                "annotations": {
                    "track-0": {
                        "label": "widget",
                        "geometry": "box",
                        "frames": [{"key": 3, "left": 1.0, "top": 2.0, "width": 3.0, "height": 4.0}],
                    }
                }
            },
        }
        _write_annotation(ann_dir / "CDAO_SRC1_clip.mp4_h.json", data)

        with caplog.at_level(logging.WARNING, logger="datamaite._formats.hmie.loader"):
            ds = load_hmie(tmp_path, annotation_dir=ann_dir)

        # Identity fallback: raw key preserved.
        assert ds.sequences[0].boxes[0].frame_index == 3
        # And it was logged.
        assert any("cannot be mapped" in r.getMessage() for r in caplog.records)

    def test_empty_label_gets_sentinel_category(self, tmp_path: Path) -> None:
        ann_dir = tmp_path / "anns"
        data = {
            "task_id": "t-empty-label",
            "status": "completed",
            "params": {"annotation_frame_rate": 5.0, "videoMetadata": {"video": {"fps": 30.0}}},
            "response": {
                "annotations": {
                    "track-0": {
                        "label": "",
                        "geometry": "box",
                        "frames": [{"key": 0, "left": 1.0, "top": 2.0, "width": 3.0, "height": 4.0}],
                    }
                }
            },
        }
        _write_annotation(ann_dir / "CDAO_SRC1_clip.mp4_h.json", data)
        ds = load_hmie(tmp_path, annotation_dir=ann_dir)

        box = ds.sequences[0].boxes[0]
        assert box.category_id == -1
        assert box.category_name is None
        assert ds.categories == {}  # unlabeled tracks do not enter the map


class TestOverrideMode:
    def _make_flat(self, tmp_path: Path, *, with_video: bool) -> tuple[Path, Path | None]:
        ann_dir = tmp_path / "annotations"
        ann_dir.mkdir()
        ann_spec = AnnotationSpec(task_id="t-flat")
        video_spec = FactoryVideoSpec()
        data = make_annotation_dict(ann_spec, video_spec)
        _write_annotation(ann_dir / "CDAO_SRC1_clip_a.mp4_hash.json", data)

        video_dir: Path | None = None
        if with_video:
            video_dir = tmp_path / "videos"
            video_dir.mkdir()
            make_video(video_dir / "clip_a.mp4", video_spec)
        return ann_dir, video_dir

    def test_pairs_annotation_to_video(self, tmp_path: Path) -> None:
        ann_dir, video_dir = self._make_flat(tmp_path, with_video=True)
        ds = load_hmie(tmp_path, annotation_dir=ann_dir, video_dir=video_dir)

        assert len(ds.sequences) == 1
        assert ds.sequences[0].video_path is not None
        assert ds.sequences[0].video_path.endswith("clip_a.mp4")

    def test_matches_exact_stem_not_substring_prefix(self, tmp_path: Path) -> None:
        # Two videos whose stems are prefixes of one another: "clip" and
        # "clip_a". The annotation is for clip_a; a substring match would
        # wrongly pair it with "clip" (sorted first).
        ann_dir = tmp_path / "annotations"
        ann_dir.mkdir()
        _write_annotation(
            ann_dir / "CDAO_SRC1_clip_a.mp4_hash.json",
            make_annotation_dict(AnnotationSpec(task_id="t-clip-a"), FactoryVideoSpec()),
        )
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        make_video(video_dir / "clip.mp4", FactoryVideoSpec())
        make_video(video_dir / "clip_a.mp4", FactoryVideoSpec())

        ds = load_hmie(tmp_path, annotation_dir=ann_dir, video_dir=video_dir)

        assert len(ds.sequences) == 1
        assert ds.sequences[0].video_path is not None
        assert ds.sequences[0].video_path.endswith("clip_a.mp4")

    def test_unanchored_substring_does_not_match(self, tmp_path: Path) -> None:
        # "lip.mp4" occurs inside the annotation's embedded "clip.mp4" but is
        # not anchored on a separator (preceded by 'c'), so it must NOT pair.
        ann_dir = tmp_path / "annotations"
        ann_dir.mkdir()
        _write_annotation(
            ann_dir / "CDAO_SRC1_clip.mp4_hash.json",
            make_annotation_dict(AnnotationSpec(task_id="t-clip"), FactoryVideoSpec()),
        )
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        make_video(video_dir / "lip.mp4", FactoryVideoSpec())

        ds = load_hmie(tmp_path, annotation_dir=ann_dir, video_dir=video_dir)

        assert len(ds.sequences) == 1
        assert ds.sequences[0].video_path is None

    def test_annotation_dir_only_leaves_video_none(self, tmp_path: Path) -> None:
        ann_dir, _ = self._make_flat(tmp_path, with_video=False)
        ds = load_hmie(tmp_path, annotation_dir=ann_dir)

        assert len(ds.sequences) == 1
        assert ds.sequences[0].video_path is None

    def test_metadata_json_is_skipped(self, tmp_path: Path) -> None:
        ann_dir = tmp_path / "annotations"
        ann_dir.mkdir()
        # A dataset-metadata JSON (no response/annotations) must not load.
        _write_annotation(ann_dir / "dataset_meta.json", {"video_id": "x", "source": "y"})
        # A real annotation alongside it.
        _write_annotation(
            ann_dir / "CDAO_SRC1_clip_a.mp4_h.json",
            make_annotation_dict(AnnotationSpec(task_id="t"), FactoryVideoSpec()),
        )
        ds = load_hmie(tmp_path, annotation_dir=ann_dir)
        assert len(ds.sequences) == 1

    def test_missing_annotation_dir_returns_empty(self, tmp_path: Path) -> None:
        ds = load_hmie(tmp_path, annotation_dir=tmp_path / "does_not_exist")
        assert len(ds.sequences) == 0


class TestFpsFallback:
    def test_seq_fps_used_when_video_fps_absent(self, tmp_path: Path) -> None:
        ann_dir = tmp_path / "anns"
        # No params.videoMetadata.video.fps; seq_fps lives in the top-level
        # extras (level-2 video metadata), as a string like the real data.
        data = {
            "task_id": "t-seqfps",
            "status": "completed",
            "params": {"annotation_frame_rate": 5.0},
            "seq_fps": "29.97",
            "response": {
                "annotations": {
                    "track-0": {
                        "label": "widget",
                        "geometry": "box",
                        "frames": [{"key": 0, "left": 1.0, "top": 2.0, "width": 3.0, "height": 4.0}],
                    }
                }
            },
        }
        _write_annotation(ann_dir / "CDAO_SRC1_clip.mp4_h.json", data)
        ds = load_hmie(tmp_path, annotation_dir=ann_dir)

        assert ds.sequences[0].fps == 29.97
        assert ds.sequences[0].video_meta["seq_fps"] == "29.97"


class TestMalformedInputs:
    def test_corrupt_video_skipped_when_require_video(self, tmp_path: Path) -> None:
        single_video_dataset(
            tmp_path,
            [SnippetSpec(name="video_001_000001", video=FactoryVideoSpec(corrupt=True))],
        )
        # Corrupt video won't open: require_video drops the snippet.
        assert len(load_hmie(tmp_path, require_video=True).sequences) == 0
        # ...but default mode still loads the annotation.
        assert len(load_hmie(tmp_path).sequences) == 1

    def test_non_json_file_in_annotation_dir_skipped(self, tmp_path: Path) -> None:
        ann_dir = tmp_path / "anns"
        ann_dir.mkdir()
        (ann_dir / "broken.json").write_text("{not valid json")
        ds = load_hmie(tmp_path, annotation_dir=ann_dir)
        assert len(ds.sequences) == 0

    def test_deeply_nested_json_skipped_override_mode(self, tmp_path: Path) -> None:
        # Pathologically nested JSON makes json.load raise RecursionError.
        # The loader must skip it (best-effort), not abort the whole load.
        ann_dir = tmp_path / "anns"
        ann_dir.mkdir()
        depth = 60_000
        (ann_dir / "CDAO_SRC1_clip.mp4_h.json").write_text("[" * depth + "]" * depth)
        # A valid annotation alongside it must still load.
        _write_annotation(
            ann_dir / "CDAO_SRC1_clip_b.mp4_h.json",
            make_annotation_dict(AnnotationSpec(task_id="t-ok"), FactoryVideoSpec()),
        )
        ds = load_hmie(tmp_path, annotation_dir=ann_dir)
        assert len(ds.sequences) == 1

    def test_deeply_nested_json_skipped_default_mode(self, tmp_path: Path) -> None:
        # Same pathological input reached through discovery (default mode),
        # where it lands in check_annotation_schema rather than the
        # override-mode content sniff.
        single_video_dataset(tmp_path, [SnippetSpec(name="video_001_000001")])
        ann = next(tmp_path.rglob("*CDAO*.json"))
        depth = 60_000
        ann.write_text("[" * depth + "]" * depth)
        ds = load_hmie(tmp_path)  # must not raise
        assert len(ds.sequences) == 0


class TestDatasetContainer:
    def test_sequences_len_and_num_boxes(self, tmp_path: Path) -> None:
        default_happy_dataset(tmp_path)
        ds = load_hmie(tmp_path)
        assert len(ds.sequences) == 4
        assert ds.num_boxes == sum(len(s.boxes) for s in ds.sequences)


class TestCloudRoots:
    def test_load_hmie_from_memory_url(self, memory_root) -> None:
        single_video_dataset(
            memory_root,
            [SnippetSpec(name="video_001_000001", video=FactoryVideoSpec(corrupt=True))],
        )
        ds = load_hmie(str(memory_root))
        assert len(ds.sequences) == 1
        assert ds.sequences[0].video_path is not None
        assert ds.sequences[0].video_path.startswith("memory://")
        assert ds.sequences[0].boxes, "annotation boxes should load from the remote JSON"


def _one_box_annotation(frames: list[dict[str, Any]], *, afr: float = 5.0, fps: float = 30.0) -> dict[str, Any]:
    return {
        "task_id": "t",
        "status": "completed",
        "params": {"annotation_frame_rate": afr, "videoMetadata": {"video": {"fps": fps}}},
        "response": {"annotations": {"track-0": {"label": "widget", "geometry": "box", "frames": frames}}},
    }


class TestReviewFixes:
    def test_relative_override_dirs_anchor_to_root(self, tmp_path: Path) -> None:
        # annotation_dir is relative; it must resolve under `root`, not CWD.
        ann_dir = tmp_path / "annotations"
        _write_annotation(
            ann_dir / "CDAO_SRC1_clip.mp4_h.json",
            _one_box_annotation([{"key": 0, "left": 1.0, "top": 2.0, "width": 3.0, "height": 4.0}]),
        )
        ds = load_hmie(tmp_path, annotation_dir="annotations")
        assert len(ds.sequences) == 1
        assert ds.num_boxes == 1

    def test_non_finite_bbox_coords_dropped(self, tmp_path: Path) -> None:
        ann_dir = tmp_path / "anns"
        data = _one_box_annotation(
            [
                {"key": 0, "left": float("nan"), "top": 2.0, "width": 3.0, "height": 4.0},
                {"key": 1, "left": 5.0, "top": 6.0, "width": 7.0, "height": 8.0},
            ]
        )
        _write_annotation(ann_dir / "CDAO_SRC1_clip.mp4_h.json", data)
        ds = load_hmie(tmp_path, annotation_dir=ann_dir)
        # NaN-left box dropped; only the finite box survives.
        assert ds.num_boxes == 1
        assert ds.sequences[0].boxes[0].bbox == (5.0, 6.0, 7.0, 8.0)

    def test_non_finite_fps_falls_back_to_unknown(self, tmp_path: Path) -> None:
        ann_dir = tmp_path / "anns"
        data = _one_box_annotation([{"key": 0, "left": 1.0, "top": 2.0, "width": 3.0, "height": 4.0}], fps=float("inf"))
        # also poison the seq_fps extra with a negative value
        data["seq_fps"] = -5.0
        _write_annotation(ann_dir / "CDAO_SRC1_clip.mp4_h.json", data)
        ds = load_hmie(tmp_path, annotation_dir=ann_dir)
        assert ds.sequences[0].fps == 0.0  # inf fps and negative seq_fps both rejected

    def test_negative_duration_rejected(self, tmp_path: Path) -> None:
        ann_dir = tmp_path / "anns"
        data = _one_box_annotation([{"key": 0, "left": 1.0, "top": 2.0, "width": 3.0, "height": 4.0}])
        data["duration"] = {"seconds": "-5"}
        _write_annotation(ann_dir / "CDAO_SRC1_clip.mp4_h.json", data)
        ds = load_hmie(tmp_path, annotation_dir=ann_dir)
        # negative declared duration ignored; falls back to num_frames/fps (1 frame / 30fps)
        assert ds.sequences[0].duration is None or ds.sequences[0].duration > 0
