"""Tests for the MAITE multi-object-tracking surface of ``BoxTrackDataset``.

``load_hmie`` returns an object that *is* a MAITE MOT dataset, so these tests
index it directly (``ds[i]``) and configure it with ``ds.with_mot_options(...)``
rather than through any adapter/conversion call.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from maite.protocols import multiobject_tracking as mot

from databridge.maite._decode import DecodedFrame, VideoInfo
from databridge.model import BoxTrackDataset

from ._maite_factory import WIDGET, box, make_mp4, sample_dataset, sequence


class TestStructuralConformance:
    def test_is_maite_mot_dataset(self, tmp_path: Path) -> None:
        ds, _ = sample_dataset(tmp_path)
        assert isinstance(ds, mot.Dataset)

    def test_dataset_metadata_has_id_and_index2label(self, tmp_path: Path) -> None:
        ds, _ = sample_dataset(tmp_path)
        assert ds.metadata["id"] == "databridge"
        assert ds.with_mot_options(dataset_id="my-set").metadata["id"] == "my-set"
        assert ds.metadata["index2label"] == {1: "widget", 2: "boat"}

    def test_len_is_number_of_videos(self, tmp_path: Path) -> None:
        ds, _ = sample_dataset(tmp_path)
        assert len(ds) == 1


class TestTargets:
    def test_frame_tracks_align_with_annotated_frames(self, tmp_path: Path) -> None:
        ds, _ = sample_dataset(tmp_path)
        _stream, target, _meta = ds[0]
        # Annotated frames are 0 and 2 -> two frame targets.
        assert len(target.frame_tracks) == 2
        assert target.frame_tracks[0].boxes.shape == (1, 4)  # frame 0: widget
        assert target.frame_tracks[1].boxes.shape == (2, 4)  # frame 2: widget + boat

    def test_box_xywh_converted_to_xyxy(self, tmp_path: Path) -> None:
        ds, _ = sample_dataset(tmp_path)
        _stream, target, _meta = ds[0]
        # frame 0 widget: (left=1, top=2, w=10, h=20) -> (1, 2, 11, 22)
        np.testing.assert_array_equal(target.frame_tracks[0].boxes[0], [1, 2, 11, 22])

    def test_labels_scores_track_ids(self, tmp_path: Path) -> None:
        ds, _ = sample_dataset(tmp_path)
        _stream, target, _meta = ds[0]
        ft = target.frame_tracks[1]  # frame 2
        np.testing.assert_array_equal(ft.labels, [1, 2])
        np.testing.assert_array_equal(ft.scores, [1.0, 1.0])  # ground truth
        np.testing.assert_array_equal(ft.track_ids, [0, 1])
        assert ft.labels.dtype == np.int64
        assert ft.scores.dtype == np.float32


class TestDatumMetadata:
    def test_required_fmv_fields(self, tmp_path: Path) -> None:
        ds, video_path = sample_dataset(tmp_path)
        _stream, _target, meta = ds[0]
        assert set(meta) >= {"id", "height", "width", "time_base", "size"}
        assert meta["width"] == 64
        assert meta["height"] == 48
        assert isinstance(meta["time_base"], Fraction)
        assert meta["size"] == video_path.stat().st_size


class TestVideoStream:
    def test_yields_videoframes(self, tmp_path: Path) -> None:
        ds, _ = sample_dataset(tmp_path)
        stream, _target, _meta = ds[0]
        frames = list(stream)
        assert len(frames) == 2  # annotated frames only
        first = frames[0]
        assert first.pixels.shape == (3, 48, 64)  # (C, H, W)
        assert isinstance(first.time_s, float)
        assert isinstance(first.pts, int)
        # frame_index is emit order, 0..k-1, regardless of source index.
        assert [f.frame_index for f in frames] == [0, 1]

    def test_stream_is_reiterable(self, tmp_path: Path) -> None:
        ds, _ = sample_dataset(tmp_path)
        stream, _target, _meta = ds[0]
        assert len(list(stream)) == len(list(stream)) == 2


class TestEmptyFramePolicy:
    def test_all_emits_empty_targets_for_unlabeled_frames(self, tmp_path: Path) -> None:
        ds, _ = sample_dataset(tmp_path)  # exact 6-frame mp4, boxes on 0 and 2
        stream, target, _meta = ds.with_mot_options(empty_frame_policy="all")[0]
        assert len(target.frame_tracks) == 6
        assert len(list(stream)) == 6
        # Frame 1 is unlabeled -> empty target.
        assert target.frame_tracks[1].boxes.shape == (0, 4)
        assert target.frame_tracks[0].boxes.shape == (1, 4)

    def test_all_falls_back_when_count_estimated(self, tmp_path: Path, caplog) -> None:
        # num_frames present but NOT probed (estimate) -> 'all' must not trust it.
        video_path = make_mp4(tmp_path / "v.mp4")
        seq = sequence(
            str(video_path),
            num_frames=3,
            num_frames_exact=False,
            boxes=[box(track_id=0, category_id=1, uri=WIDGET, name="widget", bbox=(1, 2, 3, 4), frame_index=0)],
        )
        ds = BoxTrackDataset(sequences=[seq], categories={WIDGET: 1})
        with caplog.at_level("WARNING"):
            _stream, target, _meta = ds.with_mot_options(empty_frame_policy="all")[0]
        assert len(target.frame_tracks) == 1  # fell back to annotated
        assert "estimated count" in caplog.text

    def test_all_falls_back_when_frame_count_unknown(self, tmp_path: Path, caplog) -> None:
        video_path = make_mp4(tmp_path / "v.mp4")
        seq = sequence(
            str(video_path),
            num_frames=None,
            boxes=[box(track_id=0, category_id=1, uri=WIDGET, name="widget", bbox=(1, 2, 3, 4), frame_index=0)],
        )
        ds = BoxTrackDataset(sequences=[seq], categories={WIDGET: 1})
        with caplog.at_level("WARNING"):
            _stream, target, _meta = ds.with_mot_options(empty_frame_policy="all")[0]
        assert len(target.frame_tracks) == 1  # fell back to annotated
        assert "no frame count" in caplog.text

    def test_invalid_policy_raises(self, tmp_path: Path) -> None:
        ds, _ = sample_dataset(tmp_path)
        with pytest.raises(ValueError, match="empty_frame_policy"):
            ds.with_mot_options(empty_frame_policy="bogus")  # type: ignore[arg-type]


class TestRecordVsItemViews:
    def test_video_less_sequences_excluded_from_items_but_kept_as_records(self, tmp_path: Path) -> None:
        ds, _ = sample_dataset(tmp_path)
        # A no-video sequence is a record but not a MAITE item (MOT needs pixels).
        ds2 = BoxTrackDataset(
            sequences=(*ds.sequences, sequence(None, video_id=1)),
            categories=dict(ds.categories),
        )
        assert len(ds2) == 1  # MAITE items: only the video-bearing sequence
        assert ds2.sequence_count == 2  # records: both
        assert len(list(ds2.iter_sequences())) == 2


class TestDecoderInjection:
    def test_custom_decoder_is_used(self, tmp_path: Path) -> None:
        ds, _ = sample_dataset(tmp_path)
        calls: list[str] = []

        class FakeDecoder:
            def info(self, video_path: str) -> VideoInfo:
                calls.append(f"info:{video_path}")
                return VideoInfo(width=10, height=20, time_base=Fraction(1, 25), size_bytes=999)

            def stream(self, _video_path: str, source_indices):
                calls.append("stream")
                indices = list(source_indices) if source_indices is not None else [0]
                return [
                    DecodedFrame(pixels=np.zeros((3, 20, 10), dtype=np.uint8), time_s=0.0, pts=0, frame_index=i)
                    for i, _ in enumerate(indices)
                ]

            def decode_one(self, _video_path: str, _source_index: int) -> DecodedFrame:
                return DecodedFrame(pixels=np.zeros((3, 20, 10), dtype=np.uint8), time_s=0.0, pts=0, frame_index=0)

        _stream, _target, meta = ds.with_mot_options(decoder=FakeDecoder())[0]
        assert meta["width"] == 10
        assert meta["size"] == 999
        assert any(c.startswith("info:") for c in calls)


class TestMaiteToolingConformance:
    """Drive the dataset through MAITE's own workflow, not just ``isinstance``.

    ``maite.tasks.predict`` builds its own dataloader, collates with MAITE's
    ``default_collate_fn``, and iterates our object -- so a green run proves the
    structure is genuinely consumable by MAITE tooling, not merely shaped right.
    """

    def test_predict_consumes_dataset(self, tmp_path: Path) -> None:
        from maite.tasks import predict

        ds, _ = sample_dataset(tmp_path)

        class _StubTarget:
            # Minimal MultiobjectTrackingTarget: a (here empty) frame_tracks sequence.
            frame_tracks: ClassVar[list] = []

        class StubMotModel:
            metadata: ClassVar[dict] = {"id": "stub-mot"}

            def __call__(self, batch):
                # batch: Sequence[VideoStream] -> one MOT target per input video.
                return [_StubTarget() for _ in batch]

        preds, _ = predict(model=StubMotModel(), dataset=ds, batch_size=1)
        assert len(preds) == len(ds)  # one prediction batch per video item
