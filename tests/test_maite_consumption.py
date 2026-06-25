"""Consumption tests: MOT decode behavior, lazy targets, datum-metadata caching.

These guard the CPU/memory behavior flagged on MR !6: MOT ``"all"`` must stream
without building a selection set, unlabeled frames must share one target
instance, and the annotated policy must hand the decoder the exact frame
indices. The dataset is indexed directly (it *is* a MAITE MOT dataset);
MOT-view options are set with ``ds.with_mot_options(...)``.
"""

from __future__ import annotations

from pathlib import Path

from datamaite.maite._common import EMPTY_BOXES, boxes_array
from datamaite.maite._decode import DecodedFrame, PyAVDecoder, VideoInfo
from datamaite.maite._mot import _EMPTY_FRAME_TARGET

from ._maite_factory import sample_dataset


class SpyDecoder:
    """Wraps PyAVDecoder, counting calls and recording stream() selections."""

    def __init__(self) -> None:
        self._inner = PyAVDecoder()
        self.info_calls = 0
        self.stream_calls = 0
        self.decode_one_calls = 0
        self.stream_args: list[object] = []

    def info(self, video_path: str) -> VideoInfo:
        self.info_calls += 1
        return self._inner.info(video_path)

    def stream(self, video_path: str, source_indices):  # type: ignore[no-untyped-def]
        self.stream_calls += 1
        self.stream_args.append(source_indices)
        return self._inner.stream(video_path, source_indices)

    def decode_one(self, video_path: str, source_index: int) -> DecodedFrame:
        self.decode_one_calls += 1
        return self._inner.decode_one(video_path, source_index)


class TestMotLazyTargets:
    def test_all_policy_streams_all_without_selection_set(self, tmp_path: Path) -> None:
        ds, _ = sample_dataset(tmp_path)  # exact 6-frame mp4, boxes on 0 and 2
        spy = SpyDecoder()
        mds = ds.with_mot_options(empty_frame_policy="all", decoder=spy)
        _stream, target, _meta = mds[0]
        assert len(target.frame_tracks) == 6
        # "all" hands the decoder None (stream everything), not a built range/set.
        assert spy.stream_args[-1] is None

    def test_empty_frames_share_singleton_target(self, tmp_path: Path) -> None:
        ds, _ = sample_dataset(tmp_path)
        _stream, target, _meta = ds.with_mot_options(empty_frame_policy="all")[0]
        # Frame 1 is unlabeled -> the shared empty target instance.
        assert target.frame_tracks[1] is _EMPTY_FRAME_TARGET
        assert target.frame_tracks[0] is not _EMPTY_FRAME_TARGET  # frame 0 has a box

    def test_annotated_passes_explicit_indices(self, tmp_path: Path) -> None:
        ds, _ = sample_dataset(tmp_path)
        spy = SpyDecoder()
        mds = ds.with_mot_options(decoder=spy)
        mds[0]
        assert spy.stream_args[-1] == [0, 2]  # sorted annotated frames


class TestSharedEmptyArrays:
    def test_empty_boxes_is_shared_and_readonly(self) -> None:
        assert boxes_array([]) is EMPTY_BOXES
        assert EMPTY_BOXES.flags.writeable is False
