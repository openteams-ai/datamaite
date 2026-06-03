"""Multi-object-tracking surface: a box-track dataset as MAITE MOT.

:class:`~databridge.model.BoxTrackDataset` natively implements the MAITE MOT
protocol (``__len__`` / ``__getitem__`` via :func:`build_mot_item`); consumers
index it directly. MOT-view options are set with
:meth:`~databridge.model.BoxTrackDataset.with_mot_options`.

The module does not import ``maite`` -- MAITE protocols are structural, so the
concrete classes here conform by shape alone. This keeps the runtime dependency
to numpy + a decoder.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from databridge.maite._common import (
    EMPTY_BOXES,
    EMPTY_LABELS,
    EMPTY_SCORES,
    EMPTY_TRACK_IDS,
    boxes_array,
    cached_video_info,
    labels_array,
    scores_array,
    track_ids_array,
)
from databridge.maite._decode import DecodedFrame, default_decoder
from databridge.model import BoxAnnotation, BoxTrackDataset, VideoSequence

logger = logging.getLogger(__name__)

EmptyFramePolicy = Literal["annotated", "all"]


@dataclass(frozen=True)
class _FrameTarget:
    """One frame's detections; satisfies ``SingleFrameObjectTrackingTarget``."""

    boxes: np.ndarray
    labels: np.ndarray
    scores: np.ndarray
    track_ids: np.ndarray


# Shared target for unlabeled frames: zero detections, no per-frame allocation.
# Reused across every empty frame (common under empty_frame_policy="all").
_EMPTY_FRAME_TARGET = _FrameTarget(
    boxes=EMPTY_BOXES,
    labels=EMPTY_LABELS,
    scores=EMPTY_SCORES,
    track_ids=EMPTY_TRACK_IDS,
)


def _frame_target(boxes: list[BoxAnnotation]) -> _FrameTarget:
    if not boxes:
        return _EMPTY_FRAME_TARGET
    return _FrameTarget(
        boxes=boxes_array(boxes),
        labels=labels_array(boxes),
        scores=scores_array(boxes),
        track_ids=track_ids_array(boxes),
    )


class _FrameTracks(Sequence):
    """Lazy ``Sequence[SingleFrameObjectTrackingTarget]`` over a video's frames.

    Builds a frame target on access rather than materializing one per frame up
    front, so ``empty_frame_policy="all"`` on a long video doesn't allocate a
    target per frame. Position ``i`` maps to source frame ``frame_order[i]``
    (which keeps it aligned with the paired ``VideoStream``); unlabeled frames
    return the shared empty target.
    """

    def __init__(self, frame_order: Sequence[int], by_frame: dict[int, list[BoxAnnotation]]) -> None:
        self._frame_order = frame_order
        self._by_frame = by_frame

    def __len__(self) -> int:
        return len(self._frame_order)

    def __getitem__(self, index: int) -> _FrameTarget:  # type: ignore[override]
        return _frame_target(self._by_frame.get(self._frame_order[index], []))


@dataclass(frozen=True)
class _MotTarget:
    """Tracks over a video's frames; satisfies ``MultiobjectTrackingTarget``."""

    frame_tracks: Sequence[_FrameTarget]


def _frame_plan(
    seq: VideoSequence, by_frame: dict[int, list[BoxAnnotation]], policy: str
) -> tuple[Sequence[int], Sequence[int] | None]:
    """Return (frame_order, source_indices) for the active empty-frame policy.

    ``frame_order`` indexes the per-frame targets; ``source_indices`` tells the
    decoder which source frames to emit. For ``"all"`` we hand the decoder
    ``None`` so it streams sequentially without building/copying a ``range``
    into a selection set. ``policy`` is ``str`` (not ``EmptyFramePolicy``)
    because the stored ``BoxTrackDataset.empty_frame_policy`` is loosely typed
    to keep the core model free of MAITE-layer types.
    """
    if policy == "all":
        # Only stream every frame when the count is the video's *probed* length.
        # An estimated num_frames (max annotated frame + 1) would make
        # frame_tracks disagree with the real stream length, so degrade to
        # annotated frames with a warning instead.
        if seq.num_frames is not None and seq.num_frames > 0 and seq.num_frames_exact:
            return range(seq.num_frames), None
        logger.warning(
            "empty_frame_policy='all' needs an exact (probed) frame count; %s has %s "
            "-- emitting annotated frames only. Load with require_video=True to use 'all'.",
            Path(seq.annotation_path).name,
            "an estimated count" if seq.num_frames else "no frame count",
        )
    order = sorted(by_frame)
    return order, order


def build_mot_item(
    dataset: BoxTrackDataset, seq: VideoSequence
) -> tuple[Iterable[DecodedFrame], _MotTarget, dict[str, Any]]:
    """Build one MAITE MOT item ``(VideoStream, MotTarget, DatumMetadata)`` for ``seq``.

    ``seq`` is one video-bearing sequence, selected by the caller
    (:meth:`BoxTrackDataset.__getitem__`) from the dataset's cached
    ``_mot_sequences`` list -- so item access stays O(1) and full iteration
    O(N). Decoder/empty-frame-policy come from the dataset's MOT options.
    """
    decoder = dataset._decoder or default_decoder()
    video_path = cast(str, seq.video_path)

    by_frame = seq.boxes_by_frame()
    frame_order, source_indices = _frame_plan(seq, by_frame, dataset.empty_frame_policy)
    target = _MotTarget(frame_tracks=_FrameTracks(frame_order, by_frame))
    stream = decoder.stream(video_path, source_indices)

    info_cache = dataset._caches.setdefault("video_info", {})
    info = cached_video_info(info_cache, video_path, seq, decoder)
    metadata: dict[str, Any] = {
        "id": seq.video_id,
        "height": info.height,
        "width": info.width,
        "time_base": info.time_base,
        "size": info.size_bytes,
    }
    return stream, target, metadata
