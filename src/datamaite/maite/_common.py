"""Shared array/metadata helpers for the MAITE adapters.

These turn datamaite's neutral :class:`~datamaite.model.BoxAnnotation`s into
the numpy arrays MAITE's target protocols expect, build the fallback
:class:`~datamaite.maite._decode.VideoInfo` used when a video can't be probed,
and memoize per-video probes so a dataset never re-opens the same file just to
read its dimensions.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from datamaite.maite._decode import Decoder, VideoInfo, resolve_video_info
from datamaite.model import BBox, BoxAnnotation, VideoSequence


def _readonly(array: np.ndarray) -> np.ndarray:
    array.flags.writeable = False
    return array


# Shared, immutable empty arrays. Unlabeled frames (common under the "all"
# frame/empty-frame policies) all share these instead of each allocating its
# own zero-length arrays. Marked read-only so a consumer can't mutate the
# shared instance.
EMPTY_BOXES = _readonly(np.zeros((0, 4), dtype=np.float32))
EMPTY_LABELS = _readonly(np.zeros((0,), dtype=np.int64))
EMPTY_SCORES = _readonly(np.zeros((0,), dtype=np.float32))
EMPTY_TRACK_IDS = _readonly(np.zeros((0,), dtype=np.int64))


def xywh_to_xyxy(bbox: BBox) -> tuple[float, float, float, float]:
    """Convert ``(left, top, width, height)`` to MAITE's ``(x0, y0, x1, y1)``."""
    left, top, width, height = bbox
    return (left, top, left + width, top + height)


def boxes_array(boxes: list[BoxAnnotation]) -> np.ndarray:
    """``(N, 4)`` float32 array of xyxy boxes (shared empty when no boxes)."""
    if not boxes:
        return EMPTY_BOXES
    return np.array([xywh_to_xyxy(b.bbox) for b in boxes], dtype=np.float32)


def labels_array(boxes: list[BoxAnnotation]) -> np.ndarray:
    """``(N,)`` int64 array of category ids (``-1`` marks unlabeled tracks)."""
    if not boxes:
        return EMPTY_LABELS
    return np.array([b.category_id for b in boxes], dtype=np.int64)


def scores_array(boxes: list[BoxAnnotation]) -> np.ndarray:
    """``(N,)`` float32 array of scores.

    These are ground-truth annotations, so every box scores ``1.0`` by
    convention -- there is no model confidence attached to a human label.
    """
    if not boxes:
        return EMPTY_SCORES
    return np.ones((len(boxes),), dtype=np.float32)


def track_ids_array(boxes: list[BoxAnnotation]) -> np.ndarray:
    """``(N,)`` int64 array of per-box track ids (``>= 0``; MAITE reserves ``-1``)."""
    if not boxes:
        return EMPTY_TRACK_IDS
    return np.array([b.track_id for b in boxes], dtype=np.int64)


def fallback_video_info(seq: VideoSequence) -> VideoInfo:
    """Build a :class:`VideoInfo` from the loader's stored metadata.

    Used only when a live PyAV probe fails. ``time_base`` is derived from
    ``fps`` (``1/round(fps)``) when fps is known, else a 1-millisecond base;
    width/height/size come from the sequence fields (``0`` when absent).
    """
    time_base = Fraction(1, round(seq.fps)) if seq.fps and seq.fps > 0 else Fraction(1, 1000)
    return VideoInfo(
        width=seq.width or 0,
        height=seq.height or 0,
        time_base=time_base,
        size_bytes=seq.size_bytes or 0,
    )


def cached_video_info(cache: dict[str, VideoInfo], video_path: str, seq: VideoSequence, decoder: Decoder) -> VideoInfo:
    """Probe ``video_path`` once and memoize the result in ``cache``.

    Datum metadata is identical for every frame of a video, so probing it on
    each ``__getitem__`` would re-``av.open()`` the same file per frame. The
    cache makes it one probe per distinct video for the lifetime of the adapter.
    """
    info = cache.get(video_path)
    if info is None:
        info = resolve_video_info(video_path, decoder, fallback_video_info(seq))
        cache[video_path] = info
    return info
