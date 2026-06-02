"""Databridge's neutral in-memory dataset model.

This is the format-agnostic intermediate representation at the center of
the bridge: every *loader* produces it and every *converter* consumes it.
A loader (e.g. :func:`databridge.load_hmie`) turns an on-disk dataset into
a :class:`Dataset`; a converter turns a :class:`Dataset` into an output
format (MOTChallenge, YOLO, ...). Because the model is tied to neither a
specific input nor a specific output format, databridge is an N-to-M
bridge -- add a loader to gain an input, add a converter to gain an
output -- rather than a one-off HMIE-to-X path.

The dataclasses are intentionally plain (no external protocol dependency).
They are shaped so a MAITE-protocol adapter can wrap them later without
changing this module.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

# Bounding box as (left, top, width, height) in pixels.
BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class BoxAnnotation:
    """One bounding box on one frame of one track.

    ``bbox`` is ``(left, top, width, height)`` in pixels, matching the
    per-frame fields of the source annotation. ``category_id`` is assigned
    per dataset (stable across all sequences in the same :class:`Dataset`),
    and ``category_name`` is the final path segment of the ontology URI.

    ``keyframe_type`` (start/middle/end) and ``is_inferred`` come straight
    from the source frame: ``is_inferred=True`` marks a tool-interpolated
    box rather than a human-placed keyframe, which a downstream consumer
    may want to weight or filter.
    """

    track_uuid: str
    track_id: int
    category_id: int
    category_uri: str
    category_name: str | None
    bbox: BBox
    attributes: dict[str, Any]
    frame_index: int
    timestamp: float | None
    keyframe_type: str | None = None
    is_inferred: bool | None = None


@dataclass(frozen=True)
class VideoSequence:
    """One snippet: a video plus all of its box annotations.

    ``video_path`` is ``None`` when the snippet has an annotation but no
    discoverable video (and the loader was not asked to require one).

    ``num_frames`` is the video's *true* frame count only when the loader
    probed the video. Otherwise it is a lower-bound *estimate* -- the
    maximum annotated ``frame_index`` plus one (``None`` when the snippet
    has no boxes). Because labeling usually stops before the end of a
    snippet, this estimate understates the true video length; do not treat
    the non-probed value as the real frame count. ``duration``
    (``num_frames / fps``) inherits the same caveat.

    ``status`` is the source task status (``completed``/``pending``/etc.),
    so a consumer can filter non-final tasks. ``video_meta`` is video-level
    metadata (e.g. codec, dimensions, plus any global attributes);
    ``metadata`` is the source task-level ``metadata`` object (which may
    carry the original video filename).
    """

    video_id: int
    video_path: str | None
    fps: float
    num_frames: int | None
    duration: float | None
    annotation_path: str
    status: str | None = None
    video_meta: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    boxes: list[BoxAnnotation] = field(default_factory=list)


@dataclass(frozen=True)
class Dataset:
    """A loaded dataset: its sequences plus a shared category map.

    This is the neutral container that every loader returns and every
    converter accepts. ``categories`` maps each ontology URI to the integer
    ``category_id`` used throughout ``sequences``. The map is built once for
    the whole dataset so the same label always gets the same id across
    sequences.
    """

    sequences: list[VideoSequence]
    categories: dict[str, int]

    def __len__(self) -> int:
        return len(self.sequences)

    def __iter__(self) -> Iterable[VideoSequence]:
        return iter(self.sequences)

    @property
    def num_boxes(self) -> int:
        """Total box annotations across all sequences."""
        return sum(len(seq.boxes) for seq in self.sequences)
