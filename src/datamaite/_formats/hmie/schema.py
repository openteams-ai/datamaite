"""Pydantic models for Scale AI Video Playback annotation format.

Reference: Scale Video Playback JSON Overview (internal doc) +
https://scale.com/docs/api-reference/image-and-video-reference

The schema is intentionally permissive (``extra="allow"``) because Scale
exports routinely carry program-specific or version-specific fields we
don't consume. Required fields are limited to what the validator must
read; everything else is either optional or ignored.
"""

from __future__ import annotations

from typing import Any, Literal, get_args

from pydantic import BaseModel, Field

# Scale's Video Playback supports these geometries. Modeling as Literal
# lets Pydantic reject typos at parse time rather than leaving the magic
# string set in checks.py.
Geometry = Literal["box", "polygon", "line", "point", "cuboid", "ellipse"]

# Real-world Scale statuses observed across HMIE exports, per the public
# Scale API docs. Intentionally typed loose on ScaleAnnotation.status (str
# rather than this Literal) so an unrecognized status doesn't fail schema
# parsing and lose the rest of the annotation. The runtime check in
# checks.py compares against KNOWN_SCALE_STATUSES derived from this Literal.
ScaleStatus = Literal[
    "completed",
    "pending",
    "canceled",
    "error",
    "expired",
    "unassigned",
    "submitted",
]
KNOWN_SCALE_STATUSES: frozenset[str] = frozenset(get_args(ScaleStatus))


class FrameAnnotation(BaseModel):
    """A single frame within a track annotation.

    ``left``/``top``/``width``/``height`` are the bounding-box fields
    Scale emits for ``geometry="box"`` tracks. Non-box tracks (polygon,
    line, point, cuboid, ellipse) carry their own per-geometry fields
    (``vertices``, ``points``, etc.) and do not populate the box fields,
    so they must be optional at the schema level. Box-specific
    consumers (``_check_box_track_coordinates``,
    ``_check_track_bbox_bounds``) gate on ``geometry == "box"`` before
    accessing them; they should not dereference these on non-box
    tracks.
    """

    key: int = Field(ge=0)
    left: float | None = None
    top: float | None = None
    height: float | None = Field(default=None, ge=0)
    width: float | None = Field(default=None, ge=0)
    keyframeType: str | None = None  # noqa: N815
    isInferredKeyframe: bool | None = None  # noqa: N815
    attributes: dict[str, Any] | None = None
    timestamp_secs: float | None = None

    model_config = {"extra": "allow"}


class TrackAnnotation(BaseModel):
    """A labeled track (one object across multiple frames)."""

    label: str
    geometry: Geometry
    frames: list[FrameAnnotation]

    model_config = {"extra": "allow"}


class VideoMetadataInner(BaseModel):
    fps: float | None = None

    model_config = {"extra": "allow"}


class VideoMetadata(BaseModel):
    video: VideoMetadataInner | None = None

    model_config = {"extra": "allow"}


class Params(BaseModel):
    annotation_frame_rate: float | None = None
    videoMetadata: VideoMetadata | None = None  # noqa: N815

    model_config = {"extra": "allow"}


class Response(BaseModel):
    annotations: dict[str, TrackAnnotation]
    events: dict[str, Any] | list[Any] | None = None

    model_config = {"extra": "allow"}


class ScaleAnnotation(BaseModel):
    """Top-level Scale Video Playback annotation task.

    Only fields the validator actively reads are modeled. Everything else
    from Scale's payload (created_at, completed_at, project, etc.) is
    preserved via ``extra="allow"`` but not enumerated.
    """

    task_id: str
    response: Response
    status: str | None = None
    metadata: dict[str, Any] | None = None
    params: Params | None = None

    model_config = {"extra": "allow"}

    @property
    def video_fps(self) -> float | None:
        """Walk the optional params.videoMetadata.video.fps chain in one place."""
        if self.params is None or self.params.videoMetadata is None or self.params.videoMetadata.video is None:
            return None
        return self.params.videoMetadata.video.fps
