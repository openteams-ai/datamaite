"""HMIE/Scale format handler.

- ``loader``: reads HMIE/Scale on disk into ``BoxTrackDataset``
- ``writer``: serialises ``BoxTrackDataset`` back to the HMIE layout
- ``schema``: Pydantic models for Scale Video Playback annotation JSON
- ``discovery``: walks the HMIE folder hierarchy and pairs annotations
  with their videos
- ``annotation_checks``: annotation-file schema and per-track validators
- ``video_checks``: cv2-based video integrity probe + cached properties
- ``consistency_checks``: cross-validation between annotation and video
- ``categories``: HMIE-specific grouping of findings into the 4
  requirement categories from issue #634 (structure, video, coverage,
  scale_spec).
"""

from typing import TYPE_CHECKING

from datamaite._formats.hmie.annotation_checks import check_annotation_schema
from datamaite._formats.hmie.consistency_checks import check_video_annotation_consistency
from datamaite._formats.hmie.discovery import (
    DiscoveryResult,
    SnippetPair,
    discover_hmie_pairs,
    find_batch_roots,
)
from datamaite._formats.hmie.schema import (
    FrameAnnotation,
    Params,
    Response,
    ScaleAnnotation,
    TrackAnnotation,
    VideoMetadata,
    VideoMetadataInner,
)
from datamaite._formats.hmie.video_checks import VideoProperties, probe_video

if TYPE_CHECKING:
    from datamaite._formats.hmie.loader import HmieLoader, load_hmie
    from datamaite._formats.hmie.writer import HmieWriter

__all__ = [
    "DiscoveryResult",
    "FrameAnnotation",
    "HmieLoader",
    "HmieWriter",
    "Params",
    "Response",
    "ScaleAnnotation",
    "SnippetPair",
    "TrackAnnotation",
    "VideoMetadata",
    "VideoMetadataInner",
    "VideoProperties",
    "check_annotation_schema",
    "check_video_annotation_consistency",
    "discover_hmie_pairs",
    "find_batch_roots",
    "load_hmie",
    "probe_video",
]


def __getattr__(name: str) -> object:
    """Lazily resolve loader/writer exports to keep validation imports minimal."""
    if name in {"HmieLoader", "load_hmie"}:
        from datamaite._formats.hmie.loader import HmieLoader, load_hmie

        exports = {"HmieLoader": HmieLoader, "load_hmie": load_hmie}
    elif name == "HmieWriter":
        from datamaite._formats.hmie.writer import HmieWriter

        exports = {"HmieWriter": HmieWriter}
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = exports[name]
    globals()[name] = value
    return value
