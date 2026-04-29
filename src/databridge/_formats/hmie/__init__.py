"""HMIE/Scale format handler.

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

from databridge._formats.hmie.annotation_checks import check_annotation_schema
from databridge._formats.hmie.consistency_checks import check_video_annotation_consistency
from databridge._formats.hmie.discovery import (
    DiscoveryResult,
    SnippetPair,
    discover_hmie_pairs,
    find_batch_roots,
)
from databridge._formats.hmie.schema import (
    FrameAnnotation,
    Params,
    Response,
    ScaleAnnotation,
    TrackAnnotation,
    VideoMetadata,
    VideoMetadataInner,
)
from databridge._formats.hmie.video_checks import VideoProperties, probe_video

__all__ = [
    "DiscoveryResult",
    "FrameAnnotation",
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
    "probe_video",
]
