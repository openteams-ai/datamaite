"""Builders for box-track datasets (with real synthetic mp4s) for adapter tests.

All labels/paths here are generic placeholders -- never real ontology URIs,
batch names, or video identifiers.
"""

from __future__ import annotations

from pathlib import Path

from datamaite.model import BBox, BoxAnnotation, BoxTrackDataset, VideoSequence

from ._hmie_factory import VideoSpec, make_video

WIDGET = "http://example.com/ontology/a/widget"
BOAT = "http://example.com/ontology/a/boat"
CATEGORIES = {WIDGET: 1, BOAT: 2}


def make_mp4(path: Path, *, num_frames: int = 6, fps: float = 30.0, width: int = 64, height: int = 48) -> Path:
    """Write a small real mp4 and return its path."""
    make_video(path, VideoSpec(num_frames=num_frames, fps=fps, width=width, height=height))
    return path


def box(
    *,
    track_id: int,
    category_id: int,
    uri: str,
    name: str,
    bbox: BBox,
    frame_index: int,
    is_inferred: bool = False,
    keyframe_type: str = "middle",
) -> BoxAnnotation:
    return BoxAnnotation(
        track_uuid=f"uuid-{track_id}",
        track_id=track_id,
        category_id=category_id,
        category_uri=uri,
        category_name=name,
        bbox=bbox,
        attributes={},
        frame_index=frame_index,
        timestamp=None,
        keyframe_type=keyframe_type,
        is_inferred=is_inferred,
    )


def sequence(
    video_path: str | None,
    *,
    video_id: int = 0,
    fps: float = 30.0,
    num_frames: int | None = 6,
    num_frames_exact: bool = False,
    width: int | None = 64,
    height: int | None = 48,
    size_bytes: int | None = None,
    boxes: list[BoxAnnotation] | None = None,
) -> VideoSequence:
    return VideoSequence(
        video_id=video_id,
        video_path=video_path,
        fps=fps,
        num_frames=num_frames,
        duration=(num_frames / fps if (num_frames and fps) else None),
        annotation_path=f"/tmp/ann_{video_id}.json",  # noqa: S108 - synthetic label, not a real file
        width=width,
        height=height,
        size_bytes=size_bytes,
        boxes=boxes or [],
        num_frames_exact=num_frames_exact,
    )


def sample_dataset(tmp_path: Path) -> tuple[BoxTrackDataset, Path]:
    """One sequence, a real 6-frame mp4, boxes on frames 0 and 2.

    Frame 0: one human-placed widget box.
    Frame 2: one *inferred* widget box + one human-placed boat box.
    """
    video_path = make_mp4(tmp_path / "v.mp4")
    boxes = [
        box(
            track_id=0,
            category_id=1,
            uri=WIDGET,
            name="widget",
            bbox=(1, 2, 10, 20),
            frame_index=0,
            keyframe_type="start",
        ),
        box(track_id=0, category_id=1, uri=WIDGET, name="widget", bbox=(3, 4, 10, 20), frame_index=2, is_inferred=True),
        box(track_id=1, category_id=2, uri=BOAT, name="boat", bbox=(5, 6, 8, 8), frame_index=2, keyframe_type="start"),
    ]
    # num_frames_exact=True: the mp4 really has 6 frames, so empty_frame_policy="all"
    # may trust the count (mirrors a require_video=True load).
    seq = sequence(str(video_path), num_frames_exact=True, boxes=boxes, size_bytes=video_path.stat().st_size)
    return BoxTrackDataset(sequences=[seq], categories=dict(CATEGORIES)), video_path
