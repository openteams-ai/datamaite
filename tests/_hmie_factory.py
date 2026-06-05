"""Factory helpers for building realistic HMIE test datasets.

Creates the full directory hierarchy described in the databridge HMIE
validator design: dataset root -> full-length video dirs -> snippet dirs
-> labeler subdirs with Scale annotation JSON, plus seq_mp4/ siblings
with matching mp4 files.

Used by e2e tests that exercise the entire discovery + validation pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class VideoSpec:
    """Parameters for a synthetic mp4."""

    num_frames: int = 30
    fps: float = 30.0
    width: int = 320
    height: int = 240
    corrupt: bool = False  # write garbage bytes instead of a real video


@dataclass
class TrackSpec:
    """Parameters for a single track in the annotation JSON."""

    label: str = "vehicle"
    geometry: str = "box"
    num_frames: int = 5  # number of labeled frames
    bbox: tuple[float, float, float, float] = (10.0, 10.0, 50.0, 40.0)  # left, top, width, height


@dataclass
class AnnotationSpec:
    """Parameters for a Scale annotation JSON."""

    task_id: str = "test-task"
    status: str = "completed"
    afr: float = 5.0  # annotation frame rate
    video_fps: float = 30.0
    tracks: list[TrackSpec] = field(default_factory=lambda: [TrackSpec()])
    valid_json: bool = True  # if False, write garbage
    include_task_id: bool = True  # if False, omit required field (schema violation)
    # Optional metadata exercised by the writer round trip. ``video_meta`` is
    # the set of top-level video-level keys the loader harvests (origin_id,
    # codec_name, width/heigth, duration, ...); ``metadata`` is the task-level
    # metadata object; ``global_attributes`` are merged into response.events.
    video_meta: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    global_attributes: dict[str, Any] | None = None


@dataclass
class SnippetSpec:
    """A single snippet: annotation JSON in a subdir + sibling seq_mp4/video.

    Annotations always go in a subdirectory of the snippet (``scale/``,
    ``labeler_alpha/``, etc.) -- never at snippet level.
    """

    name: str  # e.g. "video_001_000001"
    labeler: str = "labeler_alpha"  # annotation subdirectory name
    source_designator: str = "SRC1"
    hash_suffix: str = "abc123"
    video: VideoSpec = field(default_factory=VideoSpec)
    annotation: AnnotationSpec = field(default_factory=AnnotationSpec)
    include_video: bool = True  # if False, omit seq_mp4/*.mp4 (orphan annotation)
    include_annotation: bool = True  # if False, omit annotation JSON (orphan video)


@dataclass
class FullVideoSpec:
    """A full-length video directory containing multiple snippets."""

    name: str  # e.g. "video_001_000000"
    snippets: list[SnippetSpec]


def make_video(path: Path, spec: VideoSpec) -> None:
    """Write a synthetic mp4 (or garbage if spec.corrupt).

    Frames are generated with varied gradient content (not solid black)
    so they pass the flat-frame integrity check. Uses a deterministic
    pattern per frame index so tests stay reproducible.
    """
    if spec.corrupt:
        path.write_bytes(b"this is not a video file")
        return

    import cv2  # type: ignore[import-untyped]
    import numpy as np

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, spec.fps, (spec.width, spec.height))
    try:
        for i in range(spec.num_frames):
            # Horizontal gradient shifted by frame index -- deterministic,
            # non-flat, and different per frame so std is well above the
            # _FLAT_FRAME_STD_THRESHOLD.
            gradient = np.linspace(0, 255, spec.width, dtype=np.uint8)
            row = np.roll(gradient, i * 5)
            frame = np.tile(row, (spec.height, 1))
            rgb = np.stack([frame, frame, frame], axis=-1)
            writer.write(rgb)
    finally:
        writer.release()


def make_annotation_dict(spec: AnnotationSpec, video: VideoSpec) -> dict[str, Any]:
    """Build a Scale annotation dict matching the video parameters.

    Each track's frame keys are clamped individually so tracks with
    different ``num_frames`` produce correct, non-overlapping fixtures.
    """

    # Compute the maximum legal key for a given per-track num_frames, so
    # max_key * video_fps / afr stays strictly less than video.num_frames.
    def _clamp_for(num_frames_in_track: int) -> int:
        max_key = max(num_frames_in_track - 1, 0)
        max_frame_index = int(max_key * video.fps / spec.afr)
        if max_frame_index >= video.num_frames:
            return max(int((video.num_frames - 1) * spec.afr / video.fps), 0)
        return max_key

    annotations: dict[str, Any] = {}
    for i, track in enumerate(spec.tracks):
        track_uuid = f"track-uuid-{i:03d}"
        track_max_key = _clamp_for(track.num_frames)
        frames = []
        for frame_idx in range(track.num_frames):
            key = min(frame_idx, track_max_key)
            frames.append(
                {
                    "keyframeType": "start" if frame_idx == 0 else "middle",
                    "isInferredKeyframe": False,
                    "left": track.bbox[0],
                    "top": track.bbox[1],
                    "width": track.bbox[2],
                    "height": track.bbox[3],
                    "key": key,
                    "attributes": {"is_truncated": "0%", "is_occluded": "0%"},
                    "timestamp_secs": key / spec.afr,
                }
            )
        annotations[track_uuid] = {
            "label": track.label,
            "geometry": track.geometry,
            "frames": frames,
        }

    events: Any = [{"attributes": spec.global_attributes}] if spec.global_attributes else {}
    data: dict[str, Any] = {
        "status": spec.status,
        "type": "videoannotation",
        "params": {
            "annotation_frame_rate": spec.afr,
            "videoMetadata": {"video": {"fps": spec.video_fps}},
        },
        "response": {"annotations": annotations, "events": events},
    }
    if spec.video_meta:
        # Top-level video-level metadata keys (origin_id, codec_name, ...). The
        # loader reads these from the annotation's top level.
        data.update(spec.video_meta)
    if spec.metadata is not None:
        data["metadata"] = spec.metadata
    if spec.include_task_id:
        data["task_id"] = spec.task_id
    return data


def make_snippet(parent_dir: Path, spec: SnippetSpec) -> Path:
    """Create a snippet directory with annotation in a subdir and seq_mp4/*.mp4.

    Annotation always goes in ``<labeler>/`` subdirectory of the snippet.

    Returns the snippet directory path.
    """
    snippet_dir = parent_dir / spec.name
    snippet_dir.mkdir(parents=True, exist_ok=True)

    if spec.include_annotation:
        ann_name = f"CDAO_{spec.source_designator}_{spec.name}.mp4_{spec.hash_suffix}.json"
        ann_parent = snippet_dir / spec.labeler
        ann_parent.mkdir(exist_ok=True)
        ann_path = ann_parent / ann_name

        if spec.annotation.valid_json:
            data = make_annotation_dict(spec.annotation, spec.video)
            ann_path.write_text(json.dumps(data, indent=2))
        else:
            ann_path.write_text("{this is not valid json")

    # Always create seq_mp4/ so the snippet is identifiable by discovery.
    # Only populate it with a video when include_video is True.
    mp4_dir = snippet_dir / "seq_mp4"
    mp4_dir.mkdir(exist_ok=True)
    if spec.include_video:
        mp4_path = mp4_dir / f"{spec.name}.mp4"
        make_video(mp4_path, spec.video)

    return snippet_dir


def make_full_video(root: Path, spec: FullVideoSpec) -> Path:
    """Create a full-length video directory with its metadata and snippets.

    Returns the full-length video directory.
    """
    video_dir = root / spec.name
    video_dir.mkdir(parents=True, exist_ok=True)

    # Dataset-level metadata file
    metadata_path = video_dir / f"{spec.name}.json"
    metadata_path.write_text(json.dumps({"video_id": spec.name, "source": "test"}))

    for snippet in spec.snippets:
        make_snippet(video_dir, snippet)

    return video_dir


def make_hmie_dataset(root: Path, full_videos: list[FullVideoSpec]) -> Path:
    """Create a complete HMIE dataset tree at root."""
    root.mkdir(parents=True, exist_ok=True)
    for fv in full_videos:
        make_full_video(root, fv)
    return root


def single_video_dataset(root: Path, snippets: list[SnippetSpec], *, video_name: str = "video_001_000000") -> Path:
    """Create an HMIE dataset with a single full-length video and the given snippets.

    Convenience wrapper for the common test shape of
    ``make_hmie_dataset(root, [FullVideoSpec(name=..., snippets=[...])])``.
    """
    return make_hmie_dataset(root, [FullVideoSpec(name=video_name, snippets=snippets)])


def default_happy_dataset(root: Path) -> Path:
    """Create a minimal but complete happy-path HMIE dataset.

    Two full-length videos, each with two snippets. All annotations and
    videos are valid and internally consistent.
    """
    return make_hmie_dataset(
        root,
        [
            FullVideoSpec(
                name="video_001_000000",
                snippets=[
                    SnippetSpec(
                        name="video_001_000001",
                        source_designator="SRC1",
                        hash_suffix="abc001",
                        annotation=AnnotationSpec(task_id="t-v1-s1"),
                    ),
                    SnippetSpec(
                        name="video_001_000002",
                        source_designator="SRC1",
                        hash_suffix="abc002",
                        annotation=AnnotationSpec(task_id="t-v1-s2"),
                    ),
                ],
            ),
            FullVideoSpec(
                name="video_002_000000",
                snippets=[
                    SnippetSpec(
                        name="video_002_000001",
                        labeler="labeler_beta",
                        source_designator="SRC2",
                        hash_suffix="def001",
                        annotation=AnnotationSpec(task_id="t-v2-s1"),
                    ),
                    SnippetSpec(
                        name="video_002_000002",
                        labeler="labeler_beta",
                        source_designator="SRC2",
                        hash_suffix="def002",
                        annotation=AnnotationSpec(task_id="t-v2-s2"),
                    ),
                ],
            ),
        ],
    )
