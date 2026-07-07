"""Cross-validation checks between Scale annotations and their videos."""

from __future__ import annotations

import math
from pathlib import Path

from datamaite._formats.hmie.frame_mapping import frame_key_to_index, is_mappable
from datamaite._formats.hmie.schema import ScaleAnnotation, TrackAnnotation
from datamaite._formats.hmie.video_checks import VideoProperties
from datamaite._types import Finding, Severity

# Bbox pixel tolerance for the consistency cross-check. Scale labelers
# round coordinates to the nearest pixel, which can produce a bbox that
# extends 1 pixel past the nominal frame edge on a final-column or
# final-row object. A 1-pixel tolerance suppresses that harmless
# rounding artifact while still catching genuinely off-canvas boxes.
_BBOX_PIXEL_TOLERANCE = 1

# FPS deviation tolerance between video and annotation (frames per second).
# Real videos commonly report fractional FPS (29.97, 23.976) that the
# annotation tool rounds to the nearest integer, so a small absolute
# tolerance suppresses harmless rounding mismatches.
_FPS_TOLERANCE = 0.5


def check_video_annotation_consistency(
    video_path: Path,
    annotation: ScaleAnnotation,
    video_props: VideoProperties,
) -> list[Finding]:
    """Cross-check video metadata against annotation metadata.

    video_props must come from a prior probe_video() call; this function
    does NOT open the video itself. Single-open per video is enforced at
    the call site so we don't double the av.open() container cost at 60K
    scale.
    """
    findings: list[Finding] = []

    if not video_props.opened:
        return findings

    _check_fps_consistency(annotation, video_path, video_props.fps, findings)
    _check_frame_and_bbox_bounds(
        annotation,
        video_path,
        video_props.fps,
        video_props.frame_count,
        video_props.width,
        video_props.height,
        findings,
    )

    return findings


def _check_fps_consistency(
    annotation: ScaleAnnotation,
    video_path: Path,
    video_fps: float,
    findings: list[Finding],
) -> None:
    """Check FPS consistency between video and annotation.

    Non-finite FPS values (NaN / ±Inf) must be flagged explicitly, not
    silently skipped — ``abs(x - NaN) > 0.5`` is False, so a malformed
    annotation FPS would otherwise produce no finding at all.
    """
    ann_fps = annotation.video_fps
    if ann_fps is not None and not math.isfinite(ann_fps):
        findings.append(
            Finding(
                severity=Severity.WARNING,
                path=video_path,
                check="consistency_fps_invalid",
                message=f"Annotation FPS is non-finite ({ann_fps!r})",
            )
        )
        return
    if not math.isfinite(video_fps):
        findings.append(
            Finding(
                severity=Severity.WARNING,
                path=video_path,
                check="consistency_fps_invalid",
                message=f"Video FPS is non-finite ({video_fps!r})",
            )
        )
        return
    if ann_fps is not None and abs(video_fps - ann_fps) > _FPS_TOLERANCE:
        findings.append(
            Finding(
                severity=Severity.WARNING,
                path=video_path,
                check="consistency_fps",
                message=f"Video FPS ({video_fps:.2f}) differs from annotation FPS ({ann_fps:.2f})",
            )
        )


def _check_frame_and_bbox_bounds(
    annotation: ScaleAnnotation,
    video_path: Path,
    video_fps: float,
    video_frame_count: int,
    video_width: int,
    video_height: int,
    findings: list[Finding],
) -> None:
    """Check frame key bounds and bounding box bounds against video dimensions."""
    # _usable_afr returns the annotation AFR when the key->index mapping is
    # usable, or None to skip the frame-bounds check. The actual mapping is
    # done by the shared frame_key_to_index() so loader and validator agree.
    afr = _usable_afr(annotation, video_fps, video_frame_count)
    check_bbox = video_width > 0 and video_height > 0

    for track_id, track in annotation.response.annotations.items():
        if len(track.frames) == 0:
            continue

        if afr is not None:
            _check_track_frame_bounds(track_id, track, video_path, video_fps, afr, video_frame_count, findings)

        if check_bbox and track.geometry == "box":
            _check_track_bbox_bounds(track_id, track, video_path, video_width, video_height, findings)


def _usable_afr(
    annotation: ScaleAnnotation,
    video_fps: float,
    video_frame_count: int,
) -> float | None:
    """Return the annotation AFR if frame keys can be mapped, else None.

    Returns None (skip the frame-bounds check) when the video has no frames
    or the fps/afr pair is not mappable. The fps/afr mappability rule is the
    shared :func:`is_mappable`, so this guard and the loader cannot drift.
    """
    if video_frame_count <= 0:
        return None
    afr = annotation.params.annotation_frame_rate if annotation.params is not None else None
    if not is_mappable(video_fps, afr):
        return None
    return afr


def _check_track_frame_bounds(
    track_id: str,
    track: TrackAnnotation,
    video_path: Path,
    video_fps: float,
    afr: float | None,
    video_frame_count: int,
    findings: list[Finding],
) -> None:
    """Flag a track whose max frame key maps past the end of the video.

    Frame index math: ``frame# = floor(key * fps / afr)`` (via the shared
    :func:`frame_key_to_index`). Valid frame indices are
    ``0..video_frame_count-1``, so we fail when the derived frame index is
    ``>= video_frame_count``.
    """
    max_key = max(f.key for f in track.frames)
    max_frame_index = frame_key_to_index(max_key, video_fps, afr)
    if max_frame_index < video_frame_count:
        return

    findings.append(
        Finding(
            severity=Severity.WARNING,
            path=video_path,
            check="consistency_frame_bounds",
            message=(
                f"Track {track_id}: max key {max_key} maps to frame index {max_frame_index} "
                f"but video has {video_frame_count} frames (valid 0..{video_frame_count - 1})"
            ),
        )
    )


def _check_track_bbox_bounds(
    track_id: str,
    track: TrackAnnotation,
    video_path: Path,
    video_width: int,
    video_height: int,
    findings: list[Finding],
) -> None:
    """Flag bboxes that extend outside the video canvas (past pixel tolerance)."""
    max_right = video_width + _BBOX_PIXEL_TOLERANCE
    max_bottom = video_height + _BBOX_PIXEL_TOLERANCE
    # Box fields are Optional on FrameAnnotation so non-box geometries
    # parse cleanly; a box track with missing fields is surfaced by
    # annotation_box_missing_fields and skipped here to avoid double-reporting.
    out_of_bounds_count = sum(
        1
        for frame in track.frames
        if frame.left is not None
        and frame.top is not None
        and frame.width is not None
        and frame.height is not None
        and (frame.left + frame.width > max_right or frame.top + frame.height > max_bottom)
    )
    if out_of_bounds_count == 0:
        return

    findings.append(
        Finding(
            severity=Severity.WARNING,
            path=video_path,
            check="consistency_bbox_bounds",
            message=(
                f"Track {track_id}: {out_of_bounds_count}/{len(track.frames)} "
                f"frames have bbox extending outside {video_width}x{video_height}"
            ),
        )
    )
