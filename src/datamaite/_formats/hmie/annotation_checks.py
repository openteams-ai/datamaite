"""Annotation-file validation checks for HMIE/Scale datasets."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from datamaite._formats.hmie.schema import KNOWN_SCALE_STATUSES, ScaleAnnotation, TrackAnnotation
from datamaite._types import Finding, Severity

logger = logging.getLogger(__name__)


# Annotation files larger than this are refused to avoid OOM on pathological
# or misplaced binary blobs. 100 MB is generous for any real Scale JSON.
_MAX_ANNOTATION_BYTES = 100 * 1024 * 1024  # 100 MB


def _read_annotation_json(
    annotation_path: Path,
    findings: list[Finding],
) -> dict[str, Any] | None:
    """Read and parse an annotation JSON file with size and duplicate-key checks.

    Returns the parsed dict on success, or None if the file cannot be
    read/parsed (findings are appended in-place).
    """
    try:
        file_size = annotation_path.stat().st_size
    except OSError as e:
        findings.append(
            Finding(
                severity=Severity.ERROR,
                path=annotation_path,
                check="annotation_readable",
                message=f"Cannot stat file: {e}",
            )
        )
        return None

    if file_size > _MAX_ANNOTATION_BYTES:
        size_mb = file_size / 1024 / 1024
        limit_mb = _MAX_ANNOTATION_BYTES / 1024 / 1024
        findings.append(
            Finding(
                severity=Severity.ERROR,
                path=annotation_path,
                check="annotation_too_large",
                message=f"File is {size_mb:.0f} MB, exceeds {limit_mb:.0f} MB limit",
            )
        )
        return None

    try:
        raw = annotation_path.read_text(encoding="utf-8")
    except OSError as e:
        findings.append(
            Finding(
                severity=Severity.ERROR,
                path=annotation_path,
                check="annotation_readable",
                message=f"Cannot read file: {e}",
            )
        )
        return None

    # Parse JSON with a hook that detects duplicate keys. Python's default
    # json.loads silently keeps the last value for duplicate keys, which
    # means duplicate track UUIDs (the most common Scale labeling bug)
    # would disappear before we could validate them. The hook records any
    # duplicate key seen during parsing; we emit a finding afterwards.
    duplicate_keys: list[str] = []

    def _detect_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        seen: dict[str, Any] = {}
        for key, value in pairs:
            if key in seen:
                duplicate_keys.append(key)
            seen[key] = value
        return seen

    try:
        data: Any = json.loads(raw, object_pairs_hook=_detect_duplicate_keys)
    except json.JSONDecodeError as e:
        findings.append(
            Finding(
                severity=Severity.ERROR,
                path=annotation_path,
                check="annotation_json",
                message=f"Invalid JSON: {e}",
            )
        )
        return None

    # json.loads accepts any JSON value, but every downstream step
    # (duplicate-key detection, unwrapped-format probe, pydantic schema)
    # assumes a top-level object. Non-object roots (null, list, scalar)
    # were previously silent passes (null) or crashes (list/scalar).
    if not isinstance(data, dict):
        actual = type(data).__name__ if data is not None else "null"
        findings.append(
            Finding(
                severity=Severity.ERROR,
                path=annotation_path,
                check="annotation_schema",
                message=f"Top-level JSON must be an object; got {actual}",
            )
        )
        return None

    if duplicate_keys:
        findings.append(
            Finding(
                severity=Severity.ERROR,
                path=annotation_path,
                check="annotation_duplicate_keys",
                message=(
                    f"JSON contains {len(duplicate_keys)} duplicate key(s) "
                    f"(first: {duplicate_keys[0]!r}); data was silently collapsed"
                ),
            )
        )

    return data


def check_annotation_schema(
    annotation_path: Path,
) -> tuple[list[Finding], ScaleAnnotation | None, Counter[str]]:
    """Validate a Scale annotation JSON file against the expected schema.

    Returns findings, the parsed annotation (if successful), and a
    label histogram (Counter of track labels seen in this file). The
    histogram is always returned, even on schema failure; callers that
    don't care can discard it.
    """
    findings: list[Finding] = []
    label_counter: Counter[str] = Counter()

    data = _read_annotation_json(annotation_path, findings)
    if data is None:
        return findings, None, label_counter

    # Detect unwrapped annotations: Scale sometimes delivers just the
    # response.annotations dict (top-level keys are track UUIDs mapping
    # to objects with label/geometry/frames) without the task_id/response
    # envelope. Wrap it so the schema validator can process it.
    if _is_unwrapped_annotations(data):
        logger.debug("Unwrapped annotation format detected: %s", annotation_path.name)
        findings.append(
            Finding(
                severity=Severity.WARNING,
                path=annotation_path,
                check="annotation_unwrapped",
                message=(
                    "Unwrapped annotation format: file contains raw track data without "
                    "the Scale task envelope (task_id, params, status). Frame-mapping "
                    "metadata (AFR, FPS) is missing. A full-envelope export may exist "
                    "alongside this file."
                ),
            )
        )
        data = _wrap_annotations(data, annotation_path)

    try:
        annotation = ScaleAnnotation.model_validate(data)
    except ValidationError as e:
        for err in e.errors():
            loc = ".".join(str(x) for x in err["loc"])
            findings.append(
                Finding(
                    severity=Severity.ERROR,
                    path=annotation_path,
                    check="annotation_schema",
                    message=f"{loc}: {err['msg']}",
                )
            )
        return findings, None, label_counter

    _check_annotation_metadata(annotation, annotation_path, findings)
    _check_annotation_tracks(annotation, annotation_path, findings, label_counter)

    return findings, annotation, label_counter


def _is_unwrapped_annotations(data: dict[str, Any]) -> bool:
    """Detect if data is a raw annotations dict (no task_id/response envelope).

    Unwrapped format: top-level keys are track UUIDs mapping to dicts
    with ``label``, ``geometry``, and ``frames`` keys. We check the first
    value to decide.
    """
    if "task_id" in data or "response" in data:
        return False
    if not data:
        return False
    first_value = next(iter(data.values()))
    return (
        isinstance(first_value, dict)
        and "label" in first_value
        and "geometry" in first_value
        and "frames" in first_value
    )


def _wrap_annotations(data: dict[str, Any], path: Path) -> dict[str, Any]:
    """Wrap a raw annotations dict into the expected Scale envelope."""
    return {
        "task_id": f"unwrapped-{path.stem}",
        "response": {"annotations": data},
    }


def _check_annotation_metadata(
    annotation: ScaleAnnotation,
    path: Path,
    findings: list[Finding],
) -> None:
    """Check top-level metadata values against Scale spec requirements."""
    # Scale spec: status is required, must be completed/pending/canceled
    if not annotation.status:
        findings.append(
            Finding(
                severity=Severity.WARNING,
                path=path,
                check="annotation_missing_status",
                message="Missing status field (Scale spec: required)",
            )
        )
    elif annotation.status not in KNOWN_SCALE_STATUSES:
        findings.append(
            Finding(
                severity=Severity.WARNING,
                path=path,
                check="annotation_status",
                message=f"Unexpected status: {annotation.status}",
            )
        )

    if len(annotation.response.annotations) == 0:
        findings.append(
            Finding(
                severity=Severity.WARNING,
                path=path,
                check="annotation_empty",
                message="Annotation file has no tracks",
            )
        )

    # Scale spec: params.annotation_frame_rate is required for frame mapping
    if annotation.params is None or annotation.params.annotation_frame_rate is None:
        findings.append(
            Finding(
                severity=Severity.ERROR,
                path=path,
                check="annotation_missing_afr",
                message="Missing params.annotation_frame_rate (required by Scale spec for frame mapping)",
            )
        )

    # Scale spec: params.videoMetadata.video.fps is required for frame mapping
    if annotation.video_fps is None:
        findings.append(
            Finding(
                severity=Severity.ERROR,
                path=path,
                check="annotation_missing_fps",
                message="Missing params.videoMetadata.video.fps (required by Scale spec for frame mapping)",
            )
        )


def _check_annotation_tracks(
    annotation: ScaleAnnotation,
    path: Path,
    findings: list[Finding],
    label_counter: Counter[str],
) -> None:
    """Check per-track annotation values and populate the label histogram.

    Note: geometry validity is enforced by Pydantic's Literal type on
    TrackAnnotation.geometry, so by the time a track reaches here it
    already has a valid value. Bad geometries fail schema validation
    up in check_annotation_schema().
    """
    for track_id, track in annotation.response.annotations.items():
        _check_track_label(track_id, track, path, findings, label_counter)

        if len(track.frames) == 0:
            findings.append(
                Finding(
                    severity=Severity.WARNING,
                    path=path,
                    check="annotation_empty_track",
                    message=f"Track {track_id}: has 0 frames",
                )
            )
            continue

        keys = [f.key for f in track.frames]
        if keys != sorted(keys):
            findings.append(
                Finding(
                    severity=Severity.WARNING,
                    path=path,
                    check="annotation_key_order",
                    message=f"Track {track_id}: frame keys are not monotonically increasing",
                )
            )

        if track.geometry == "box":
            _check_box_track_coordinates(track_id, track, path, findings)
        else:
            # Polygon/line/point/cuboid/ellipse are valid Scale geometries
            # at the JSON/schema level, but the HMIE validator only
            # implements deep checks for 'box' today. Emit a loud WARNING
            # so users know their non-box tracks were parsed but NOT
            # validated internally -- dataset still passes, gap is visible.
            findings.append(
                Finding(
                    severity=Severity.WARNING,
                    path=path,
                    check="annotation_geometry_pending_support",
                    message=(
                        f"Track {track_id}: geometry '{track.geometry}' is a valid Scale "
                        "type but deep validation is not yet implemented (HMIE validator "
                        "currently checks 'box' only). Non-box data passed through "
                        "unchecked -- coming soon."
                    ),
                )
            )


def _check_track_label(
    track_id: str,
    track: TrackAnnotation,
    path: Path,
    findings: list[Finding],
    label_counter: Counter[str],
) -> None:
    """Validate a track's label and contribute to the label histogram.

    Empty or whitespace-only labels are ERRORs (for a 60K-dataset buy,
    label noise is the #1 cause of downstream model confusion).
    All labels -- valid or not -- are added to the histogram so callers
    can spot taxonomy drift, casing inconsistencies, and outliers.
    """
    raw_label = track.label if track.label is not None else ""
    normalized = raw_label.strip()

    if not normalized:
        findings.append(
            Finding(
                severity=Severity.ERROR,
                path=path,
                check="annotation_label_empty",
                message=f"Track {track_id}: label is empty or whitespace-only",
            )
        )
        label_counter["<empty>"] += 1
    else:
        label_counter[normalized] += 1


def _check_box_track_coordinates(
    track_id: str,
    track: TrackAnnotation,
    path: Path,
    findings: list[Finding],
) -> None:
    """Aggregate per-track counts of zero-area and negative-coord bboxes.

    Only called for ``geometry == "box"`` tracks. Schema allows
    missing box fields (for non-box geometries); a box track with
    missing fields is a data-quality issue — count and flag separately
    so zero-area / negative counts don't mask the real root cause.
    """
    zero_area_count = 0
    negative_coord_count = 0
    missing_fields_count = 0
    for frame in track.frames:
        if frame.left is None or frame.top is None or frame.width is None or frame.height is None:
            missing_fields_count += 1
            continue
        if frame.width <= 0 or frame.height <= 0:
            zero_area_count += 1
        if frame.left < 0 or frame.top < 0:
            negative_coord_count += 1

    total = len(track.frames)
    if missing_fields_count > 0:
        findings.append(
            Finding(
                severity=Severity.ERROR,
                path=path,
                check="annotation_box_missing_fields",
                message=(
                    f"Track {track_id}: {missing_fields_count}/{total} frames "
                    f"missing one or more box fields (left/top/width/height) "
                    f"despite geometry='box'"
                ),
            )
        )
    if zero_area_count > 0:
        findings.append(
            Finding(
                severity=Severity.WARNING,
                path=path,
                check="annotation_bbox_size",
                message=f"Track {track_id}: {zero_area_count}/{total} frames have zero-area bbox",
            )
        )
    if negative_coord_count > 0:
        findings.append(
            Finding(
                severity=Severity.WARNING,
                path=path,
                check="annotation_bbox_negative",
                message=(f"Track {track_id}: {negative_coord_count}/{total} frames have negative left/top coordinates"),
            )
        )
