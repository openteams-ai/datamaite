"""HMIE-specific categorization of validation findings.

The HMIE validator groups findings into four requirement categories from
issue #634:

- ``structure``: discovery / folder-layout issues.
- ``video``: FMV (video) integrity issues.
- ``coverage``: missing or orphaned annotations / videos.
- ``scale_spec``: Scale annotation-JSON compliance.

``_types.py`` stays format-neutral: it knows about severities, findings,
and results, but not about HMIE-specific check names. The mapping below
(and the helper that applies it) is the HMIE package's responsibility.
"""

from __future__ import annotations

from collections import Counter

# Map finding check names to the 4 requirement categories from #634.
_CHECK_CATEGORIES: dict[str, str] = {
    # Folder structure
    "discovery": "structure",
    "path_exists": "structure",
    "path_is_dir": "structure",
    # FMV integrity
    "video_open": "video",
    "video_frame_count": "video",
    "video_resolution": "video",
    "video_first_frame": "video",
    "video_fps": "video",
    "video_flat_frames": "video",
    "video_mid_frame": "video",
    "video_last_frame": "video",
    "video_dependency": "video",
    "video_missing": "video",
    "multiple_videos_in_seq_mp4": "video",
    "video_probe_error": "video",
    # Annotation coverage
    "orphan_annotation": "coverage",
    "orphan_video": "coverage",
    "no_annotations": "coverage",
    # Scale spec compliance
    "annotation_readable": "scale_spec",
    "annotation_json": "scale_spec",
    "annotation_duplicate_keys": "scale_spec",
    "annotation_schema": "scale_spec",
    "annotation_unwrapped": "scale_spec",
    "annotation_status": "scale_spec",
    "annotation_missing_status": "scale_spec",
    "annotation_empty": "scale_spec",
    "annotation_missing_afr": "scale_spec",
    "annotation_missing_fps": "scale_spec",
    "annotation_label_empty": "scale_spec",
    "annotation_bbox_size": "scale_spec",
    "annotation_bbox_negative": "scale_spec",
    "annotation_box_missing_fields": "scale_spec",
    "annotation_empty_track": "scale_spec",
    "annotation_key_order": "scale_spec",
    "annotation_geometry_pending_support": "scale_spec",
    "annotation_too_large": "scale_spec",
    # Consistency
    "consistency_fps": "scale_spec",
    "consistency_fps_invalid": "scale_spec",
    "consistency_frame_bounds": "scale_spec",
    "consistency_bbox_bounds": "scale_spec",
    # Worker / batch crashes -- internal validator failures, bucketed with
    # scale_spec so they surface as scale-column reds rather than getting
    # silently dropped by _categorize_findings (which raises on unknown
    # check names). worker_crash is per-pair, validate_crash is per-batch.
    "worker_crash": "scale_spec",
    "validate_crash": "scale_spec",
}

_CATEGORY_LABELS: dict[str, str] = {
    "structure": "Folder structure",
    "video": "FMV integrity",
    "coverage": "Annotation coverage",
    "scale_spec": "Scale spec compliance",
}


def _categorize_findings(
    severity_counts: dict[str, Counter[str]],
) -> dict[str, tuple[int, int]]:
    """Categorize uncapped severity counts into (errors, warnings) per category.

    Raises ``KeyError`` if a check name is not registered in
    ``_CHECK_CATEGORIES``. A loud failure forces new checks to be
    registered explicitly: a silent fallback would let a future
    structure/video/coverage check land in the wrong bucket and
    quietly distort the dashboard.
    """
    cats: dict[str, tuple[int, int]] = {
        "structure": (0, 0),
        "video": (0, 0),
        "coverage": (0, 0),
        "scale_spec": (0, 0),
    }
    err_by_cat: Counter[str] = Counter()
    warn_by_cat: Counter[str] = Counter()
    for check, n in severity_counts.get("error", Counter()).items():
        err_by_cat[_CHECK_CATEGORIES[check]] += n
    for check, n in severity_counts.get("warning", Counter()).items():
        warn_by_cat[_CHECK_CATEGORIES[check]] += n
    for key in cats:
        cats[key] = (err_by_cat[key], warn_by_cat[key])
    return cats


# ---------------------------------------------------------------------------
# Skipped-check vocabulary
# ---------------------------------------------------------------------------
# Logical names recorded on ValidationResult.skipped_checks when a check is
# intentionally not run. Kept here (not in _types.py) so the neutral model
# stays format-agnostic while the HMIE layer owns the display mapping.

SKIP_VIDEO_INTEGRITY = "video_integrity"
SKIP_VIDEO_CONSISTENCY = "video_annotation_consistency"


def skipped_category_keys(skipped: set[str]) -> set[str]:
    """Map skipped logical-check names to the report categories they fully cover.

    Only ``video`` can be marked entirely skipped: ``video_integrity`` covers
    the whole FMV-integrity category. ``video_annotation_consistency`` is part
    of ``scale_spec`` (which still runs annotation-schema checks), so it never
    marks a category skipped -- it is surfaced via the banner instead.
    """
    keys: set[str] = set()
    if SKIP_VIDEO_INTEGRITY in skipped:
        keys.add("video")
    return keys
