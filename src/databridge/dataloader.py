"""HMIE/Scale loader -- the reference implementation of the loader contract.

:class:`HmieLoader` is databridge's first :class:`~databridge.loaders.Loader`:
it reads the HMIE/Scale on-disk layout and produces the neutral
:class:`databridge.model.Dataset` model that every converter consumes. The
generic, format-agnostic machinery (the ``Loader`` base class, the registry,
and the :func:`databridge.load` dispatcher) lives in
:mod:`databridge.loaders`; this module is what a new format loader is modeled
on. :func:`load_hmie` is a thin convenience wrapper around ``HmieLoader``.

``HmieLoader`` is the configurable successor to the hard-coded notebook
loader: instead of a fixed ``/path/to/dataset`` and a fragile
``rglob("*CDAO*.json")`` walk, it reuses the package's
:func:`databridge._formats.hmie.discovery.discover_hmie_pairs` to pair
annotations with videos across the layout variants seen in real data, and
parses each annotation through the shared :class:`ScaleAnnotation` schema.

Per the loader contract, loading is deliberately *separate* from validation
and best-effort: an annotation that cannot be parsed at all is skipped (with
a log warning) rather than aborting the load, and per-frame data with missing
box fields is dropped at the box level. Callers who want to know *why* an
annotation is bad should run :func:`databridge.validate` instead.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

from databridge._formats.hmie import ScaleAnnotation, check_annotation_schema, discover_hmie_pairs
from databridge._formats.hmie.annotation_checks import _is_unwrapped_annotations
from databridge._formats.hmie.discovery import _VIDEO_EXTENSIONS, SnippetPair, match_annotation_to_video
from databridge._formats.hmie.frame_mapping import frame_key_to_index, is_mappable
from databridge._types import DatasetFormat
from databridge.loaders import Loader, register_loader
from databridge.model import BoxAnnotation, Dataset, VideoSequence

logger = logging.getLogger(__name__)

# Top-level annotation keys carrying video-level metadata. Mirrors the
# notebook's ``_extract_video_meta`` key list (level-2 metadata). ``heigth``
# is a real misspelling observed in the source data and is preserved here so
# the field is not silently dropped.
_VIDEO_META_KEYS = (
    "origin_id",
    "data_source",
    "src_record_key",
    "seq_fps",
    "processing_pipeline",
    "codec_name",
    "width",
    "heigth",
    "height",
    "bit_rate",
    "duration",
    "nb_frames",
    "format_name",
    "fps",
    "size",
)


@register_loader
class HmieLoader(Loader):
    """Loader for the HMIE/Scale on-disk format (the reference loader).

    Subclasses :class:`databridge.loaders.Loader` and is registered for
    :attr:`DatasetFormat.HMIE`, so ``databridge.load(root)`` dispatches here.
    """

    format = DatasetFormat.HMIE

    def load(
        self,
        root: str | Path,
        *,
        annotation_dir: str | Path | None = None,
        video_dir: str | Path | None = None,
        require_video: bool = False,
        **_: Any,
    ) -> Dataset:
        """Read an HMIE/Scale dataset under ``root`` into a :class:`Dataset`.

        Parameters
        ----------
        root
            Dataset root directory. In the default mode the nested HMIE
            layout under ``root`` is auto-discovered.
        annotation_dir, video_dir
            Optional overrides for non-standard layouts where annotations and
            videos live in flat sibling directories rather than the nested
            snippet hierarchy. When *either* is given, discovery is bypassed:
            annotation JSONs are read from ``annotation_dir`` (default
            ``root``), and each is paired to a video in ``video_dir`` (if
            given) by matching the video's embedded filename (anchored) in the
            annotation filename. Annotations with no matching video get
            ``video_path = None``. Relative override paths are resolved
            against ``root`` (not the current working directory); absolute
            paths are used as-is.
        require_video
            When True, each sequence's video must exist and open: the loader
            probes it (via the ``video`` extra), sets ``num_frames`` from the
            true frame count, and *skips* snippets whose video is missing or
            unreadable (logged as a warning). When False (default), videos are
            not opened and ``num_frames`` is derived from the annotation.

        Returns
        -------
        Dataset
            Loaded sequences and the dataset-wide category map. Empty when no
            annotation/video pairs are found.
        """
        root = Path(root)

        if annotation_dir is not None or video_dir is not None:
            # Relative override paths anchor to ``root`` (not the caller's CWD),
            # so load_hmie("/data/batch", annotation_dir="ann/") reads
            # /data/batch/ann/. ``root / p`` leaves absolute overrides intact.
            pairs = _pairs_from_dirs(
                root / annotation_dir if annotation_dir is not None else root,
                root / video_dir if video_dir is not None else None,
            )
        else:
            pairs = discover_hmie_pairs(root).pairs

        categories: dict[str, int] = {}
        sequences: list[VideoSequence] = []
        for pair in pairs:
            seq = _load_sequence(pair, next_video_id=len(sequences), categories=categories, require_video=require_video)
            if seq is not None:
                sequences.append(seq)

        logger.info("Loaded %d sequence(s), %d categories from %s", len(sequences), len(categories), root)
        return Dataset(sequences=sequences, categories=categories)


def load_hmie(
    root: str | Path,
    *,
    annotation_dir: str | Path | None = None,
    video_dir: str | Path | None = None,
    require_video: bool = False,
) -> Dataset:
    """Load an HMIE/Scale dataset from disk into the neutral model.

    Thin convenience wrapper around :meth:`HmieLoader.load` (equivalent to
    ``databridge.load(root, dataset_format=DatasetFormat.HMIE, ...)``). See
    :meth:`HmieLoader.load` for the parameter and return semantics.
    """
    return HmieLoader().load(root, annotation_dir=annotation_dir, video_dir=video_dir, require_video=require_video)


def _pairs_from_dirs(annotation_dir: Path, video_dir: Path | None) -> list[SnippetPair]:
    """Build annotation/video pairs from flat directories (override mode).

    Annotation JSONs are any ``*.json`` under ``annotation_dir`` that look
    like a Scale annotation (so dataset-metadata JSONs are skipped). Each is
    paired to a video under ``video_dir`` whose filename is embedded
    (anchored) in the annotation name -- Scale annotation names embed the
    video filename (``..._<video-name>.mp4_<hash>.json``). See
    :func:`match_annotation_to_video` for the exact anchoring and tie-break
    rule (shared with batch-level discovery).
    """
    if not annotation_dir.is_dir():
        logger.warning("annotation_dir is not a directory: %s", annotation_dir)
        return []

    videos: list[Path] = []
    if video_dir is not None and video_dir.is_dir():
        videos = sorted(p for p in video_dir.rglob("*") if p.suffix.lower() in _VIDEO_EXTENSIONS)

    pairs: list[SnippetPair] = []
    for ann_path in sorted(annotation_dir.rglob("*.json")):
        if not _looks_like_annotation(ann_path):
            continue
        video_path = match_annotation_to_video(ann_path.name, videos)
        pairs.append(SnippetPair(annotation_path=ann_path, video_path=video_path))
    return pairs


def _looks_like_annotation(path: Path) -> bool:
    """Cheap content check distinguishing Scale annotations from metadata JSON.

    A Scale annotation either carries the ``response`` envelope or is an
    unwrapped track dict (UUID -> object with label/geometry/frames).
    Dataset/video metadata JSONs have neither, so they are skipped in
    override mode rather than loaded as empty sequences.
    """
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError, RecursionError):
        # RecursionError: pathologically nested JSON. Treat as "not a usable
        # annotation" and skip rather than letting it abort the whole load.
        return False
    if not isinstance(data, dict):
        return False
    # Wrapped (task_id/response envelope) or the unwrapped track-dict form.
    # Reuse the schema layer's predicate so "what is an unwrapped annotation"
    # has one definition shared with check_annotation_schema.
    if "response" in data or "task_id" in data:
        return True
    return _is_unwrapped_annotations(data)


def _load_sequence(
    pair: SnippetPair,
    *,
    next_video_id: int,
    categories: dict[str, int],
    require_video: bool,
) -> VideoSequence | None:
    """Build one VideoSequence from a discovered pair, or None to skip it.

    Returns None when the annotation cannot be parsed, or when
    ``require_video`` is set and the snippet has no usable video.
    """
    try:
        _findings, annotation, _labels = check_annotation_schema(pair.annotation_path)
    except RecursionError:
        # Pathologically nested JSON exceeds the parser's recursion limit.
        # Best-effort load: skip-and-log rather than abort the whole load.
        logger.warning("Skipping annotation exceeding JSON nesting limits: %s", pair.annotation_path)
        return None
    if annotation is None:
        logger.warning("Skipping unparseable annotation: %s", pair.annotation_path)
        return None

    video_meta = _extract_video_meta(annotation)

    video_path = pair.video_path
    num_frames: int | None = None
    # fps source order: the annotation's declared video fps, then the
    # top-level seq_fps / fps extras (the prototype used seq_fps then fps).
    # First finite, positive fps wins; NaN/Inf/<=0 are treated as "unknown"
    # and fall through to 0.0 (the model's unknown-fps sentinel).
    fps = (
        _coerce_positive_float(annotation.video_fps)
        or _coerce_positive_float(video_meta.get("seq_fps"))
        or _coerce_positive_float(video_meta.get("fps"))
        or 0.0
    )

    if require_video:
        probed = _probe_for_load(video_path)
        if probed is None:
            logger.warning("Skipping snippet with missing/unreadable video: %s", pair.annotation_path)
            return None
        probe_fps, frame_count = probed
        if probe_fps > 0:
            fps = probe_fps
        if frame_count > 0:
            num_frames = frame_count

    # Build boxes only after fps is resolved (incl. the require_video probe
    # override) so frame_index and num_frames share one clock.
    afr = annotation.params.annotation_frame_rate if annotation.params is not None else None
    if annotation.response.annotations and not is_mappable(fps, afr):
        # No usable fps/afr: frame_key_to_index falls back to the raw key, so
        # this sequence's frame_index is in label space, not video-frame
        # space. Log it -- otherwise a consumer silently mixes the two clocks.
        logger.warning(
            "Frame keys cannot be mapped to video frames (fps=%s, afr=%s); frame_index stays in label space for %s",
            fps,
            afr,
            pair.annotation_path,
        )
    boxes = _build_boxes(annotation, categories, fps=fps, afr=afr, annotation_path=pair.annotation_path)

    if num_frames is None:
        max_index = max((b.frame_index for b in boxes), default=-1)
        num_frames = max_index + 1 if max_index >= 0 else None

    # Prefer the annotation's declared duration; fall back to num_frames/fps.
    duration = _coerce_duration(annotation.model_extra)
    if duration is None:
        duration = num_frames / fps if num_frames and fps > 0 else None

    return VideoSequence(
        video_id=next_video_id,
        video_path=str(video_path) if video_path is not None else None,
        fps=fps,
        num_frames=num_frames,
        duration=duration,
        annotation_path=str(pair.annotation_path),
        status=annotation.status,
        video_meta=video_meta,
        metadata=dict(annotation.metadata or {}),
        boxes=boxes,
    )


def _build_boxes(
    annotation: ScaleAnnotation,
    categories: dict[str, int],
    *,
    fps: float,
    afr: float | None,
    annotation_path: Path,
) -> list[BoxAnnotation]:
    """Flatten a Scale annotation's box tracks into per-frame BoxAnnotations.

    Only ``geometry="box"`` tracks produce boxes; non-box geometries
    (polygon/line/point/cuboid/ellipse) carry no bounding box and are
    skipped (logged once per annotation, since the loader emits boxes
    only). Boxes with any missing coordinate (left/top/width/height
    ``None``) are also skipped. ``track_id`` is the track's positional index
    within this annotation; ``category_id`` is assigned from (and stored
    into) the dataset-wide ``categories`` map.

    ``frame_index`` is the *video* frame index, mapped from the Scale frame
    key via :func:`frame_key_to_index` (``floor(key * fps / afr)``). The
    raw key indexes label-space, which diverges from video-frame-space
    whenever ``afr != fps`` (the usual subsampled HMIE case); storing the
    raw key would put boxes and ``num_frames`` on different clocks.
    """
    boxes: list[BoxAnnotation] = []
    skipped_non_box = 0
    for track_id, (track_uuid, track) in enumerate(annotation.response.annotations.items()):
        if track.geometry != "box":
            skipped_non_box += 1
            continue
        category_uri = track.label or ""
        category_id, category_name = _resolve_category(category_uri, categories)
        for frame in track.frames:
            if frame.left is None or frame.top is None or frame.width is None or frame.height is None:
                continue
            bbox = (float(frame.left), float(frame.top), float(frame.width), float(frame.height))
            if not all(math.isfinite(coord) for coord in bbox):
                # NaN/Inf coordinates can't be serialised to a real box; drop
                # them rather than emitting a non-finite bbox into the model.
                continue
            boxes.append(
                BoxAnnotation(
                    track_uuid=track_uuid,
                    track_id=track_id,
                    category_id=category_id,
                    category_uri=category_uri,
                    category_name=category_name,
                    bbox=bbox,
                    attributes=dict(frame.attributes or {}),
                    frame_index=frame_key_to_index(frame.key, fps, afr),
                    timestamp=frame.timestamp_secs,
                    keyframe_type=frame.keyframeType,
                    is_inferred=frame.isInferredKeyframe,
                )
            )
    if skipped_non_box:
        logger.warning(
            "Dropped %d non-box track(s) (polygon/line/point/...) from %s; the loader emits bounding boxes only",
            skipped_non_box,
            annotation_path,
        )
    return boxes


def _resolve_category(category_uri: str, categories: dict[str, int]) -> tuple[int, str | None]:
    """Map an ontology URI to a stable (category_id, category_name).

    Unlabeled tracks (empty URI) get id ``-1`` and no name. Otherwise the
    URI is assigned the next 1-based id on first sight and reused
    thereafter; the name is the final path segment of the URI.
    """
    if not category_uri:
        return -1, None
    if category_uri not in categories:
        categories[category_uri] = len(categories) + 1
    return categories[category_uri], category_uri.rstrip("/").split("/")[-1]


def _extract_video_meta(annotation: ScaleAnnotation) -> dict[str, Any]:
    """Collect video-level metadata from the annotation.

    Pulls the known top-level (level-2) keys, and also harvests global
    attributes from ``response.events`` into ``global_attributes``
    (level-3 sequence metadata), matching the prototype reader. ``events``
    is permissive in the schema (list or dict), so both shapes are handled.
    """
    extra = annotation.model_extra or {}
    meta: dict[str, Any] = {key: extra[key] for key in _VIDEO_META_KEYS if key in extra}

    events = annotation.response.events
    if isinstance(events, dict):
        events = list(events.values())
    global_attributes: dict[str, Any] = {}
    for event in events or []:
        attributes = event.get("attributes") if isinstance(event, dict) else None
        if isinstance(attributes, dict):
            global_attributes.update(attributes)
    if global_attributes:
        meta["global_attributes"] = global_attributes

    return meta


def _coerce_duration(extra: dict[str, Any] | None) -> float | None:
    """Read the annotation's declared top-level ``duration``.

    Handles the scalar/string form and the ``{"seconds": ...}`` dict form
    seen in Scale payloads. Returns None when absent or unusable, so the
    caller can fall back to the ``num_frames / fps`` estimate.
    """
    if not extra:
        return None
    raw = extra.get("duration")
    if isinstance(raw, dict):
        return _coerce_positive_float(raw.get("seconds"))
    return _coerce_positive_float(raw)


def _coerce_float(value: Any) -> float | None:
    """Best-effort float coercion; returns None for unusable or non-finite values."""
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _coerce_positive_float(value: Any) -> float | None:
    """Coerce to a finite, strictly-positive float, else None.

    For fps/duration-like fields where ``0``, negative, ``NaN``, and ``Inf``
    are all invalid -- any of them would yield a meaningless ``VideoSequence``
    fps or duration, so they collapse to None (caller's "unknown" fallback).
    """
    result = _coerce_float(value)
    return result if result is not None and result > 0 else None


def _probe_for_load(video_path: Path | None) -> tuple[float, int] | None:
    """Probe a video for (fps, frame_count), or None if unusable.

    Returns None when the path is absent, missing on disk, or cannot be
    opened. Reuses the validator's :func:`probe_video`; integrity findings
    are discarded -- loading only needs fps and frame count.
    """
    if video_path is None or not video_path.exists():
        return None
    from databridge._formats.hmie import probe_video

    props, _findings = probe_video(video_path)
    if not props.opened:
        return None
    return props.fps, props.frame_count
