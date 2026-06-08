"""VisDrone video dataset loader.

VisDrone's video tasks use image sequences plus comma-separated annotation
files. Both the Object Detection in Videos (VID) and Multi-Object Tracking
(MOT) variants use rows shaped like::

    <frame_index>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>

Official split archives are laid out as::

    <VisDrone2019-VID-train or VisDrone2019-MOT-train>/
        sequences/<sequence_name>/0000001.jpg
        annotations/<sequence_name>.txt

This module implements the loader side only. It reads one official split root
or a parent directory containing multiple split roots into the neutral
:class:`databridge.model.BoxTrackDataset` model. Loading is best-effort:
malformed rows are skipped with warnings so callers can still inspect the rest
of the dataset.
"""

from __future__ import annotations

import csv
import logging
import math
import os
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from databridge._types import DatasetFormat
from databridge.loaders import Loader, register_loader
from databridge.model import BoxAnnotation, BoxTrackDataset, VideoSequence

logger = logging.getLogger(__name__)

VisDroneVariant = Literal["auto", "vid", "mot"]
VisDroneAnnotationSource = Literal["gt", "det"]

_VISDRONE_CLASS_NAMES = {
    0: "ignored_region",
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning_tricycle",
    9: "bus",
    10: "motor",
    11: "others",
}
_VALID_VARIANTS = {"auto", "vid", "mot"}
_VALID_SOURCES = {"gt", "det"}
_SPLIT_TOKENS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("test-challenge", ("test-challenge", "test_challenge", "testchallenge")),
    ("test-dev", ("test-dev", "test_dev", "testdev")),
    ("train", ("train", "training")),
    ("val", ("val", "validation")),
    ("test", ("test",)),
)
_SPLIT_ORDER = {"train": 0, "val": 1, "test-dev": 2, "test-challenge": 3, "test": 4}


@dataclass(frozen=True)
class _SplitRoot:
    """One official VisDrone split root discovered under the user root."""

    path: Path
    split: str
    variant: str


@dataclass(frozen=True)
class _ImageProbe:
    """Best-effort frame-directory metadata."""

    frame_count: int | None = None
    width: int | None = None
    height: int | None = None


@dataclass(frozen=True)
class _BaseRow:
    """Parsed VisDrone row fields common to ground truth and detections."""

    frame: int
    target_id: int
    left: float
    top: float
    width: float
    height: float
    score: float
    category_id: int
    truncation: int | None
    occlusion: int | None


@register_loader
class VisDroneVideoLoader(Loader):
    """Loader for VisDrone VID and MOT video dataset roots.

    The loader accepts either a single official split root containing
    ``sequences/`` and ``annotations/`` or a parent containing multiple such
    roots (for example ``VisDrone2019-VID-train`` and
    ``VisDrone2019-VID-val``). Use :func:`load_visdrone_video` for a
    convenience wrapper or ``databridge.load(root,
    dataset_format="visdrone_video", ...)`` for the registry-dispatched entry
    point.
    """

    format = DatasetFormat.VISDRONE_VIDEO

    def load(
        self,
        root: str | Path,
        *,
        variant: VisDroneVariant | str = "auto",
        annotation_source: VisDroneAnnotationSource | str = "gt",
        include_ignored: bool = False,
        classes: Collection[int] | None = None,
        frame_ext: str = ".jpg",
        fps: float = 0.0,
        probe_images: bool = False,
        **_: Any,
    ) -> BoxTrackDataset:
        """Read VisDrone video split(s) into :class:`BoxTrackDataset`.

        Parameters
        ----------
        root
            Either a split root with ``sequences/`` and ``annotations/`` or a
            parent directory containing one or more such split roots.
        variant
            ``"vid"`` for Object Detection in Videos, ``"mot"`` for
            Multi-Object Tracking, or ``"auto"`` (default) to infer from split
            directory names. The row format is the same; the variant is stored
            in sequence metadata.
        annotation_source
            ``"gt"`` (default) treats the rows as ground truth: ``score <= 0``
            and category ``0`` rows are ignored unless ``include_ignored`` is
            true. ``"det"`` treats rows as detector proposals and stores the
            score as a detection score; non-positive target IDs receive stable
            negative pseudo track IDs.
        include_ignored
            Include rows with VisDrone category ``0`` (ignored regions) or
            non-positive GT score. Defaults to ``False``.
        classes
            Optional VisDrone category-id allowlist (for example ``{1, 4}`` for
            pedestrians and cars). Omit/``None`` to keep all selected classes.
        frame_ext
            Frame image extension to count/probe. Official VisDrone video
            archives use ``.jpg`` and seven-digit 1-based filenames.
        fps
            Optional sequence frame rate. Official annotations do not carry
            frame rate metadata, so the default is ``0.0`` and durations remain
            unknown. If supplied and frames are countable, duration is computed.
        probe_images
            When True, use OpenCV (``pip install databridge[video]``) to read
            the first frame image and fill dimensions. Loading continues with a
            warning if OpenCV is unavailable.

        Returns
        -------
        BoxTrackDataset
            Image-sequence-backed VisDrone sequence records. Empty when no
            official split roots or no annotation files are found.
        """
        root = Path(root)
        resolved_variant = _normalize_variant(variant)
        source = _normalize_source(annotation_source)
        class_filter = _normalize_classes(classes)
        frame_ext = _normalize_extension(frame_ext)
        fps_value = _coerce_non_negative_float(fps, option="fps")

        split_roots = _split_roots(root, variant=resolved_variant)
        if not split_roots:
            logger.warning(
                "VisDrone video root must contain sequences/ and annotations/ directories, "
                "or child split roots that do: %s",
                root,
            )
            return BoxTrackDataset(sequences=(), categories={})

        categories: dict[str, int] = {}
        sequences: list[VideoSequence] = []
        for split_root in split_roots:
            for annotation_path in _annotation_files(split_root.path):
                seq = _load_sequence(
                    split_root,
                    annotation_path=annotation_path,
                    source=source,
                    include_ignored=include_ignored,
                    class_filter=class_filter,
                    frame_ext=frame_ext,
                    fps=fps_value,
                    probe_images=probe_images,
                    categories=categories,
                    next_video_id=len(sequences),
                )
                if seq is not None:
                    sequences.append(seq)

        if not sequences:
            logger.warning("VisDrone video root contains no loadable annotation files: %s", root)
        logger.info(
            "Loaded %d VisDrone video sequence(s), %d categories from %s", len(sequences), len(categories), root
        )
        return BoxTrackDataset(sequences=tuple(sequences), categories=categories)


def load_visdrone_video(
    root: str | Path,
    *,
    variant: VisDroneVariant | str = "auto",
    annotation_source: VisDroneAnnotationSource | str = "gt",
    include_ignored: bool = False,
    classes: Collection[int] | None = None,
    frame_ext: str = ".jpg",
    fps: float = 0.0,
    probe_images: bool = False,
) -> BoxTrackDataset:
    """Load a VisDrone VID or MOT video dataset root.

    Equivalent to ``databridge.load(root, dataset_format="visdrone_video", ...)``.
    See :meth:`VisDroneVideoLoader.load` for parameter semantics.
    """
    return VisDroneVideoLoader().load(
        root,
        variant=variant,
        annotation_source=annotation_source,
        include_ignored=include_ignored,
        classes=classes,
        frame_ext=frame_ext,
        fps=fps,
        probe_images=probe_images,
    )


def _split_roots(root: Path, *, variant: str) -> list[_SplitRoot]:
    """Return official split roots in deterministic order."""
    if _is_split_root(root):
        return [_build_split_root(root, variant=variant)]

    try:
        children = sorted(path for path in root.iterdir() if path.is_dir())
    except OSError as exc:
        logger.warning("Could not list VisDrone video root %s: %s", root, exc)
        return []
    split_roots = [_build_split_root(path, variant=variant) for path in children if _is_split_root(path)]
    return sorted(split_roots, key=_split_sort_key)


def _is_split_root(path: Path) -> bool:
    """Return True when ``path`` has the official VisDrone split shape."""
    return (path / "sequences").is_dir() and (path / "annotations").is_dir()


def _split_sort_key(split_root: _SplitRoot) -> tuple[int, str]:
    """Sort split roots in the usual train/val/test order, then by path."""
    return _SPLIT_ORDER.get(split_root.split, len(_SPLIT_ORDER)), split_root.path.name


def _build_split_root(path: Path, *, variant: str) -> _SplitRoot:
    """Build a split descriptor with inferred split/variant metadata."""
    return _SplitRoot(
        path=path,
        split=_infer_split(path.name),
        variant=_infer_variant(path.name) if variant == "auto" else variant,
    )


def _annotation_files(split_root: Path) -> list[Path]:
    """Return per-sequence annotation files for one split root."""
    annotation_dir = split_root / "annotations"
    try:
        files = sorted(path for path in annotation_dir.iterdir() if path.is_file() and path.suffix.lower() == ".txt")
    except OSError as exc:
        logger.warning("Could not list VisDrone annotation directory %s: %s", annotation_dir, exc)
        return []
    if not files and (split_root / "sequences").is_dir():
        logger.warning("VisDrone split has sequences but no .txt annotation files: %s", split_root)
    return files


def _load_sequence(
    split_root: _SplitRoot,
    *,
    annotation_path: Path,
    source: str,
    include_ignored: bool,
    class_filter: frozenset[int] | None,
    frame_ext: str,
    fps: float,
    probe_images: bool,
    categories: dict[str, int],
    next_video_id: int,
) -> VideoSequence | None:
    """Build one sequence record from one VisDrone annotation file."""
    sequence_name = annotation_path.stem
    frame_dir = split_root.path / "sequences" / sequence_name
    if not frame_dir.is_dir():
        logger.warning("VisDrone sequence frame directory is missing: %s", frame_dir)

    boxes = _parse_annotation_file(
        annotation_path,
        split=split_root.split,
        variant=split_root.variant,
        source=source,
        include_ignored=include_ignored,
        class_filter=class_filter,
        categories=categories,
    )
    image_probe = _probe_images(frame_dir, frame_ext, enabled=probe_images)
    inferred_num_frames = _max_frame_index(boxes)
    num_frames = image_probe.frame_count or inferred_num_frames
    num_frames_exact = image_probe.frame_count is not None
    duration = num_frames / fps if num_frames_exact and num_frames is not None and fps > 0 else None

    video_meta: dict[str, Any] = {
        "format": "visdrone_video",
        "variant": split_root.variant,
        "split": split_root.split,
        "sequence_name": sequence_name,
        "sequence_dir": str(frame_dir),
        "annotation_source": source,
        "frame_ext": frame_ext,
    }

    return VideoSequence(
        video_id=next_video_id,
        video_path=None,
        fps=fps,
        num_frames=num_frames,
        duration=duration,
        annotation_path=str(annotation_path),
        frame_dir=str(frame_dir),
        frame_pattern=f"{{frame:07d}}{frame_ext}",
        frame_number_base=1,
        video_meta=video_meta,
        boxes=boxes,
        width=image_probe.width,
        height=image_probe.height,
        num_frames_exact=num_frames_exact,
    )


def _parse_annotation_file(
    path: Path,
    *,
    split: str,
    variant: str,
    source: str,
    include_ignored: bool,
    class_filter: frozenset[int] | None,
    categories: dict[str, int],
) -> list[BoxAnnotation]:
    """Parse one VisDrone annotation file into per-frame boxes."""
    boxes: list[BoxAnnotation] = []
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            for line_no, row in enumerate(reader, start=1):
                parsed = _parse_row(
                    row,
                    line_no=line_no,
                    path=path,
                    split=split,
                    variant=variant,
                    source=source,
                    include_ignored=include_ignored,
                    class_filter=class_filter,
                    categories=categories,
                )
                if parsed is not None:
                    boxes.append(parsed)
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Could not read VisDrone annotation file %s: %s", path, exc)
    return boxes


def _parse_row(
    row: list[str],
    *,
    line_no: int,
    path: Path,
    split: str,
    variant: str,
    source: str,
    include_ignored: bool,
    class_filter: frozenset[int] | None,
    categories: dict[str, int],
) -> BoxAnnotation | None:
    """Parse one VisDrone row, returning None when it should be skipped."""
    if _is_blank_or_comment(row):
        return None
    base = _parse_base_row(row, line_no=line_no, path=path)
    if base is None:
        return None

    is_ignored = base.category_id == 0 or (source == "gt" and base.score <= 0)
    if is_ignored and not include_ignored:
        return None
    if class_filter is not None and base.category_id not in class_filter:
        return None
    if source == "gt" and base.target_id <= 0 and not is_ignored:
        _warn_bad_row(path, line_no, "GT target id must be positive for non-ignored rows")
        return None

    track_id = base.target_id if base.target_id > 0 else -line_no
    category_id, category_uri, category_name = _category_for_visdrone_class(base.category_id, categories)
    attributes = _row_attributes(base, source=source, split=split, variant=variant)
    sequence_name = path.stem

    return BoxAnnotation(
        track_uuid=f"{split}:{sequence_name}:{source}:{track_id}",
        track_id=track_id,
        category_id=category_id,
        category_uri=category_uri,
        category_name=category_name,
        bbox=(base.left, base.top, base.width, base.height),
        attributes=attributes,
        frame_index=base.frame - 1,
        timestamp=None,
    )


def _parse_base_row(row: list[str], *, line_no: int, path: Path) -> _BaseRow | None:
    """Parse the official ten-column VisDrone row shape."""
    if len(row) < 8:
        _warn_bad_row(path, line_no, "expected at least 8 columns")
        return None
    frame = _parse_positive_frame(row[0], line_no=line_no, path=path)
    target_id = _parse_required_int(row[1], field="target id", line_no=line_no, path=path)
    bbox = _parse_bbox_fields(row, line_no=line_no, path=path)
    score = _parse_required_float(row[6], field="score", line_no=line_no, path=path)
    category_id = _parse_category_id(row[7], line_no=line_no, path=path)
    truncation_ok, truncation = _parse_optional_int(row, 8, field="truncation", line_no=line_no, path=path)
    occlusion_ok, occlusion = _parse_optional_int(row, 9, field="occlusion", line_no=line_no, path=path)
    if (
        frame is None
        or target_id is None
        or bbox is None
        or score is None
        or category_id is None
        or not truncation_ok
        or not occlusion_ok
    ):
        return None
    left, top, width, height = bbox
    return _BaseRow(
        frame=frame,
        target_id=target_id,
        left=left,
        top=top,
        width=width,
        height=height,
        score=score,
        category_id=category_id,
        truncation=truncation,
        occlusion=occlusion,
    )


def _parse_positive_frame(value: object, *, line_no: int, path: Path) -> int | None:
    """Parse a positive 1-based frame number."""
    frame = _parse_int(value)
    if frame is None or frame <= 0:
        _warn_bad_row(path, line_no, "frame number must be a positive integer")
        return None
    return frame


def _parse_required_int(value: object, *, field: str, line_no: int, path: Path) -> int | None:
    """Parse a required integer field with a row-level warning on failure."""
    parsed = _parse_int(value)
    if parsed is None:
        _warn_bad_row(path, line_no, f"{field} must be an integer")
    return parsed


def _parse_required_float(value: object, *, field: str, line_no: int, path: Path) -> float | None:
    """Parse a required finite float field with a row-level warning on failure."""
    parsed = _parse_float(value)
    if parsed is None:
        _warn_bad_row(path, line_no, f"{field} must be a finite number")
    return parsed


def _parse_category_id(value: object, *, line_no: int, path: Path) -> int | None:
    """Parse a non-negative VisDrone object category id."""
    category_id = _parse_required_int(value, field="object category", line_no=line_no, path=path)
    if category_id is not None and category_id < 0:
        _warn_bad_row(path, line_no, "object category must be non-negative")
        return None
    return category_id


def _parse_bbox_fields(row: list[str], *, line_no: int, path: Path) -> tuple[float, float, float, float] | None:
    """Parse VisDrone bbox columns as finite xywh with positive size."""
    left = _parse_float(row[2])
    top = _parse_float(row[3])
    width = _parse_float(row[4])
    height = _parse_float(row[5])
    if left is None or top is None or width is None or height is None or width <= 0 or height <= 0:
        _warn_bad_row(path, line_no, "bbox must contain finite positive width/height")
        return None
    return left, top, width, height


def _parse_optional_int(
    row: list[str],
    index: int,
    *,
    field: str,
    line_no: int,
    path: Path,
) -> tuple[bool, int | None]:
    """Parse an optional integer field, distinguishing absent from malformed."""
    if len(row) <= index:
        return True, None
    parsed = _parse_int(row[index])
    if parsed is None:
        _warn_bad_row(path, line_no, f"{field} must be an integer")
        return False, None
    return True, parsed


def _row_attributes(base: _BaseRow, *, source: str, split: str, variant: str) -> dict[str, Any]:
    """Build source-preserving attributes for one parsed row."""
    attributes: dict[str, Any] = {
        "source_format": "visdrone_video",
        "variant": variant,
        "split": split,
        "annotation_source": source,
        "visdrone_frame": base.frame,
        "visdrone_target_id": base.target_id,
        "visdrone_category_id": base.category_id,
        "visdrone_score": base.score,
    }
    if source == "det":
        attributes["score"] = base.score
    else:
        attributes["confidence"] = base.score
    if base.truncation is not None and base.truncation != -1:
        attributes["truncation"] = base.truncation
    if base.occlusion is not None and base.occlusion != -1:
        attributes["occlusion"] = base.occlusion
    return attributes


def _category_for_visdrone_class(category_id: int, categories: dict[str, int]) -> tuple[int, str, str]:
    """Return model category fields for a VisDrone class id."""
    name = _VISDRONE_CLASS_NAMES.get(category_id, f"class_{category_id}")
    uri = f"visdrone_video/{name}"
    categories.setdefault(uri, category_id)
    return category_id, uri, name


def _probe_images(frame_dir: Path, frame_ext: str, *, enabled: bool) -> _ImageProbe:
    """Count frame files and optionally read the first frame via OpenCV."""
    frame_count = _count_frame_files(frame_dir, frame_ext)
    if frame_count is None:
        _warn_no_matching_frames(frame_dir, frame_ext)
    if not enabled:
        return _ImageProbe(frame_count=frame_count)

    frame_paths = _frame_paths(frame_dir, frame_ext)
    if not frame_paths:
        return _ImageProbe(frame_count=frame_count)
    try:
        import cv2  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("OpenCV not installed; skipping VisDrone image probing (install databridge[video])")
        return _ImageProbe(frame_count=frame_count)

    first = frame_paths[0]
    image = cv2.imread(str(first), cv2.IMREAD_UNCHANGED)
    if image is None:
        logger.warning("Could not read VisDrone frame image for probing: %s", first)
        return _ImageProbe(frame_count=frame_count)
    height, width = image.shape[:2]
    return _ImageProbe(frame_count=frame_count, width=int(width), height=int(height))


def _frame_paths(frame_dir: Path, frame_ext: str) -> list[Path]:
    """Return sorted frame paths matching the configured extension."""
    if not frame_dir.is_dir():
        return []
    ext = frame_ext.lower()
    try:
        with os.scandir(frame_dir) as entries:
            paths = [Path(entry.path) for entry in entries if _entry_is_matching_file(entry, ext)]
    except OSError as exc:
        logger.warning("Could not list VisDrone frame directory %s: %s", frame_dir, exc)
        return []
    return sorted(paths)


def _count_frame_files(frame_dir: Path, frame_ext: str) -> int | None:
    """Count matching frame files without materializing/sorting Path objects."""
    if not frame_dir.is_dir():
        return None
    ext = frame_ext.lower()
    try:
        with os.scandir(frame_dir) as entries:
            count = sum(1 for entry in entries if _entry_is_matching_file(entry, ext))
    except OSError as exc:
        logger.warning("Could not list VisDrone frame directory %s: %s", frame_dir, exc)
        return None
    return count if count > 0 else None


def _entry_is_matching_file(entry: os.DirEntry[str], ext: str) -> bool:
    """Return True when an os.scandir entry is a frame file for ``ext``."""
    try:
        return entry.is_file() and (not ext or Path(entry.name).suffix.lower() == ext)
    except OSError:
        return False


def _warn_no_matching_frames(frame_dir: Path, frame_ext: str) -> None:
    """Warn when a present frame directory has no files for the configured extension."""
    if frame_dir.is_dir():
        logger.warning("No VisDrone frame images matching frame_ext=%r found in %s", frame_ext, frame_dir)


def _max_frame_index(boxes: Iterable[BoxAnnotation]) -> int | None:
    """Return count-like max frame index + 1 from parsed boxes."""
    max_index = max((box.frame_index for box in boxes), default=-1)
    return max_index + 1 if max_index >= 0 else None


def _infer_split(name: str) -> str:
    """Infer split metadata from an official split directory name."""
    normalized = name.lower().replace("_", "-")
    for split, tokens in _SPLIT_TOKENS:
        if any(token in normalized for token in tokens):
            return split
    return name


def _infer_variant(name: str) -> str:
    """Infer VisDrone video variant from an official split directory name."""
    normalized = name.lower()
    if "mot" in normalized:
        return "mot"
    if "vid" in normalized:
        return "vid"
    return "unknown"


def _normalize_variant(value: VisDroneVariant | str) -> str:
    """Validate and normalize a user-provided variant option."""
    variant = str(value).lower().strip()
    if variant not in _VALID_VARIANTS:
        valid = ", ".join(sorted(_VALID_VARIANTS))
        raise ValueError(f"variant must be one of {valid}; got {value!r}")
    return variant


def _normalize_source(value: VisDroneAnnotationSource | str) -> str:
    """Validate and normalize a user-provided annotation source option."""
    source = str(value).lower().strip()
    if source not in _VALID_SOURCES:
        valid = ", ".join(sorted(_VALID_SOURCES))
        raise ValueError(f"annotation_source must be one of {valid}; got {value!r}")
    return source


def _normalize_classes(classes: Collection[int] | None) -> frozenset[int] | None:
    """Coerce a user-provided class allowlist to integers."""
    if classes is None:
        return None
    try:
        return frozenset(int(value) for value in classes)
    except (TypeError, ValueError) as exc:
        raise ValueError("classes must contain integer VisDrone category IDs") from exc


def _normalize_extension(value: str) -> str:
    """Return a safe image extension with a leading dot."""
    stripped = value.strip()
    ext = stripped if stripped.startswith(".") else f".{stripped}" if stripped else ""
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    if ext and ext != "." and ".." not in ext and all(char in allowed for char in ext):
        return ext
    raise ValueError("frame_ext must be a safe file extension such as '.jpg'")


def _coerce_non_negative_float(value: object, *, option: str) -> float:
    """Coerce an option value to a finite, non-negative float."""
    parsed = _parse_float(value)
    if parsed is None or parsed < 0:
        raise ValueError(f"{option} must be a finite non-negative number")
    return parsed


def _is_blank_or_comment(row: list[str]) -> bool:
    """Return True for empty CSV rows and comment rows."""
    return not row or not row[0].strip() or row[0].lstrip().startswith("#")


def _parse_int(value: object) -> int | None:
    """Parse an integer field, accepting integer-looking floats like ``1.0``."""
    number = _parse_float(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _parse_float(value: object) -> float | None:
    """Parse a finite float field."""
    if isinstance(value, bool):
        return None
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _warn_bad_row(path: Path, line_no: int, reason: str) -> None:
    """Log a row-level skip in a consistent form."""
    logger.warning("Skipping malformed VisDrone row %s:%d (%s)", path, line_no, reason)
