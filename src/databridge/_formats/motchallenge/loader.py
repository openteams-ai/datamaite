"""MOTChallenge loader.

The Multi-Object Tracking Benchmark (MOTChallenge) stores each sequence as a
folder of frame images plus CSV-like annotation files::

    <root>/
        train/
            MOT17-02/
                img1/000001.jpg
                gt/gt.txt
                det/det.txt
                seqinfo.ini
        test/
            ...

This module implements the loader side only: it reads standard benchmark
roots (``train`` and/or ``test`` split directories) into the neutral
:class:`databridge.model.BoxTrackDataset` model. It intentionally mirrors the
HMIE loader contract: loading is best-effort, malformed rows are skipped with
warnings, and callers who need format validation should use a validator when
one exists.
"""

from __future__ import annotations

import configparser
import csv
import logging
import math
import os
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from databridge._types import DatasetFormat
from databridge.loaders import Loader, register_loader
from databridge.model import BoxAnnotation, BoxTrackDataset, VideoSequence

logger = logging.getLogger(__name__)

_MOT_CLASS_NAMES = {
    1: "pedestrian",
    2: "person_on_vehicle",
    3: "car",
    4: "bicycle",
    5: "motorbike",
    6: "non_motorized_vehicle",
    7: "static_person",
    8: "distractor",
    9: "occluder",
    10: "occluder_on_ground",
    11: "occluder_full",
    12: "reflection",
}
_VALID_SOURCES = {"gt", "det"}


@dataclass(frozen=True)
class _SeqInfo:
    """Parsed ``seqinfo.ini`` values with MOT defaults."""

    name: str
    im_dir: str = "img1"
    frame_rate: float = 0.0
    seq_length: int | None = None
    width: int | None = None
    height: int | None = None
    im_ext: str = ".jpg"
    path: Path | None = None
    raw: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class _ImageProbe:
    """Best-effort frame-directory metadata."""

    frame_count: int | None = None
    width: int | None = None
    height: int | None = None


@dataclass(frozen=True)
class _BaseRow:
    """Parsed MOT row fields common to GT and detection rows."""

    frame: int
    mot_id: int
    left: float
    top: float
    width: float
    height: float
    confidence: float | None


@register_loader
class MotChallengeLoader(Loader):
    """Loader for standard MOTChallenge benchmark roots.

    The loader expects a full benchmark root containing ``train/`` and/or
    ``test/`` split directories. Each split contains one or more sequence
    directories with the standard MOTChallenge layout. Use
    :func:`load_motchallenge` for a convenience wrapper or
    ``databridge.load(root, dataset_format="motchallenge", ...)`` for the
    registry-dispatched entry point.
    """

    format = DatasetFormat.MOTCHALLENGE

    def load(
        self,
        root: str | Path,
        *,
        annotation_source: str = "gt",
        include_ignored: bool = False,
        classes: Collection[int] | None = None,
        class_names: Mapping[int, str] | None = None,
        probe_images: bool = False,
        **_: Any,
    ) -> BoxTrackDataset:
        """Read a MOTChallenge benchmark root into :class:`BoxTrackDataset`.

        Parameters
        ----------
        root
            Full benchmark root containing ``train/`` and/or ``test/``.
            Single-sequence roots are intentionally not auto-detected for this
            issue; pass the benchmark root so split membership is preserved.
        annotation_source
            ``"gt"`` reads ``gt/gt.txt`` ground truth. ``"det"`` reads
            ``det/det.txt`` detector proposals.
        include_ignored
            For ``annotation_source="gt"``, include rows whose MOT ``conf`` /
            valid flag is ``0`` or negative. Defaults to ``False``.
        classes
            Optional MOT class-id allowlist for GT rows (for example ``{1}``
            for pedestrians). Ignored for ``annotation_source="det"`` because
            standard ``det.txt`` rows do not carry class IDs.
        class_names
            Optional custom names for MOT class IDs, useful for MOT-style
            datasets with a non-standard ontology. Missing IDs still use the
            built-in MOTChallenge names when known, else ``class_<id>``. An
            empty mapping behaves the same as omitting the option.
        probe_images
            When True, use OpenCV (``pip install databridge[video]``) to read
            the first frame image and fill/verify dimensions. Without OpenCV,
            loading continues with a warning and metadata from ``seqinfo.ini``.

        Returns
        -------
        BoxTrackDataset
            Loaded sequences and a MOT class map. Empty when the root has no
            usable split directories or no selected annotation files.
        """
        root = Path(root)
        source = annotation_source.lower()
        if source not in _VALID_SOURCES:
            valid = ", ".join(sorted(_VALID_SOURCES))
            raise ValueError(f"annotation_source must be one of {valid}; got {annotation_source!r}")

        class_filter = _normalize_classes(classes)
        class_name_map = _normalize_class_names(class_names)
        if source == "det" and class_filter is not None:
            logger.warning(
                "Ignoring classes filter for MOTChallenge det.txt rows (standard detections have no class ID)"
            )
            class_filter = None

        split_dirs = _split_dirs(root)
        if not split_dirs:
            logger.warning("MOTChallenge root must contain train/ and/or test/ directories: %s", root)
            return BoxTrackDataset(sequences=(), categories={})

        categories: dict[str, int] = {}
        sequences: list[VideoSequence] = []
        for split, split_dir in split_dirs:
            for seq_dir in _sequence_dirs(split_dir):
                seq = _load_sequence(
                    seq_dir,
                    split=split,
                    source=source,
                    include_ignored=include_ignored,
                    class_filter=class_filter,
                    class_names=class_name_map,
                    probe_images=probe_images,
                    categories=categories,
                    next_video_id=len(sequences),
                )
                if seq is not None:
                    sequences.append(seq)

        logger.info("Loaded %d MOTChallenge sequence(s), %d categories from %s", len(sequences), len(categories), root)
        return BoxTrackDataset(sequences=tuple(sequences), categories=categories)


def load_motchallenge(
    root: str | Path,
    *,
    annotation_source: str = "gt",
    include_ignored: bool = False,
    classes: Collection[int] | None = None,
    class_names: Mapping[int, str] | None = None,
    probe_images: bool = False,
) -> BoxTrackDataset:
    """Load a standard MOTChallenge dataset root.

    Equivalent to ``databridge.load(root, dataset_format="motchallenge", ...)``.
    See :meth:`MotChallengeLoader.load` for parameter semantics.
    """
    return MotChallengeLoader().load(
        root,
        annotation_source=annotation_source,
        include_ignored=include_ignored,
        classes=classes,
        class_names=class_names,
        probe_images=probe_images,
    )


def _split_dirs(root: Path) -> list[tuple[str, Path]]:
    """Return existing standard split directories in deterministic order."""
    return [(name, root / name) for name in ("train", "test") if (root / name).is_dir()]


def _sequence_dirs(split_dir: Path) -> list[Path]:
    """Return immediate child directories that may be MOT sequences."""
    try:
        return sorted(p for p in split_dir.iterdir() if p.is_dir())
    except OSError as exc:
        logger.warning("Could not list MOTChallenge split directory %s: %s", split_dir, exc)
        return []


def _load_sequence(
    seq_dir: Path,
    *,
    split: str,
    source: str,
    include_ignored: bool,
    class_filter: frozenset[int] | None,
    class_names: Mapping[int, str],
    probe_images: bool,
    categories: dict[str, int],
    next_video_id: int,
) -> VideoSequence | None:
    """Build one sequence record, or skip it when the selected annotation file is absent."""
    info = _parse_seqinfo(seq_dir)
    annotation_path = _annotation_path(seq_dir, source)
    if annotation_path is None:
        return None

    frame_dir = seq_dir / info.im_dir
    if not frame_dir.is_dir():
        logger.warning("MOTChallenge sequence frame directory is missing: %s", frame_dir)

    image_probe = _probe_images(
        frame_dir,
        info.im_ext,
        enabled=probe_images,
        count_frames=info.seq_length is None,
    )
    width = image_probe.width or info.width
    height = image_probe.height or info.height

    boxes = _parse_annotation_file(
        annotation_path,
        source=source,
        categories=categories,
        include_ignored=include_ignored,
        class_filter=class_filter,
        class_names=class_names,
    )

    inferred_num_frames = _max_frame_index(boxes)
    counted_frames = image_probe.frame_count
    num_frames = info.seq_length or counted_frames or inferred_num_frames
    num_frames_exact = info.seq_length is not None or counted_frames is not None
    duration = (
        num_frames / info.frame_rate if num_frames_exact and num_frames is not None and info.frame_rate > 0 else None
    )

    video_meta: dict[str, Any] = {
        "format": "motchallenge",
        "split": split,
        "sequence_name": info.name,
        "sequence_dir": str(seq_dir),
        "annotation_source": source,
        "im_dir": info.im_dir,
        "im_ext": info.im_ext,
    }
    if info.path is not None:
        video_meta["seqinfo_path"] = str(info.path)
    if info.raw:
        video_meta["seqinfo"] = dict(info.raw)

    return VideoSequence(
        video_id=next_video_id,
        video_path=None,
        fps=info.frame_rate,
        num_frames=num_frames,
        duration=duration,
        annotation_path=str(annotation_path),
        frame_dir=str(frame_dir),
        frame_pattern=f"{{frame:06d}}{info.im_ext}",
        frame_number_base=1,
        video_meta=video_meta,
        boxes=boxes,
        width=width,
        height=height,
        num_frames_exact=num_frames_exact,
    )


def _annotation_path(seq_dir: Path, source: str) -> Path | None:
    """Return the selected MOT annotation path, or None with a warning if missing."""
    rel = Path("gt") / "gt.txt" if source == "gt" else Path("det") / "det.txt"
    path = seq_dir / rel
    if not path.is_file():
        logger.warning("Skipping MOTChallenge sequence with missing %s annotation file: %s", source, path)
        return None
    return path


def _parse_seqinfo(seq_dir: Path) -> _SeqInfo:
    """Parse ``seqinfo.ini`` with standard MOT defaults on missing/bad data."""
    path = seq_dir / "seqinfo.ini"
    if not path.is_file():
        logger.warning("MOTChallenge sequence is missing seqinfo.ini; using defaults: %s", seq_dir)
        return _SeqInfo(name=seq_dir.name)

    parser = configparser.ConfigParser()
    try:
        parser.read(path, encoding="utf-8")
    except (configparser.Error, OSError, UnicodeDecodeError) as exc:
        logger.warning("Could not parse MOTChallenge seqinfo.ini %s; using defaults: %s", path, exc)
        return _SeqInfo(name=seq_dir.name, path=path)

    if not parser.has_section("Sequence"):
        logger.warning("MOTChallenge seqinfo.ini has no [Sequence] section; using defaults: %s", path)
        return _SeqInfo(name=seq_dir.name, path=path)

    section = parser["Sequence"]
    raw = dict(section.items())
    name = section.get("name") or seq_dir.name
    im_dir = _sanitize_im_dir(section.get("imdir") or section.get("imDir"), seqinfo_path=path)
    im_ext = _normalize_extension(section.get("imext") or section.get("imExt") or ".jpg", seqinfo_path=path)
    return _SeqInfo(
        name=name,
        im_dir=im_dir,
        frame_rate=_coerce_positive_float(section.get("framerate") or section.get("frameRate")) or 0.0,
        seq_length=_coerce_positive_int(section.get("seqlength") or section.get("seqLength")),
        width=_coerce_positive_int(section.get("imwidth") or section.get("imWidth")),
        height=_coerce_positive_int(section.get("imheight") or section.get("imHeight")),
        im_ext=im_ext,
        path=path,
        raw=raw,
    )


def _parse_annotation_file(
    path: Path,
    *,
    source: str,
    categories: dict[str, int],
    include_ignored: bool,
    class_filter: frozenset[int] | None,
    class_names: Mapping[int, str],
) -> list[BoxAnnotation]:
    """Parse ``gt.txt`` or ``det.txt`` rows into per-frame boxes."""
    boxes: list[BoxAnnotation] = []
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            for line_no, row in enumerate(reader, start=1):
                parsed = _parse_row(
                    row,
                    line_no=line_no,
                    path=path,
                    source=source,
                    categories=categories,
                    include_ignored=include_ignored,
                    class_filter=class_filter,
                    class_names=class_names,
                )
                if parsed is not None:
                    boxes.append(parsed)
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Could not read MOTChallenge annotation file %s: %s", path, exc)
    return boxes


def _parse_row(
    row: list[str],
    *,
    line_no: int,
    path: Path,
    source: str,
    categories: dict[str, int],
    include_ignored: bool,
    class_filter: frozenset[int] | None,
    class_names: Mapping[int, str],
) -> BoxAnnotation | None:
    """Parse one MOT row, returning None when it should be skipped."""
    if _is_blank_or_comment(row):
        return None
    base = _parse_base_row(row, line_no=line_no, path=path)
    if base is None:
        return None

    mot_class_id: int | None = None
    visibility: float | None = None
    if source == "gt":
        if base.mot_id <= 0:
            _warn_bad_row(path, line_no, "GT track id must be a positive integer")
            return None
        gt_fields = _parse_gt_fields(
            row,
            line_no=line_no,
            path=path,
            confidence=base.confidence,
            include_ignored=include_ignored,
            class_filter=class_filter,
        )
        if gt_fields is None:
            return None
        mot_class_id, visibility = gt_fields

    category_id, category_uri, category_name = _category_for_mot_class(
        mot_class_id, categories, class_names=class_names
    )
    track_id = base.mot_id if source == "gt" or base.mot_id > 0 else -line_no
    attributes = _row_attributes(row, source=source, base=base, mot_class_id=mot_class_id, visibility=visibility)
    seq_name = path.parent.parent.name
    split = path.parent.parent.parent.name

    return BoxAnnotation(
        track_uuid=f"{split}:{seq_name}:{source}:{track_id}",
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
    """Parse row fields common to MOT GT and detection files."""
    if len(row) < 6:
        _warn_bad_row(path, line_no, "expected at least 6 columns")
        return None
    frame = _parse_int(row[0])
    mot_id = _parse_int(row[1])
    left = _parse_float(row[2])
    top = _parse_float(row[3])
    width = _parse_float(row[4])
    height = _parse_float(row[5])
    if frame is None or frame <= 0:
        _warn_bad_row(path, line_no, "frame number must be a positive integer")
        return None
    if mot_id is None:
        _warn_bad_row(path, line_no, "track/detection id must be an integer")
        return None
    if left is None or top is None or width is None or height is None or width <= 0 or height <= 0:
        _warn_bad_row(path, line_no, "bbox must contain finite positive width/height")
        return None
    confidence = None
    if len(row) > 6:
        confidence = _parse_float(row[6])
        if confidence is None:
            _warn_bad_row(path, line_no, "confidence/score must be a finite number")
            return None
    return _BaseRow(
        frame=frame,
        mot_id=mot_id,
        left=left,
        top=top,
        width=width,
        height=height,
        confidence=confidence,
    )


def _parse_gt_fields(
    row: list[str],
    *,
    line_no: int,
    path: Path,
    confidence: float | None,
    include_ignored: bool,
    class_filter: frozenset[int] | None,
) -> tuple[int | None, float | None] | None:
    """Parse and filter GT-only class/visibility fields."""
    if not include_ignored and confidence is not None and confidence <= 0:
        return None
    mot_class_id: int | None = None
    if len(row) > 7:
        mot_class_id = _parse_int(row[7])
        if mot_class_id is None:
            _warn_bad_row(path, line_no, "MOT class id must be an integer")
            return None
    if class_filter is not None and mot_class_id is None:
        _warn_bad_row(path, line_no, "classes filter requires a MOT class column")
        return None
    if class_filter is not None and mot_class_id not in class_filter:
        return None
    visibility: float | None = None
    if len(row) > 8:
        visibility = _parse_float(row[8])
        if visibility is None:
            _warn_bad_row(path, line_no, "visibility must be a finite number")
            return None
    return mot_class_id, visibility


def _row_attributes(
    row: list[str],
    *,
    source: str,
    base: _BaseRow,
    mot_class_id: int | None,
    visibility: float | None,
) -> dict[str, Any]:
    """Build source-preserving attributes for one parsed row."""
    attributes: dict[str, Any] = {
        "source_format": "motchallenge",
        "annotation_source": source,
        "mot_frame": base.frame,
        "mot_track_id": base.mot_id,
    }
    if base.confidence is not None:
        attributes["confidence" if source == "gt" else "score"] = base.confidence
    if mot_class_id is not None:
        attributes["mot_class_id"] = mot_class_id
    if visibility is not None:
        attributes["visibility"] = visibility
    if source == "det":
        _add_detection_world_coordinates(row, attributes)
    return attributes


def _is_blank_or_comment(row: list[str]) -> bool:
    """Return True for empty CSV rows and comment rows."""
    return not row or not row[0].strip() or row[0].lstrip().startswith("#")


def _add_detection_world_coordinates(row: list[str], attributes: dict[str, Any]) -> None:
    """Preserve optional MOT ``det.txt`` world-coordinate columns when parseable."""
    for key, index in (("world_x", 7), ("world_y", 8), ("world_z", 9)):
        if len(row) > index:
            value = _parse_float(row[index])
            if value is not None and value != -1.0:
                attributes[key] = value


def _category_for_mot_class(
    mot_class_id: int | None,
    categories: dict[str, int],
    *,
    class_names: Mapping[int, str],
) -> tuple[int, str, str | None]:
    """Return model category fields for a MOT class id."""
    if mot_class_id is None:
        return -1, "", None
    custom_name = class_names.get(mot_class_id)
    if custom_name is not None:
        name = custom_name
        uri = f"motchallenge/class_{mot_class_id}/{name}"
    else:
        name = _MOT_CLASS_NAMES.get(mot_class_id, f"class_{mot_class_id}")
        uri = f"motchallenge/{name}"
    categories.setdefault(uri, mot_class_id)
    return mot_class_id, uri, name


def _normalize_classes(classes: Collection[int] | None) -> frozenset[int] | None:
    """Coerce a user-provided class allowlist to integers."""
    if classes is None:
        return None
    try:
        return frozenset(int(value) for value in classes)
    except (TypeError, ValueError) as exc:
        raise ValueError("classes must contain integer MOT class IDs") from exc


def _normalize_class_names(class_names: Mapping[int, str] | None) -> dict[int, str]:
    """Coerce a custom class-id -> display-name map."""
    if class_names is None:
        return {}
    if not isinstance(class_names, Mapping):
        raise ValueError("class_names must be a mapping of integer MOT class IDs to names")
    if not class_names:
        return {}

    normalized: dict[int, str] = {}
    for raw_id, raw_name in class_names.items():
        try:
            class_id = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise ValueError("class_names keys must be integer MOT class IDs") from exc
        if not isinstance(raw_name, str):
            raise ValueError("class_names values must be non-empty strings")
        name = raw_name.strip()
        if not name:
            raise ValueError("class_names values must be non-empty strings")
        if "/" in name:
            raise ValueError("class_names values must not contain '/'")
        existing = normalized.get(class_id)
        if existing is not None and existing != name:
            raise ValueError(f"class_names has conflicting names for MOT class ID {class_id}")
        normalized[class_id] = name
    return normalized


def _probe_images(frame_dir: Path, im_ext: str, *, enabled: bool, count_frames: bool) -> _ImageProbe:
    """Count frame files when needed and optionally read the first frame via OpenCV."""
    if enabled:
        frame_paths = _frame_paths(frame_dir, im_ext)
        frame_count = len(frame_paths) if frame_paths else None
        if not frame_paths:
            _warn_no_matching_frames(frame_dir, im_ext)
            return _ImageProbe(frame_count=frame_count)

        try:
            import cv2  # type: ignore[import-untyped]
        except ImportError:
            logger.warning("OpenCV not installed; skipping MOTChallenge image probing (install databridge[video])")
            return _ImageProbe(frame_count=frame_count)

        first = frame_paths[0]
        image = cv2.imread(str(first), cv2.IMREAD_UNCHANGED)
        if image is None:
            logger.warning("Could not read MOTChallenge frame image for probing: %s", first)
            return _ImageProbe(frame_count=frame_count)
        height, width = image.shape[:2]
        return _ImageProbe(frame_count=frame_count, width=int(width), height=int(height))

    if not count_frames:
        return _ImageProbe()

    frame_count = _count_frame_files(frame_dir, im_ext)
    if frame_count is None:
        _warn_no_matching_frames(frame_dir, im_ext)
    return _ImageProbe(frame_count=frame_count)


def _frame_paths(frame_dir: Path, im_ext: str) -> list[Path]:
    """Return sorted frame paths matching the configured extension."""
    if not frame_dir.is_dir():
        return []
    ext = im_ext.lower()
    try:
        with os.scandir(frame_dir) as entries:
            paths = [Path(entry.path) for entry in entries if _entry_is_matching_file(entry, ext)]
    except OSError as exc:
        logger.warning("Could not list MOTChallenge frame directory %s: %s", frame_dir, exc)
        return []
    return sorted(paths)


def _count_frame_files(frame_dir: Path, im_ext: str) -> int | None:
    """Count matching frame files without materializing/sorting Path objects."""
    if not frame_dir.is_dir():
        return None
    ext = im_ext.lower()
    try:
        with os.scandir(frame_dir) as entries:
            count = sum(1 for entry in entries if _entry_is_matching_file(entry, ext))
    except OSError as exc:
        logger.warning("Could not list MOTChallenge frame directory %s: %s", frame_dir, exc)
        return None
    return count if count > 0 else None


def _entry_is_matching_file(entry: os.DirEntry[str], ext: str) -> bool:
    """Return True when an os.scandir entry is a frame file for ``ext``."""
    try:
        return entry.is_file() and (not ext or Path(entry.name).suffix.lower() == ext)
    except OSError:
        return False


def _warn_no_matching_frames(frame_dir: Path, im_ext: str) -> None:
    """Warn when a present frame directory has no files for the configured extension."""
    if frame_dir.is_dir():
        logger.warning("No MOTChallenge frame images matching imExt=%r found in %s", im_ext, frame_dir)


def _max_frame_index(boxes: Iterable[BoxAnnotation]) -> int | None:
    """Return count-like max frame index + 1 from parsed boxes."""
    max_index = max((box.frame_index for box in boxes), default=-1)
    return max_index + 1 if max_index >= 0 else None


def _sanitize_im_dir(value: str | None, *, seqinfo_path: Path) -> str:
    """Return a safe single-component frame directory name from ``imDir``."""
    raw = (value or "img1").strip()
    if raw and raw not in {".", ".."} and "/" not in raw and "\\" not in raw and ":" not in raw:
        return raw
    logger.warning("Unsafe MOTChallenge imDir %r in %s; using 'img1'", raw, seqinfo_path)
    return "img1"


def _normalize_extension(value: str, *, seqinfo_path: Path | None = None) -> str:
    """Return a safe image extension with a leading dot."""
    stripped = value.strip()
    ext = stripped if stripped.startswith(".") else f".{stripped}" if stripped else ""
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    if ext and ext != "." and ".." not in ext and all(char in allowed for char in ext):
        return ext
    if seqinfo_path is not None:
        logger.warning("Unsafe MOTChallenge imExt %r in %s; using '.jpg'", value, seqinfo_path)
    return ".jpg"


def _parse_int(value: str) -> int | None:
    """Parse an integer field, accepting integer-looking floats like ``1.0``."""
    number = _parse_float(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _parse_float(value: str) -> float | None:
    """Parse a finite float field."""
    try:
        result = float(value.strip())
    except (AttributeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _coerce_positive_float(value: Any) -> float | None:
    """Coerce a config value to a finite, positive float."""
    if value is None:
        return None
    parsed = _parse_float(str(value))
    return parsed if parsed is not None and parsed > 0 else None


def _coerce_positive_int(value: Any) -> int | None:
    """Coerce a config value to a finite, positive integer."""
    if value is None:
        return None
    parsed = _parse_int(str(value))
    return parsed if parsed is not None and parsed > 0 else None


def _warn_bad_row(path: Path, line_no: int, reason: str) -> None:
    """Log a row-level skip in a consistent form."""
    logger.warning("Skipping malformed MOTChallenge row %s:%d (%s)", path, line_no, reason)
