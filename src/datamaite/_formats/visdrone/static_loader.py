# src/datamaite/_formats/visdrone/static_loader.py
"""VisDrone Static-Images loaders (object detection + image classification).

One on-disk layout (``images/`` + ``annotations/``) read two ways under the
shared ``DatasetFormat.VISDRONE`` family. OD ports the official VisDrone-DET
reading; IC derives one classification sample per labeled box (the crop, labeled
by its VisDrone category).
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from datamaite._io import list_files
from datamaite._types import DatasetFormat, Task
from datamaite._upath import storage_options_for, to_dataset_path
from datamaite.geometry import BBox, has_positive_area
from datamaite.image_classification import ImageClassificationDataset
from datamaite.loaders import Loader, register_loader
from datamaite.object_detection import ObjectDetectionDataset
from datamaite.records import (
    ClassificationLabel,
    DatasetMetadata,
    ImageClassificationSample,
    ImageObjectDetectionSample,
    ObjectDetectionAnnotation,
)
from datamaite.taxonomy import CategoryEntry, Taxonomy

logger = logging.getLogger(__name__)

VISDRONE_STATIC_CLASSES: tuple[str, ...] = (
    "ignored regions",
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
    "others",
)
_EVAL_EXCLUDED_IDS = frozenset({0, 11})
#: Official VisDrone-DET releases ship ``.jpg`` images; loading is best-effort,
#: so the loader also reads the other still-image suffixes datamaite writers
#: copy verbatim into a VisDrone root (e.g. ``.tif`` sources arriving via the
#: ``flat_images`` loader). The VisDrone static writer only emits suffixes in
#: this set, keeping write -> reload lossless.
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
#: Official split names, most specific first: ``test-dev``/``test-challenge``
#: must match before plain ``test`` so those roots keep their identity.
_SPLIT_TOKENS = ("test-challenge", "test-dev", "train", "val", "test")
# Load order for a parent of several split roots (the usual train/val/test order).
_SPLIT_TOKENS_ORDER = ("train", "val", "test-dev", "test-challenge", "test")


@dataclass(frozen=True)
class _VisDroneRow:
    left: float
    top: float
    width: float
    height: float
    score: float
    category: int
    truncation: int
    occlusion: int
    line_number: int


def _parse_float(value: str) -> float | None:
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _parse_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def _parse_row(fields: list[str], *, path: Path, line_number: int) -> _VisDroneRow | None:
    coords = [_parse_float(v) for v in fields[:5]]
    ints = [_parse_int(v) for v in fields[5:8]]
    if any(c is None for c in coords) or any(i is None for i in ints):
        logger.warning("Skipping VisDrone row %s:%d with non-numeric fields", path, line_number)
        return None
    left, top, width, height, score = (float(c) for c in coords)  # type: ignore[misc]
    category, truncation, occlusion = (int(i) for i in ints)  # type: ignore[misc]
    box: BBox = (left, top, width, height)
    if not has_positive_area(box):
        logger.warning("Skipping VisDrone row %s:%d with non-positive box area", path, line_number)
        return None
    if not 0 <= category <= 11:
        logger.warning("Skipping VisDrone row %s:%d with out-of-range category %d", path, line_number, category)
        return None
    return _VisDroneRow(left, top, width, height, score, category, truncation, occlusion, line_number)


def parse_annotation_file(path: Path) -> list[_VisDroneRow]:
    """Parse one VisDrone-DET annotation file, skipping malformed rows with warnings."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Could not read VisDrone annotation file %s: %s", path, exc)
        return []
    rows: list[_VisDroneRow] = []
    for line_number, raw in enumerate(lines, start=1):
        text = raw.strip()
        if not text:
            continue
        fields = [f.strip() for f in text.split(",")]
        if len(fields) < 8:
            logger.warning("Skipping malformed VisDrone row %s:%d (expected >= 8 fields)", path, line_number)
            continue
        row = _parse_row(fields[:8], path=path, line_number=line_number)
        if row is not None:
            rows.append(row)
    return rows


def build_taxonomy(*, include_ignored_regions: bool) -> Taxonomy:
    """Fixed VisDrone taxonomy. OD passes True (all 12); IC default False (drops class 0)."""
    ids = range(12) if include_ignored_regions else range(1, 12)
    entries = tuple(
        CategoryEntry(
            source_id=i,
            name=VISDRONE_STATIC_CLASSES[i],
            eval_excluded=i in _EVAL_EXCLUDED_IDS,
        )
        for i in ids
    )
    source_ids = [e.source_id for e in entries]
    id_density = "dense" if source_ids == list(range(len(entries))) else "sparse"
    return Taxonomy(
        entries=entries,
        source_dataset="visdrone",
        id_density=id_density,
        ordered_names=tuple(e.name for e in entries),
    )


def infer_split(name: str) -> str | None:
    """Infer the official split from a VisDrone split-root directory name.

    Recognises all official VisDrone-DET splits — ``train``, ``val``,
    ``test-dev``, ``test-challenge``, and plain ``test`` — so a
    ``VisDrone2019-DET-test-dev`` root reloads as ``split="test-dev"``
    rather than collapsing onto ``"test"``.
    """
    tokens = [t for t in re.split(r"[^a-z]+", name.lower()) if t]
    joined = f"-{'-'.join(tokens)}-"
    for token in _SPLIT_TOKENS:
        if f"-{token}-" in joined:
            return token
    return None


def _normalize_extensions(image_extensions: Any) -> frozenset[str]:
    """Coerce a user-supplied extension spec to a lowercased, dot-prefixed set.

    ``None`` yields the built-in defaults. A bare string (``".jpg"`` or ``"jpg"``)
    is treated as a single extension, not iterated into characters; any other
    iterable of strings is normalized element-wise. Each entry is lowercased and
    gets a leading dot. Blank entries are dropped; an all-blank spec falls back to
    the defaults so a stray ``""`` never silently loads zero images.
    """
    if image_extensions is None:
        return IMAGE_EXTENSIONS
    items = [image_extensions] if isinstance(image_extensions, str) else list(image_extensions)
    normalized: set[str] = set()
    for raw in items:
        ext = str(raw).strip().lower()
        if not ext:
            continue
        normalized.add(ext if ext.startswith(".") else f".{ext}")
    return frozenset(normalized) if normalized else IMAGE_EXTENSIONS


def iter_images(images_dir: Path, extensions: frozenset[str]) -> list[Path]:
    """Sorted image files directly under ``images/`` (VisDrone-DET is flat)."""
    return [path for path in list_files(images_dir) if path.suffix.lower() in extensions]


def _has_matching_image(images_dir: Path, stem: str, extensions: frozenset[str]) -> bool:
    return any((images_dir / f"{stem}{ext}").is_file() for ext in extensions)


def _warn_orphan_annotations(annotations_dir: Path, images_dir: Path, extensions: frozenset[str]) -> None:
    image_stems = {path.stem for path in iter_images(images_dir, extensions)}
    for ann in (path for path in list_files(annotations_dir) if path.suffix.lower() == ".txt"):
        if ann.stem not in image_stems:
            logger.warning("VisDrone annotation %s has no matching image; skipping", ann)


def _row_to_detection(row: _VisDroneRow) -> ObjectDetectionAnnotation:
    return ObjectDetectionAnnotation(
        bbox=(row.left, row.top, row.width, row.height),
        category_id=row.category,
        category_name=VISDRONE_STATIC_CLASSES[row.category],
        source_category_id=row.category,
        score=None,  # ground truth carries no confidence; raw flag kept in attributes
        attributes={
            "visdrone_category_id": row.category,
            "visdrone_score": row.score,
            "truncation": row.truncation,
            "occlusion": row.occlusion,
            "source_line": row.line_number,
        },
    )


def _splits_of(split_roots: list[Path]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(split for root in split_roots if (split := infer_split(root.name)) is not None))


def _is_split_root(path: Path) -> bool:
    return (path / "images").is_dir() and (path / "annotations").is_dir()


def _split_roots(root: Path) -> list[Path]:
    """Split roots to read: ``root`` itself, or its child split roots.

    Mirrors the VisDrone video loader: ``root`` may be one official split root
    (``images/`` + ``annotations/``) or a parent of several, such as the
    ``VisDrone2019-DET-train/`` and ``VisDrone2019-DET-val/`` directories the
    VisDrone writers emit, so loading a write destination returns its data.
    """
    if _is_split_root(root):
        return [root]
    try:
        children = sorted(path for path in root.iterdir() if path.is_dir())
    except OSError:
        children = []
    split_roots = [path for path in children if _is_split_root(path)]
    if not split_roots:
        logger.warning(
            "VisDrone static root must contain images/ and annotations/, or child split roots that do: %s", root
        )
    return sorted(split_roots, key=_split_sort_key)


def _split_sort_key(path: Path) -> tuple[int, str]:
    split = infer_split(path.name)
    return (_SPLIT_TOKENS_ORDER.index(split) if split in _SPLIT_TOKENS_ORDER else len(_SPLIT_TOKENS_ORDER), path.name)


def _id_prefixes(split_roots: list[Path], extensions: frozenset[str]) -> dict[Path, str]:
    """Per-split-root image-id prefix: empty unless an image stem repeats across split roots.

    Single-root loads keep bare stems (the long-standing ids); a stem present in
    two split roots gets ``<split-root-name>/`` so datum ids stay unique.
    """
    if len(split_roots) < 2:
        return dict.fromkeys(split_roots, "")
    stems = [{path.stem for path in iter_images(root / "images", extensions)} for root in split_roots]
    repeated = set().union(*(a & b for i, a in enumerate(stems) for b in stems[i + 1 :]))
    return {root: (f"{root.name}/" if stems[i] & repeated else "") for i, root in enumerate(split_roots)}


def _looks_like_visdrone_static(root: str | Path) -> bool:
    """True when ``root`` (or a child split root) has a VisDrone-shaped annotation line."""
    path = to_dataset_path(root)
    candidates = [path] if _is_split_root(path) else _child_split_roots_quiet(path)
    return any(_has_visdrone_annotation_line(candidate / "annotations") for candidate in candidates)


def _child_split_roots_quiet(path: Path) -> list[Path]:
    try:
        return sorted(child for child in path.iterdir() if child.is_dir() and _is_split_root(child))
    except OSError:
        return []


def _has_visdrone_annotation_line(annotations_dir: Path) -> bool:
    for ann in sorted(annotations_dir.glob("*.txt")):
        try:
            lines = ann.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for raw in lines:
            text = raw.strip()
            if not text:
                continue
            fields = [f.strip() for f in text.split(",")]
            return len(fields) >= 8 and all(_parse_float(f) is not None for f in fields[:8])
    return False


@register_loader
class VisDroneObjectDetectionLoader(Loader):
    """Load VisDrone-DET still images as an object-detection dataset."""

    task: ClassVar[Task] = Task.OD
    format = DatasetFormat.VISDRONE
    variant: ClassVar[str] = "default"
    supports_remote: ClassVar[bool] = True

    @classmethod
    def sniff(cls, root: str | Path) -> bool:
        return _looks_like_visdrone_static(root)

    def load(
        self,
        root: str | Path,
        *,
        image_extensions: Any = None,
        storage_options: dict[str, Any] | None = None,
        **_: Any,
    ) -> ObjectDetectionDataset:
        root = to_dataset_path(root, storage_options)
        split_roots = _split_roots(root)
        extensions = _normalize_extensions(image_extensions)
        taxonomy = build_taxonomy(include_ignored_regions=True)
        prefixes = _id_prefixes(split_roots, extensions)

        samples: list[ImageObjectDetectionSample] = []
        for split_root in split_roots:
            images_dir, annotations_dir = split_root / "images", split_root / "annotations"
            split = infer_split(split_root.name)
            for image_path in iter_images(images_dir, extensions):
                stem = image_path.stem
                ann_path = annotations_dir / f"{stem}.txt"
                rows = parse_annotation_file(ann_path) if ann_path.is_file() else []
                samples.append(
                    ImageObjectDetectionSample(
                        image_id=f"{prefixes[split_root]}{stem}",
                        path_or_uri=str(image_path),
                        file_name=image_path.name,
                        split=split,
                        detections=tuple(_row_to_detection(r) for r in rows),
                        metadata={
                            "source_format": "visdrone",
                            "variant": self.variant,
                            "source_file_name": f"images/{image_path.name}",
                            "annotation_file": f"annotations/{stem}.txt" if ann_path.is_file() else None,
                        },
                    )
                )
            _warn_orphan_annotations(annotations_dir, images_dir, extensions)

        if not samples:
            if split_roots:
                logger.warning("No VisDrone images found under %s", root)
            return ObjectDetectionDataset(samples=(), dataset_metadata=DatasetMetadata(source_dataset="visdrone"))
        splits = _splits_of(split_roots)
        logger.info("Loaded %d VisDrone OD image(s) from %s", len(samples), root)
        return ObjectDetectionDataset(
            samples=tuple(samples),
            dataset_metadata=DatasetMetadata(taxonomy=taxonomy, source_dataset="visdrone", splits=splits),
            dataset_id="visdrone",
            _storage_options=storage_options_for(root),
        )


@register_loader
class VisDroneImageClassificationLoader(Loader):
    """Load VisDrone-DET still images as an image-classification dataset (object crops)."""

    task: ClassVar[Task] = Task.IC
    format = DatasetFormat.VISDRONE
    variant: ClassVar[str] = "default"
    supports_remote: ClassVar[bool] = True

    @classmethod
    def sniff(cls, root: str | Path) -> bool:
        return _looks_like_visdrone_static(root)

    def load(
        self,
        root: str | Path,
        *,
        include_ignored_regions: bool = False,
        image_extensions: Any = None,
        storage_options: dict[str, Any] | None = None,
        **_: Any,
    ) -> ImageClassificationDataset:
        root = to_dataset_path(root, storage_options)
        split_roots = _split_roots(root)
        extensions = _normalize_extensions(image_extensions)
        excluded: set[int] = set() if include_ignored_regions else {0}
        taxonomy = build_taxonomy(include_ignored_regions=include_ignored_regions)
        prefixes = _id_prefixes(split_roots, extensions)

        samples: list[ImageClassificationSample] = []
        for split_root in split_roots:
            images_dir, annotations_dir = split_root / "images", split_root / "annotations"
            split = infer_split(split_root.name)
            for image_path in iter_images(images_dir, extensions):
                stem = image_path.stem
                ann_path = annotations_dir / f"{stem}.txt"
                if not ann_path.is_file():
                    continue
                for row in parse_annotation_file(ann_path):
                    if row.category in excluded:
                        continue
                    name = VISDRONE_STATIC_CLASSES[row.category]
                    samples.append(
                        ImageClassificationSample(
                            image_id=f"{prefixes[split_root]}{stem}#{row.line_number}",
                            path_or_uri=str(image_path),
                            file_name=image_path.name,
                            width=round(row.width),
                            height=round(row.height),
                            split=split,
                            region=(row.left, row.top, row.width, row.height),
                            labels=(
                                ClassificationLabel(
                                    category_id=row.category,
                                    source_category_id=row.category,
                                    category_name=name,
                                    attributes={
                                        "visdrone_category_id": row.category,
                                        "visdrone_score": row.score,
                                        "truncation": row.truncation,
                                        "occlusion": row.occlusion,
                                    },
                                ),
                            ),
                            metadata={
                                "source_format": "visdrone",
                                "variant": self.variant,
                                "source_image": f"images/{image_path.name}",
                                "annotation_file": f"annotations/{stem}.txt",
                                "source_line": row.line_number,
                            },
                        )
                    )

        if not samples:
            if split_roots:
                logger.warning("No VisDrone classification crops found under %s", root)
            return ImageClassificationDataset(samples=(), dataset_metadata=DatasetMetadata(source_dataset="visdrone"))
        splits = _splits_of(split_roots)
        logger.info("Loaded %d VisDrone IC crop(s) from %s", len(samples), root)
        return ImageClassificationDataset(
            samples=tuple(samples),
            dataset_metadata=DatasetMetadata(taxonomy=taxonomy, source_dataset="visdrone", splits=splits),
            dataset_id="visdrone",
            _storage_options=storage_options_for(root),
        )
