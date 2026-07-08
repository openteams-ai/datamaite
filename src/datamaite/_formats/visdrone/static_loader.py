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

from datamaite._types import DatasetFormat, Task
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
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp"})
_SPLIT_TOKENS = ("train", "val", "test")


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
    """Infer train/val/test from a VisDrone split-root directory name."""
    tokens = re.split(r"[^a-z]+", name.lower())
    for token in _SPLIT_TOKENS:
        if token in tokens:
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
    return sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in extensions)


def _has_matching_image(images_dir: Path, stem: str, extensions: frozenset[str]) -> bool:
    return any((images_dir / f"{stem}{ext}").is_file() for ext in extensions)


def _warn_orphan_annotations(annotations_dir: Path, images_dir: Path, extensions: frozenset[str]) -> None:
    for ann in sorted(annotations_dir.glob("*.txt")):
        if not _has_matching_image(images_dir, ann.stem, extensions):
            logger.warning("VisDrone annotation %s has no matching image; skipping", ann)


def _row_to_detection(row: _VisDroneRow) -> ObjectDetectionAnnotation:
    return ObjectDetectionAnnotation(
        bbox=(row.left, row.top, row.width, row.height),
        category_id=row.category,
        category_name=VISDRONE_STATIC_CLASSES[row.category],
        source_category_id=row.category,
        score=None,  # ground truth carries no confidence; raw flag kept in attributes
        attributes={
            "visdrone_score": row.score,
            "truncation": row.truncation,
            "occlusion": row.occlusion,
            "source_line": row.line_number,
        },
    )


def _dirs(root: str | Path) -> tuple[Path, Path, Path] | None:
    """Return ``(root, images_dir, annotations_dir)`` when both subdirs exist."""
    root_path = Path(root)
    images_dir = root_path / "images"
    annotations_dir = root_path / "annotations"
    if not images_dir.is_dir() or not annotations_dir.is_dir():
        logger.warning("VisDrone static root missing images/ or annotations/: %s", root_path)
        return None
    return root_path, images_dir, annotations_dir


def _looks_like_visdrone_static(root: str | Path) -> bool:
    """True when ``root`` has ``images/``+``annotations/`` and a VisDrone-shaped annotation line."""
    path = Path(root)
    images_dir = path / "images"
    annotations_dir = path / "annotations"
    if not images_dir.is_dir() or not annotations_dir.is_dir():
        return False
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

    @classmethod
    def sniff(cls, root: str | Path) -> bool:
        return _looks_like_visdrone_static(root)

    def load(self, root: str | Path, *, image_extensions: Any = None, **_: Any) -> ObjectDetectionDataset:
        dirs = _dirs(root)
        if dirs is None:
            return ObjectDetectionDataset(samples=(), dataset_metadata=DatasetMetadata(source_dataset="visdrone"))
        root_path, images_dir, annotations_dir = dirs
        extensions = _normalize_extensions(image_extensions)
        split = infer_split(root_path.name)
        taxonomy = build_taxonomy(include_ignored_regions=True)

        samples: list[ImageObjectDetectionSample] = []
        for image_path in iter_images(images_dir, extensions):
            stem = image_path.stem
            ann_path = annotations_dir / f"{stem}.txt"
            rows = parse_annotation_file(ann_path) if ann_path.is_file() else []
            samples.append(
                ImageObjectDetectionSample(
                    image_id=stem,
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
            logger.warning("No VisDrone images found under %s", images_dir)
            return ObjectDetectionDataset(samples=(), dataset_metadata=DatasetMetadata(source_dataset="visdrone"))
        splits = (split,) if split is not None else ()
        logger.info("Loaded %d VisDrone OD image(s) from %s", len(samples), root_path)
        return ObjectDetectionDataset(
            samples=tuple(samples),
            dataset_metadata=DatasetMetadata(taxonomy=taxonomy, source_dataset="visdrone", splits=splits),
            dataset_id="visdrone",
        )


@register_loader
class VisDroneImageClassificationLoader(Loader):
    """Load VisDrone-DET still images as an image-classification dataset (object crops)."""

    task: ClassVar[Task] = Task.IC
    format = DatasetFormat.VISDRONE
    variant: ClassVar[str] = "default"

    @classmethod
    def sniff(cls, root: str | Path) -> bool:
        return _looks_like_visdrone_static(root)

    def load(
        self, root: str | Path, *, include_ignored_regions: bool = False, image_extensions: Any = None, **_: Any
    ) -> ImageClassificationDataset:
        dirs = _dirs(root)
        if dirs is None:
            return ImageClassificationDataset(samples=(), dataset_metadata=DatasetMetadata(source_dataset="visdrone"))
        root_path, images_dir, annotations_dir = dirs
        extensions = _normalize_extensions(image_extensions)
        split = infer_split(root_path.name)
        excluded: set[int] = set() if include_ignored_regions else {0}
        taxonomy = build_taxonomy(include_ignored_regions=include_ignored_regions)

        samples: list[ImageClassificationSample] = []
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
                        image_id=f"{stem}#{row.line_number}",
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
            logger.warning("No VisDrone classification crops found under %s", root_path)
            return ImageClassificationDataset(samples=(), dataset_metadata=DatasetMetadata(source_dataset="visdrone"))
        splits = (split,) if split is not None else ()
        logger.info("Loaded %d VisDrone IC crop(s) from %s", len(samples), root_path)
        return ImageClassificationDataset(
            samples=tuple(samples),
            dataset_metadata=DatasetMetadata(taxonomy=taxonomy, source_dataset="visdrone", splits=splits),
            dataset_id="visdrone",
        )
