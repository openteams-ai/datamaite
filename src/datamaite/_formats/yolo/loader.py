"""YOLO/Ultralytics dataset loaders.

Two task variants are registered under the shared ``DatasetFormat.YOLO`` family:

* ``Task.IC``: the ImageFolder-style classification layout
  (``train/cat/0001.jpg`` or ``cat/0001.jpg``).
* ``Task.OD``: the standard YOLO detection layout with image files mirrored by
  ``.txt`` label files (``images/train/0001.jpg`` + ``labels/train/0001.txt``),
  including the common ``train/images`` + ``train/labels`` variant.

The task axis keeps the two variants independent: use ``load_ic(...,
dataset_format="yolo")`` for classification and ``load_od(...,
dataset_format="yolo")`` for object detection.
"""

from __future__ import annotations

import ast
import logging
import math
import struct
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar

from datamaite._formats.yolo._common import (
    IMAGE_EXTENSIONS,
    infer_split,
    normalize_extensions,
    ordered_unique,
    relative_posix,
    safe_children,
    split_sort_key,
    within,
)
from datamaite._types import DatasetFormat, Task
from datamaite.geometry import from_yolo, has_positive_area
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

_YOLO_YAML_NAMES = ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml")
_OD_SPLIT_KEYS = ("train", "val", "test")
_SOFS = frozenset({0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF})


@register_loader
class YoloImageClassificationLoader(Loader):
    """Load YOLO/Ultralytics image-classification folder datasets."""

    task: ClassVar[Task] = Task.IC
    format = DatasetFormat.YOLO
    variant: ClassVar[str] = "default"

    @classmethod
    def sniff(cls, root: str | Path) -> bool:
        path = Path(root)
        if not path.is_dir():
            return False
        return _looks_like_yolo_classification_root(path, IMAGE_EXTENSIONS)

    def load(
        self,
        root: str | Path,
        *,
        image_extensions: Collection[str] | str | None = None,
        **_: Any,
    ) -> ImageClassificationDataset:
        """Read a YOLO classification dataset root.

        Images are discovered directly under each class folder (the documented
        flat ``<split>/<class>/<image>`` layout), matching :meth:`sniff`. Class
        indices are derived from the sorted class-folder names; an existing
        ``data.yaml`` is not consulted -- the on-disk folder layout is the
        source of truth for class names and order.
        """
        root_path = Path(root)
        if not root_path.is_dir():
            logger.warning("YOLO image-classification root is not a directory: %s", root_path)
            return ImageClassificationDataset(samples=(), dataset_metadata=DatasetMetadata(source_dataset="yolo"))

        extensions = normalize_extensions(image_extensions)
        records = _discover_classification_records(root_path, extensions)
        if not records:
            logger.warning("No YOLO image-classification images found in %s", root_path)
            return ImageClassificationDataset(samples=(), dataset_metadata=DatasetMetadata(source_dataset="yolo"))

        class_names = tuple(sorted({record[2] for record in records}))
        class_to_id = {name: idx for idx, name in enumerate(class_names)}
        taxonomy = Taxonomy(
            entries=tuple(CategoryEntry(source_id=idx, name=name) for idx, name in enumerate(class_names)),
            source_dataset="yolo",
            id_density="dense",
            ordered_names=class_names,
        )
        splits = tuple(ordered_unique(record[1] for record in records if record[1] is not None))
        samples = tuple(
            ImageClassificationSample(
                image_id=rel_path,
                path_or_uri=str(image_path),
                file_name=rel_path,
                split=split,
                labels=(
                    ClassificationLabel(
                        category_id=class_to_id[class_name],
                        source_category_id=class_to_id[class_name],
                        category_name=class_name,
                    ),
                ),
                metadata={"source_format": "yolo", "variant": self.variant},
            )
            for image_path, split, class_name, rel_path in records
        )
        logger.info(
            "Loaded %d YOLO image-classification image(s), %d class(es) from %s",
            len(samples),
            len(class_names),
            root_path,
        )
        return ImageClassificationDataset(
            samples=samples,
            dataset_metadata=DatasetMetadata(taxonomy=taxonomy, source_dataset="yolo", splits=splits),
            dataset_id="yolo",
        )


@register_loader
class YoloObjectDetectionLoader(Loader):
    """Load YOLO/Ultralytics object-detection datasets."""

    task: ClassVar[Task] = Task.OD
    format = DatasetFormat.YOLO
    variant: ClassVar[str] = "default"

    @classmethod
    def sniff(cls, root: str | Path) -> bool:
        path = Path(root)
        if not path.is_dir():
            return False
        return _looks_like_yolo_od_root(path, IMAGE_EXTENSIONS)

    def load(
        self,
        root: str | Path,
        *,
        image_extensions: Collection[str] | str | None = None,
        **_: Any,
    ) -> ObjectDetectionDataset:
        """Read a YOLO detection dataset root.

        Supported layouts include both common Ultralytics arrangements::

            root/images/train/*.jpg      root/labels/train/*.txt
            root/train/images/*.jpg      root/train/labels/*.txt

        plus the split-less ``root/images`` + ``root/labels`` variant. A
        ``data.yaml``/``data.yml`` file is used for class names and, when it
        declares split paths, for image discovery. Labels are standard YOLO
        ``class cx cy w h`` rows with normalized center boxes; a sixth value is
        accepted as a confidence score for prediction-style TXT files.
        """
        root_path = Path(root)
        if not root_path.is_dir():
            logger.warning("YOLO object-detection root is not a directory: %s", root_path)
            return ObjectDetectionDataset(samples=(), dataset_metadata=DatasetMetadata(source_dataset="yolo"))

        extensions = normalize_extensions(image_extensions)
        yaml_path = _find_yolo_yaml(root_path)
        yaml_data = _read_data_yaml(yaml_path) if yaml_path is not None else {}
        records = _discover_od_records(root_path, extensions, yaml_data=yaml_data, yaml_path=yaml_path)
        if not records:
            logger.warning("No YOLO object-detection images found in %s", root_path)
            return ObjectDetectionDataset(samples=(), dataset_metadata=DatasetMetadata(source_dataset="yolo"))

        names = _names_from_yaml(yaml_data.get("names")) if yaml_data else ()
        taxonomy = _build_od_taxonomy(names, records)
        names_by_id = taxonomy.index2label()
        samples: list[ImageObjectDetectionSample] = []
        for record in records:
            width, height = _read_image_size(record.image_path)
            detections = _load_label_file(
                record.label_path,
                image_width=width,
                image_height=height,
                names_by_id=names_by_id,
            )
            samples.append(
                ImageObjectDetectionSample(
                    image_id=relative_posix(record.image_path, root_path),
                    path_or_uri=str(record.image_path),
                    file_name=record.file_name,
                    width=width,
                    height=height,
                    split=record.split,
                    detections=detections,
                    metadata={
                        "source_format": "yolo",
                        "variant": self.variant,
                        "source_file_name": relative_posix(record.image_path, root_path),
                        "label_file": relative_posix(record.label_path, root_path),
                    },
                )
            )

        splits = tuple(ordered_unique(record.split for record in records if record.split is not None))
        logger.info(
            "Loaded %d YOLO object-detection image(s), %d annotation(s), %d class(es) from %s",
            len(samples),
            sum(len(sample.detections) for sample in samples),
            len(taxonomy.entries),
            root_path,
        )
        return ObjectDetectionDataset(
            samples=tuple(samples),
            dataset_metadata=DatasetMetadata(taxonomy=taxonomy, source_dataset="yolo", splits=splits),
            dataset_id="yolo",
        )


# ---------------------------------------------------------------------------
# Classification loader helpers
# ---------------------------------------------------------------------------


def _looks_like_yolo_classification_root(root: Path, extensions: frozenset[str]) -> bool:
    """Shallow, cheap sniff for split/class/image or class/image layouts."""
    for child in safe_children(root):
        if not child.is_dir():
            continue
        if _is_classification_split_dir(child, extensions):
            return True
        if infer_split(child.name) is None and _has_direct_image(child, extensions):
            return True
    return False


def _has_direct_image(path: Path, extensions: frozenset[str]) -> bool:
    return any(child.is_file() and child.suffix.lower() in extensions for child in safe_children(path))


def _is_classification_split_dir(child: Path, extensions: frozenset[str]) -> bool:
    """Whether ``child`` is a split directory in the ``<split>/<class>/<image>`` layout."""
    if infer_split(child.name) is None:
        return False
    return any(sub.is_dir() and _has_direct_image(sub, extensions) for sub in safe_children(child))


def _discover_classification_records(root: Path, extensions: frozenset[str]) -> list[tuple[Path, str | None, str, str]]:
    """Return ``(image_path, split, class_name, rel_path)`` rows."""
    split_dirs = [
        (child, infer_split(child.name))
        for child in safe_children(root)
        if child.is_dir() and _is_classification_split_dir(child, extensions)
    ]
    records: list[tuple[Path, str | None, str, str]] = []
    if split_dirs:
        for split_dir, split in sorted(split_dirs, key=lambda item: (split_sort_key(item[1]), item[0].name)):
            records.extend(_classification_records_from_class_dirs(root, split_dir, split=split, extensions=extensions))
    else:
        records.extend(_classification_records_from_class_dirs(root, root, split=None, extensions=extensions))
    return sorted(records, key=lambda row: row[3])


def _classification_records_from_class_dirs(
    root: Path,
    base: Path,
    *,
    split: str | None,
    extensions: frozenset[str],
) -> list[tuple[Path, str | None, str, str]]:
    records: list[tuple[Path, str | None, str, str]] = []
    for class_dir in safe_children(base):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        # Direct children only -- the documented flat layout, and consistent with
        # the shallow ``sniff``. (``rglob`` would silently flatten nested
        # subdirectories into the class and diverge from autodetect.)
        for image_path in safe_children(class_dir):
            if not image_path.is_file() or image_path.suffix.lower() not in extensions:
                continue
            if image_path.is_symlink() and not within(image_path, root):
                logger.warning("Skipping symlinked image escaping the dataset root: %s", image_path)
                continue
            records.append((image_path, split, class_name, relative_posix(image_path, root)))
    return records


# ---------------------------------------------------------------------------
# Object-detection loader helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _OdRecord:
    image_path: Path
    label_path: Path
    split: str | None
    file_name: str


def _looks_like_yolo_od_root(root: Path, extensions: frozenset[str]) -> bool:
    yaml_path = _find_yolo_yaml(root)
    if yaml_path is not None:
        yaml_data = _read_data_yaml(yaml_path) or {}
        if _discover_od_records(root, extensions, yaml_data=yaml_data, yaml_path=yaml_path, limit=1):
            return True
    return bool(_discover_od_records(root, extensions, yaml_data={}, yaml_path=None, limit=1))


def _find_yolo_yaml(root: Path) -> Path | None:
    for name in _YOLO_YAML_NAMES:
        path = root / name
        if path.is_file():
            return path
    return None


def _discover_od_records(
    root: Path,
    extensions: frozenset[str],
    *,
    yaml_data: Mapping[str, Any],
    yaml_path: Path | None,
    limit: int | None = None,
) -> list[_OdRecord]:
    records: list[_OdRecord] = []
    seen: set[Path] = set()

    if yaml_path is not None:
        base = _yaml_dataset_base(root, yaml_path, yaml_data)
        for split in _OD_SPLIT_KEYS:
            for source in _yaml_split_sources(yaml_data.get(split), base=base, yaml_path=yaml_path):
                for record in _records_from_image_source(
                    source,
                    root=root,
                    split=split,
                    extensions=extensions,
                ):
                    if _append_unique_record(records, record, seen=seen, limit=limit):
                        return records

    if not records:
        for record in _records_from_standard_od_layouts(root, extensions):
            if _append_unique_record(records, record, seen=seen, limit=limit):
                return records
    return sorted(records, key=lambda record: (split_sort_key(record.split), record.file_name, record.image_path.name))


def _append_unique_record(
    records: list[_OdRecord],
    record: _OdRecord,
    *,
    seen: set[Path],
    limit: int | None,
) -> bool:
    try:
        key = record.image_path.resolve()
    except OSError:
        key = record.image_path.absolute()
    if key in seen:
        return False
    seen.add(key)
    records.append(record)
    return limit is not None and len(records) >= limit


def _yaml_dataset_base(root: Path, yaml_path: Path, yaml_data: Mapping[str, Any]) -> Path:
    raw_path = yaml_data.get("path")
    if isinstance(raw_path, str) and raw_path.strip():
        candidate = Path(raw_path.strip())
        return candidate if candidate.is_absolute() else yaml_path.parent / candidate
    return root


def _yaml_split_sources(raw_value: Any, *, base: Path, yaml_path: Path) -> list[Path]:
    if raw_value is None:
        return []
    values = list(raw_value) if isinstance(raw_value, list | tuple) else [raw_value]
    sources: list[Path] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            continue
        path = Path(value.strip())
        path = path if path.is_absolute() else base / path
        if path.is_file() and path.suffix.lower() == ".txt":
            sources.extend(_read_image_list(path, base=base, yaml_path=yaml_path))
        else:
            sources.append(path)
    return sources


def _read_image_list(path: Path, *, base: Path, yaml_path: Path) -> list[Path]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Could not read YOLO image-list file %s: %s", path, exc)
        return []
    sources: list[Path] = []
    for raw in lines:
        text = raw.partition("#")[0].strip()
        if not text:
            continue
        candidate = Path(text)
        if not candidate.is_absolute():
            # Ultralytics treats paths in list files as relative to the dataset
            # YAML's ``path`` (or YAML directory when ``path`` is absent), not to
            # the process CWD.
            candidate = base / candidate
            if not candidate.exists():
                alt = yaml_path.parent / text
                candidate = alt if alt.exists() else candidate
        sources.append(candidate)
    return sources


def _records_from_standard_od_layouts(root: Path, extensions: frozenset[str]) -> list[_OdRecord]:
    records: list[_OdRecord] = []
    images_dir = root / "images"
    labels_dir = root / "labels"
    if images_dir.is_dir() and labels_dir.is_dir():
        for image_path in _iter_images(images_dir, extensions=extensions, root=root):
            rel = image_path.relative_to(images_dir)
            split = infer_split(rel.parts[0]) if len(rel.parts) > 1 else None
            file_name = PurePosixPath(*rel.parts[1:]).as_posix() if split is not None else rel.as_posix()
            records.append(
                _OdRecord(
                    image_path=image_path,
                    label_path=labels_dir / rel.with_suffix(".txt"),
                    split=split,
                    file_name=file_name,
                )
            )

    for child in safe_children(root):
        split = infer_split(child.name)
        if split is None or not child.is_dir():
            continue
        split_images = child / "images"
        split_labels = child / "labels"
        if not split_images.is_dir() or not split_labels.is_dir():
            continue
        for image_path in _iter_images(split_images, extensions=extensions, root=root):
            rel = image_path.relative_to(split_images)
            records.append(
                _OdRecord(
                    image_path=image_path,
                    label_path=split_labels / rel.with_suffix(".txt"),
                    split=split,
                    file_name=rel.as_posix(),
                )
            )
    return records


def _records_from_image_source(
    source: Path,
    *,
    root: Path,
    split: str,
    extensions: frozenset[str],
) -> list[_OdRecord]:
    if source.is_dir():
        label_dir = _infer_label_dir(source)
        records: list[_OdRecord] = []
        for image_path in _iter_images(source, extensions=extensions, root=root):
            rel = image_path.relative_to(source)
            records.append(
                _OdRecord(
                    image_path=image_path,
                    label_path=label_dir / rel.with_suffix(".txt"),
                    split=split,
                    file_name=rel.as_posix(),
                )
            )
        return records
    if source.is_file() and source.suffix.lower() in extensions:
        return [
            _OdRecord(
                image_path=source,
                label_path=_infer_label_path(source, root=root),
                split=split,
                file_name=_relative_image_file_name(source, root=root, split=split),
            )
        ]
    return []


def _infer_label_dir(image_dir: Path) -> Path:
    parts = list(image_dir.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "images":
            parts[index] = "labels"
            return Path(*parts)
    if infer_split(image_dir.name) is not None and image_dir.parent.name == "images":
        return image_dir.parent.parent / "labels" / image_dir.name
    return image_dir.parent / "labels"


def _infer_label_path(image_path: Path, *, root: Path) -> Path:
    try:
        rel = image_path.relative_to(root)
    except ValueError:
        return _infer_label_dir(image_path.parent) / f"{image_path.stem}.txt"
    parts = list(rel.parts)
    for index, part in enumerate(parts):
        if part == "images":
            parts[index] = "labels"
            return root.joinpath(*parts).with_suffix(".txt")
    if len(parts) > 2 and infer_split(parts[0]) is not None and parts[1] == "images":
        parts[1] = "labels"
        return root.joinpath(*parts).with_suffix(".txt")
    return root / "labels" / rel.with_suffix(".txt")


def _relative_image_file_name(image_path: Path, *, root: Path, split: str | None) -> str:
    try:
        rel = image_path.relative_to(root)
    except ValueError:
        return image_path.name
    parts = list(rel.parts)
    if parts and parts[0] == "images":
        parts.pop(0)
    if parts and split is not None and infer_split(parts[0]) == split:
        parts.pop(0)
    if parts and parts[0] == "images":
        parts.pop(0)
    return PurePosixPath(*parts).as_posix() if parts else image_path.name


def _iter_images(directory: Path, *, extensions: frozenset[str], root: Path) -> list[Path]:
    images: list[Path] = []
    try:
        candidates = sorted(directory.rglob("*"))
    except OSError as exc:
        logger.warning("Could not read YOLO image directory %s: %s", directory, exc)
        return []
    for candidate in candidates:
        if any(part.startswith(".") for part in candidate.relative_to(directory).parts):
            continue
        if not candidate.is_file() or candidate.suffix.lower() not in extensions:
            continue
        if candidate.is_symlink() and not within(candidate, root):
            logger.warning("Skipping symlinked YOLO image escaping the dataset root: %s", candidate)
            continue
        images.append(candidate)
    return images


def _build_od_taxonomy(names: Sequence[tuple[int, str]], records: Sequence[_OdRecord]) -> Taxonomy:
    if names:
        source_ids = [source_id for source_id, _name in names]
        id_density = "dense" if source_ids == list(range(len(source_ids))) else "sparse"
        return Taxonomy(
            entries=tuple(CategoryEntry(source_id=source_id, name=name) for source_id, name in names),
            source_dataset="yolo",
            id_density=id_density,
            ordered_names=tuple(name for _source_id, name in names),
        )
    class_ids = sorted(_scan_label_class_ids(records))
    id_density = "dense" if class_ids == list(range(len(class_ids))) else "sparse"
    return Taxonomy(
        entries=tuple(CategoryEntry(source_id=class_id, name=str(class_id)) for class_id in class_ids),
        source_dataset="yolo",
        id_density=id_density,
        ordered_names=tuple(str(class_id) for class_id in class_ids),
    )


def _scan_label_class_ids(records: Sequence[_OdRecord]) -> set[int]:
    class_ids: set[int] = set()
    for record in records:
        if not record.label_path.is_file():
            continue
        try:
            lines = record.label_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("Could not read YOLO label file %s: %s", record.label_path, exc)
            continue
        for raw in lines:
            text = raw.partition("#")[0].strip()
            if not text:
                continue
            fields = text.split()
            class_id = _parse_int_token(fields[0]) if fields else None
            if class_id is not None and class_id >= 0:
                class_ids.add(class_id)
    return class_ids


def _load_label_file(
    label_path: Path,
    *,
    image_width: int | None,
    image_height: int | None,
    names_by_id: Mapping[int, str],
) -> tuple[ObjectDetectionAnnotation, ...]:
    if not label_path.exists():
        return ()
    if not label_path.is_file():
        logger.warning("Skipping YOLO label path that is not a file: %s", label_path)
        return ()
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Could not read YOLO label file %s: %s", label_path, exc)
        return ()
    if image_width is None or image_height is None:
        if any(raw.partition("#")[0].strip() for raw in lines):
            logger.warning(
                "Skipping labels in %s because image dimensions could not be determined",
                label_path,
            )
        return ()
    detections: list[ObjectDetectionAnnotation] = []
    for line_number, raw in enumerate(lines, start=1):
        parsed = _parse_label_line(
            raw,
            line_number=line_number,
            label_path=label_path,
            image_width=image_width,
            image_height=image_height,
            names_by_id=names_by_id,
        )
        if parsed is not None:
            detections.append(parsed)
    return tuple(detections)


def _parse_label_line(
    raw: str,
    *,
    line_number: int,
    label_path: Path,
    image_width: int,
    image_height: int,
    names_by_id: Mapping[int, str],
) -> ObjectDetectionAnnotation | None:
    text = raw.partition("#")[0].strip()
    if not text:
        return None
    fields = text.split()
    if len(fields) not in {5, 6}:
        logger.warning("Skipping malformed YOLO label row %s:%d (expected 5 or 6 fields)", label_path, line_number)
        return None
    class_id = _parse_int_token(fields[0])
    if class_id is None or class_id < 0:
        logger.warning("Skipping YOLO label row %s:%d with invalid class id", label_path, line_number)
        return None
    coords = [_parse_float_token(value) for value in fields[1:5]]
    if any(value is None for value in coords):
        logger.warning("Skipping YOLO label row %s:%d with invalid bbox", label_path, line_number)
        return None
    cx, cy, width, height = (float(value) for value in coords if value is not None)
    if width <= 0 or height <= 0 or not all(0.0 <= value <= 1.0 for value in (cx, cy, width, height)):
        logger.warning("Skipping YOLO label row %s:%d with out-of-range normalized bbox", label_path, line_number)
        return None
    score = _parse_float_token(fields[5]) if len(fields) == 6 else None
    if len(fields) == 6 and (score is None or not 0.0 <= score <= 1.0):
        logger.warning("Skipping YOLO label row %s:%d with invalid confidence", label_path, line_number)
        return None
    bbox = from_yolo(cx, cy, width, height, float(image_width), float(image_height))
    if not has_positive_area(bbox):
        logger.warning("Skipping YOLO label row %s:%d with non-positive bbox area", label_path, line_number)
        return None
    if names_by_id and class_id not in names_by_id:
        logger.warning(
            "YOLO label row %s:%d references class id %d not defined in data.yaml names",
            label_path,
            line_number,
            class_id,
        )
    return ObjectDetectionAnnotation(
        bbox=bbox,
        category_id=class_id,
        category_name=names_by_id.get(class_id),
        source_category_id=class_id,
        score=score,
        attributes={"yolo_bbox": (cx, cy, width, height), "source_line": line_number},
    )


def _parse_int_token(value: object) -> int | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if str(parsed) == value or value == f"+{parsed}" else None


def _parse_float_token(value: object) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


# ---------------------------------------------------------------------------
# data.yaml parsing and image dimensions
# ---------------------------------------------------------------------------


def _read_data_yaml(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Could not read YOLO data YAML %s: %s", path, exc)
        return {}
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        yaml = None  # type: ignore[assignment]
    if yaml is not None:
        try:
            data = yaml.safe_load(text)
        except Exception as exc:
            logger.warning("Could not parse YOLO data YAML %s with PyYAML: %s", path, exc)
        else:
            return dict(data) if isinstance(data, dict) else {}
    return _parse_simple_yaml(text)


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Small fallback parser for the data.yaml shapes YOLO datasets usually use."""
    lines = text.splitlines()
    data: dict[str, Any] = {}
    index = 0
    while index < len(lines):
        raw = _strip_yaml_comment(lines[index])
        if not raw.strip():
            index += 1
            continue
        if raw[:1].isspace() or ":" not in raw:
            index += 1
            continue
        key, value = raw.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            data[key] = _parse_yaml_scalar(value)
            index += 1
            continue
        block: list[str] = []
        index += 1
        while index < len(lines):
            block_raw = _strip_yaml_comment(lines[index])
            if block_raw.strip() and not block_raw[:1].isspace():
                break
            if block_raw.strip():
                block.append(block_raw.strip())
            index += 1
        data[key] = _parse_yaml_block(block)
    return data


def _strip_yaml_comment(line: str) -> str:
    quote: str | None = None
    escaped = False
    chars: list[str] = []
    for char in line:
        if escaped:
            chars.append(char)
            escaped = False
            continue
        if char == "\\" and quote is not None:
            chars.append(char)
            escaped = True
            continue
        if char in {"'", '"'}:
            quote = None if quote == char else char if quote is None else quote
            chars.append(char)
            continue
        if char == "#" and quote is None:
            break
        chars.append(char)
    return "".join(chars).rstrip()


def _parse_yaml_scalar(value: str) -> Any:
    text = value.strip()
    if not text or text.lower() in {"null", "none", "~"}:
        return None
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text.strip("\"'")


def _parse_yaml_block(block: list[str]) -> Any:
    if not block:
        return None
    if all(line.startswith("-") for line in block):
        return [_parse_yaml_scalar(line[1:].strip()) for line in block]
    if all(":" in line for line in block):
        result: dict[Any, Any] = {}
        for line in block:
            key, value = line.split(":", 1)
            parsed_key = _parse_yaml_scalar(key.strip())
            result[parsed_key] = _parse_yaml_scalar(value.strip())
        return result
    return [_parse_yaml_scalar(line) for line in block]


def _names_from_yaml(raw_names: Any) -> tuple[tuple[int, str], ...]:
    """Parse YOLO ``names`` into ``(source_id, name)`` pairs.

    Ultralytics normally writes a dense list, but YAML also commonly appears as
    a mapping (``{0: person, 1: car}``). Preserve mapping keys as source ids so
    sparse/non-contiguous mappings do not silently relabel detections.
    """
    if raw_names is None:
        return ()
    if isinstance(raw_names, str):
        parsed = _parse_yaml_scalar(raw_names)
        if isinstance(parsed, Mapping | Sequence) and not isinstance(parsed, str):
            return _names_from_yaml(parsed)
        name = raw_names.strip()
        return ((0, name),) if name else ()
    if isinstance(raw_names, Mapping):
        items: list[tuple[int, str]] = []
        for key, value in raw_names.items():
            class_id = _coerce_yaml_int(key)
            if class_id is None:
                continue
            name = str(value).strip()
            if name:
                items.append((class_id, name))
        return tuple(sorted(items))
    if isinstance(raw_names, Sequence):
        return tuple((idx, name) for idx, value in enumerate(raw_names) if (name := str(value).strip()))
    return ()


def _coerce_yaml_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _read_image_size(path: Path) -> tuple[int | None, int | None]:
    try:
        with path.open("rb") as fh:
            header = fh.read(32)
            if header.startswith(b"\x89PNG\r\n\x1a\n") and header[12:16] == b"IHDR":
                width, height = struct.unpack(">II", header[16:24])
                return _positive_size(width, height)
            if header.startswith(b"GIF87a") or header.startswith(b"GIF89a"):
                width, height = struct.unpack("<HH", header[6:10])
                return _positive_size(width, height)
            if header.startswith(b"BM") and len(header) >= 26:
                width = struct.unpack("<i", header[18:22])[0]
                height = abs(struct.unpack("<i", header[22:26])[0])
                return _positive_size(width, height)
            if header.startswith(b"\xff\xd8"):
                return _read_jpeg_size(fh)
    except (OSError, struct.error) as exc:
        logger.warning("Could not read image dimensions for %s: %s", path, exc)
        return (None, None)
    logger.warning("Could not determine image dimensions for %s", path)
    return (None, None)


def _read_jpeg_size(fh: Any) -> tuple[int | None, int | None]:  # noqa: C901 - JPEG marker scan is branchy
    # The SOI marker was consumed into the initial header; start scanning after it.
    fh.seek(2)
    while True:
        marker_prefix = fh.read(1)
        if not marker_prefix:
            return (None, None)
        if marker_prefix != b"\xff":
            continue
        marker = fh.read(1)
        while marker == b"\xff":
            marker = fh.read(1)
        if not marker:
            return (None, None)
        marker_value = marker[0]
        if marker_value in {0x01, 0xD8, 0xD9} or 0xD0 <= marker_value <= 0xD7:
            continue
        length_bytes = fh.read(2)
        if len(length_bytes) != 2:
            return (None, None)
        length = struct.unpack(">H", length_bytes)[0]
        if length < 2:
            return (None, None)
        if marker_value in _SOFS:
            data = fh.read(length - 2)
            if len(data) < 5:
                return (None, None)
            height, width = struct.unpack(">HH", data[1:5])
            return _positive_size(width, height)
        fh.seek(length - 2, 1)


def _positive_size(width: int, height: int) -> tuple[int | None, int | None]:
    if width > 0 and height > 0:
        return (width, height)
    return (None, None)
