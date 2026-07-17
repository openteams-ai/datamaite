"""Hugging Face Vision (still-image) dataset loaders (IR-3.2-S-2).

Two task variants are registered under the shared
``DatasetFormat.HUGGINGFACE_VISION`` family, mirroring how the Hugging Face
Hub documents local image dataset repositories:

* ``Task.IC``: the ImageFolder classification convention — image files under
  class-named folders (``cat/0001.jpg``), optionally nested under split
  folders (``train/cat/0001.jpg``), or a ``metadata.csv`` / ``metadata.jsonl``
  file with a required ``file_name`` column and a ``label`` column.
* ``Task.OD``: the ImageFolder object-detection convention — images plus a
  metadata file whose ``objects`` column carries parallel lists
  (``{"bbox": [[x, y, w, h], ...], "categories": [...]}``, with optional
  ``id`` / ``area`` lists). Boxes are absolute-pixel COCO-style ``xywh``,
  which is also datamaite's canonical box format.

Like :mod:`datamaite._formats.huggingface_video_classification`, this reads
the local repository layout directly without requiring the Hugging Face
``datasets`` package. Experimental ``metadata.parquet`` reading is attempted
when optional ``pyarrow`` or ``pandas`` support is installed.

Scope: this is support for the *local ImageFolder-compatible layout* — what
``datasets.load_dataset("imagefolder", data_dir=...)`` reads from disk — not
general Hugging Face ``datasets`` support. Hub repositories, Arrow/parquet
dataset dumps, dataset scripts, and full feature schemas (e.g. ``ClassLabel``
name tables) are out of scope. ``metadata.csv`` rows whose ``objects`` value
is a JSON-encoded string are also parsed as a datamaite extension (the
matching writer's ``metadata_format="csv"``); Hugging Face documents the OD
``objects`` convention for ``metadata.jsonl`` only.

Neither loader implements ``sniff``: a plain folder-of-class-folders matches
far too much (see the dataset-structures discussion in #40), so the format is
explicit opt-in via ``load_ic(root, dataset_format="huggingface_vision")`` or
``load_od(root, dataset_format="huggingface_vision")``.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
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
from datamaite.taxonomy import CategoryEntry, SourceId, Taxonomy

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = frozenset({".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
_METADATA_FILENAMES = ("metadata.csv", "metadata.jsonl", "metadata.parquet")
_LABEL_COLUMNS = ("label", "labels", "class", "category", "category_name")
_OBJECTS_COLUMN = "objects"
# The split-directory keywords Hugging Face ImageFolder split inference
# recognizes (datasets' data_files keyword lists), normalized to the three
# canonical split names. Kept in sync with the writer's table so every split
# directory the writer can emit is recognized on reload.
_SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "validation": "validation",
    "valid": "validation",
    "val": "validation",
    "dev": "validation",
    "test": "test",
    "testing": "test",
    "eval": "test",
    "evaluation": "test",
}
_SPLIT_ORDER = {"train": 0, "validation": 1, "test": 2}


@dataclass(frozen=True)
class _ImageRow:
    """One candidate image discovered from metadata or folder layout."""

    path: Path
    rel_path: str
    split: str | None
    label: str | None
    objects: Any | None = None
    width: int | None = None
    height: int | None = None
    metadata_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@register_loader
class HuggingFaceVisionImageClassificationLoader(Loader):
    """Load a Hugging Face ImageFolder classification repository."""

    task: ClassVar[Task] = Task.IC
    format = DatasetFormat.HUGGINGFACE_VISION
    variant: ClassVar[str] = "default"

    def load(
        self,
        root: str | Path,
        *,
        image_extensions: Collection[str] | str | None = None,
        **_: Any,
    ) -> ImageClassificationDataset:
        """Read a Hugging Face image classification dataset root.

        Parameters
        ----------
        root
            Local dataset repository root. Supported shapes are class folders
            (``cat/*.jpg``), split/class folders (``train/cat/*.jpg``), or a
            Hugging Face metadata file with ``file_name`` and ``label`` columns.
        image_extensions
            Optional allowlist of image extensions. Defaults to common Hugging
            Face-compatible suffixes (``.jpg``, ``.jpeg``, ``.png``, ``.bmp``,
            ``.gif``, ``.tif``, ``.tiff``, ``.webp``).
        """
        root = Path(root)
        if not root.is_dir():
            logger.warning("Hugging Face vision root is not a directory: %s", root)
            return ImageClassificationDataset(
                samples=(), dataset_metadata=DatasetMetadata(source_dataset="huggingface_vision")
            )

        extensions = _normalize_image_extensions(image_extensions)
        rows = _discover_rows(root, extensions, folder_fallback=True)
        rows = _dedupe_rows(_filter_existing_rows(rows))
        taxonomy, label_lookup = _ic_taxonomy(rows)

        samples: list[ImageClassificationSample] = []
        for row in sorted(rows, key=_row_sort_key):
            labels: tuple[ClassificationLabel, ...] = ()
            if row.label is not None:
                category_id, source_id = label_lookup[row.label]
                labels = (
                    ClassificationLabel(category_id=category_id, source_category_id=source_id, category_name=row.label),
                )
            samples.append(
                ImageClassificationSample(
                    image_id=row.rel_path,
                    path_or_uri=str(row.path),
                    file_name=row.rel_path,
                    width=row.width,
                    height=row.height,
                    split=row.split,
                    labels=labels,
                    metadata=_sample_metadata(row),
                )
            )
        if not samples:
            logger.warning("No loadable Hugging Face vision classification images found in %s", root)
        logger.info("Loaded %d Hugging Face vision IC image(s) from %s", len(samples), root)
        return ImageClassificationDataset(
            samples=tuple(samples),
            dataset_metadata=DatasetMetadata(
                taxonomy=taxonomy, source_dataset="huggingface_vision", splits=_splits_of(samples)
            ),
            dataset_id="huggingface_vision",
        )


@register_loader
class HuggingFaceVisionObjectDetectionLoader(Loader):
    """Load a Hugging Face ImageFolder object-detection repository."""

    task: ClassVar[Task] = Task.OD
    format = DatasetFormat.HUGGINGFACE_VISION
    variant: ClassVar[str] = "default"

    def load(
        self,
        root: str | Path,
        *,
        image_extensions: Collection[str] | str | None = None,
        **_: Any,
    ) -> ObjectDetectionDataset:
        """Read a Hugging Face object-detection dataset root.

        The OD convention is metadata-driven: a ``metadata.csv`` /
        ``metadata.jsonl`` file (at the root or inside first-level image
        directories such as ``train/`` or ``data/``) whose
        rows carry ``file_name`` plus an ``objects`` column of parallel
        ``bbox`` / ``categories`` lists. A row without ``objects`` loads as a
        background image with zero detections; a repository without any
        metadata file has no boxes to read and loads empty with a warning
        (use the IC variant or another format for label-free folders).
        """
        root = Path(root)
        if not root.is_dir():
            logger.warning("Hugging Face vision root is not a directory: %s", root)
            return ObjectDetectionDataset(
                samples=(), dataset_metadata=DatasetMetadata(source_dataset="huggingface_vision")
            )

        extensions = _normalize_image_extensions(image_extensions)
        rows = _discover_rows(root, extensions, folder_fallback=False)
        if not rows:
            logger.warning(
                "No Hugging Face metadata rows found under %s; the OD convention requires a metadata file "
                "with an 'objects' column",
                root,
            )
            return ObjectDetectionDataset(
                samples=(), dataset_metadata=DatasetMetadata(source_dataset="huggingface_vision")
            )
        rows = _dedupe_rows(_filter_existing_rows(rows))

        samples: list[ImageObjectDetectionSample] = []
        for row in sorted(rows, key=_row_sort_key):
            detections = _parse_objects(row)
            samples.append(
                ImageObjectDetectionSample(
                    image_id=row.rel_path,
                    path_or_uri=str(row.path),
                    file_name=row.rel_path,
                    width=row.width,
                    height=row.height,
                    split=row.split,
                    detections=detections,
                    metadata=_sample_metadata(row),
                )
            )
        taxonomy = _od_taxonomy(samples)
        if not samples:
            logger.warning("No loadable Hugging Face vision OD images found in %s", root)
        logger.info(
            "Loaded %d Hugging Face vision OD image(s), %d detection(s) from %s",
            len(samples),
            sum(len(s.detections) for s in samples),
            root,
        )
        return ObjectDetectionDataset(
            samples=tuple(samples),
            dataset_metadata=DatasetMetadata(
                taxonomy=taxonomy, source_dataset="huggingface_vision", splits=_splits_of(samples)
            ),
            dataset_id="huggingface_vision",
        )


# ---------------------------------------------------------------------------
# Row discovery (metadata files, then folder layout)
# ---------------------------------------------------------------------------


def _discover_rows(root: Path, extensions: frozenset[str], *, folder_fallback: bool) -> list[_ImageRow]:
    """Collect candidate image rows from metadata files or the folder layout."""
    metadata_files = _metadata_files(root)
    if metadata_files:
        rows = _rows_from_metadata(root, metadata_files, extensions)
        if rows or not folder_fallback:
            return rows
        if all(path.suffix.lower() == ".parquet" for path in metadata_files):
            logger.warning(
                "Hugging Face parquet metadata produced no loadable rows in %s; falling back to folder discovery",
                root,
            )
            return _rows_from_folder_layout(root, extensions)
        return rows
    if folder_fallback:
        return _rows_from_folder_layout(root, extensions)
    return []


def _rows_from_metadata(root: Path, metadata_files: Iterable[Path], extensions: frozenset[str]) -> list[_ImageRow]:
    rows: list[_ImageRow] = []
    for metadata_path in metadata_files:
        for row_number, raw in enumerate(_read_metadata_rows(metadata_path), start=1):
            row = _row_from_metadata(root, metadata_path, raw, row_number, extensions)
            if row is not None:
                rows.append(row)
    return rows


def _row_from_metadata(
    root: Path,
    metadata_path: Path,
    raw: Mapping[str, Any],
    row_number: int,
    extensions: frozenset[str],
) -> _ImageRow | None:
    posix = _parse_file_name(raw.get("file_name"), path=metadata_path, row_number=row_number)
    if posix is None:
        return None
    if posix.suffix.lower() not in extensions:
        logger.warning(
            "Skipping Hugging Face metadata row %s:%d with unsupported image extension: %s",
            metadata_path,
            row_number,
            posix,
        )
        return None
    base_dir = _metadata_base_dir(root, metadata_path, posix)
    image_path = base_dir.joinpath(*posix.parts)
    if not _is_within_root(image_path, root):
        logger.warning(
            "Skipping Hugging Face metadata row %s:%d whose resolved path escapes the dataset root: %s",
            metadata_path,
            row_number,
            posix,
        )
        return None
    split = _split_for_metadata_path(root, metadata_path) or _split_from_parts(posix.parts)
    # A metadata file disables folder-based label inference (HF ImageFolder
    # semantics): a row with no recognized label column stays label-free rather
    # than fabricating a class from its file_name's parent directory. The
    # folder-derived fallback lives only in the pure folder-discovery path
    # (`_rows_from_image_tree`).
    label = _label_from_metadata(raw, path=metadata_path, row_number=row_number)
    return _ImageRow(
        path=image_path,
        rel_path=_relative_posix(image_path, root),
        split=split,
        label=label,
        objects=raw.get(_OBJECTS_COLUMN),
        width=_coerce_dimension(raw.get("width")),
        height=_coerce_dimension(raw.get("height")),
        metadata_path=metadata_path,
        metadata={str(key): value for key, value in raw.items() if key is not None},
    )


def _rows_from_folder_layout(root: Path, extensions: frozenset[str]) -> list[_ImageRow]:
    """Discover images from class folders, optionally under split directories."""
    child_dirs = [child for child in _safe_children(root) if child.is_dir()]
    split_dirs = [child for child in child_dirs if _infer_split(child.name) is not None]
    if split_dirs and _is_split_layout(child_dirs, split_dirs):
        ignored = [child for child in child_dirs if child not in split_dirs]
        if ignored:
            logger.warning(
                "Treating top-level dirs %s as Hugging Face splits; ignoring non-split top-level dir(s) %s",
                sorted(child.name for child in split_dirs),
                sorted(child.name for child in ignored),
            )
        rows: list[_ImageRow] = []
        for split_dir in sorted(split_dirs, key=lambda path: _split_sort_key(_infer_split(path.name), path.name)):
            rows.extend(
                _rows_from_image_tree(root, split_dir, split=_infer_split(split_dir.name), extensions=extensions)
            )
        return rows
    return _rows_from_image_tree(root, root, split=None, extensions=extensions)


def _is_split_layout(child_dirs: list[Path], split_dirs: list[Path]) -> bool:
    """Whether top-level split-named dirs are unambiguously splits, not classes.

    A dir literally named ``train``/``test``/``val`` is only a split when the
    layout is unambiguous: either every top-level dir is a split name, or the
    split-named dirs themselves contain class subdirectories (so they cannot be
    leaf class folders). Otherwise they are treated as ordinary class folders.
    """
    non_split = [child for child in child_dirs if child not in split_dirs]
    if not non_split:
        return True
    return all(_has_subdirectory(split_dir) for split_dir in split_dirs)


def _has_subdirectory(path: Path) -> bool:
    return any(child.is_dir() for child in _safe_children(path))


def _rows_from_image_tree(root: Path, base: Path, *, split: str | None, extensions: frozenset[str]) -> list[_ImageRow]:
    rows: list[_ImageRow] = []
    for image_path in _iter_image_files(base, extensions):
        if not _is_within_root(image_path, root):
            logger.warning("Skipping Hugging Face image whose path escapes the dataset root: %s", image_path)
            continue
        try:
            rel_to_base = image_path.relative_to(base)
        except ValueError:
            rel_to_base = image_path
        rows.append(
            _ImageRow(
                path=image_path,
                rel_path=_relative_posix(image_path, root),
                split=split,
                label=_label_from_parts(rel_to_base.parts),
            )
        )
    return rows


def _filter_existing_rows(rows: Iterable[_ImageRow]) -> list[_ImageRow]:
    valid: list[_ImageRow] = []
    for row in rows:
        if not row.path.is_file():
            logger.warning("Skipping Hugging Face vision entry with missing image file: %s", row.path)
            continue
        valid.append(row)
    return valid


def _dedupe_rows(rows: Iterable[_ImageRow]) -> list[_ImageRow]:
    seen: set[Path] = set()
    deduped: list[_ImageRow] = []
    for row in rows:
        key = row.path.resolve(strict=False)
        if key in seen:
            logger.warning("Skipping duplicate Hugging Face vision entry: %s", row.path)
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


# ---------------------------------------------------------------------------
# IC taxonomy
# ---------------------------------------------------------------------------


def _ic_taxonomy(rows: Iterable[_ImageRow]) -> tuple[Taxonomy | None, dict[str, tuple[int, int]]]:
    """Class ids from discovered label names.

    When every discovered label is an integer-valued string (e.g. Hugging Face
    ``ClassLabel`` indices stringified by :func:`_coerce_label`), the integer is
    preserved as both ``category_id`` and ``source_category_id`` and labels are
    ordered NUMERICALLY, so integer label ``10`` keeps id ``10`` instead of being
    re-indexed to a lexical position. Genuinely non-numeric string labels fall
    back to dense positional ids ordered case-insensitively.
    """
    unique = {row.label for row in rows if row.label is not None}
    if not unique:
        return None, {}
    int_values = _integer_label_values(unique)
    if int_values is not None:
        ordered = sorted(unique, key=lambda label: int_values[label])
        entries = tuple(CategoryEntry(source_id=int_values[label], name=label) for label in ordered)
        int_ids = [int_values[label] for label in ordered]
        density = "dense" if int_ids == list(range(len(ordered))) else "sparse"
        taxonomy = Taxonomy(
            entries=entries,
            source_dataset="huggingface_vision",
            id_density=density,
            ordered_names=tuple(ordered),
        )
        return taxonomy, {label: (int_values[label], int_values[label]) for label in ordered}
    ordered = sorted(unique, key=lambda value: value.casefold())
    entries = tuple(CategoryEntry(source_id=index, name=label) for index, label in enumerate(ordered))
    taxonomy = Taxonomy(
        entries=entries,
        source_dataset="huggingface_vision",
        id_density="dense",
        ordered_names=tuple(ordered),
    )
    return taxonomy, {label: (index, index) for index, label in enumerate(ordered)}


def _integer_label_values(labels: Iterable[str]) -> dict[str, int] | None:
    """Map each label to its integer value, or ``None`` if any label is non-integer.

    Only canonical integer strings count (``str(int(label)) == label``), so
    ``"01"``, ``"+1"``, or ``"1_0"`` stay string labels rather than being coerced.
    """
    values: dict[str, int] = {}
    for label in labels:
        try:
            number = int(label)
        except (TypeError, ValueError):
            return None
        if str(number) != label:
            return None
        values[label] = number
    return values


# ---------------------------------------------------------------------------
# OD objects parsing
# ---------------------------------------------------------------------------


def _parse_objects(row: _ImageRow) -> tuple[ObjectDetectionAnnotation, ...]:  # noqa: C901 - per-detection validation is intentionally explicit
    """Parse one row's ``objects`` value into detections, skipping bad boxes with warnings."""
    raw = row.objects
    if raw is None or _is_missing(raw):
        return ()
    if isinstance(raw, str):  # CSV carries objects as a JSON-encoded string
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("Skipping malformed Hugging Face objects JSON for %s: %s", row.rel_path, exc)
            return ()
    if not isinstance(raw, Mapping):
        logger.warning("Skipping non-object Hugging Face objects value for %s", row.rel_path)
        return ()

    bboxes = _as_list(raw.get("bbox"))
    if bboxes is None:
        bboxes = _as_list(raw.get("bboxes"))
    if bboxes is None:
        logger.warning("Skipping Hugging Face objects for %s without a bbox list", row.rel_path)
        return ()
    categories = _as_list(raw.get("categories"))
    if categories is None:
        categories = _as_list(raw.get("category"))
    ids = _as_list(raw.get("id"))
    areas = _as_list(raw.get("area"))
    if categories is not None and len(categories) != len(bboxes):
        logger.warning(
            "Hugging Face objects for %s has %d bbox(es) but %d categorie(s); pairing up to the shorter list",
            row.rel_path,
            len(bboxes),
            len(categories),
        )

    detections: list[ObjectDetectionAnnotation] = []
    for index, raw_bbox in enumerate(bboxes):
        bbox = _parse_bbox(raw_bbox, rel_path=row.rel_path, index=index)
        if bbox is None:
            continue
        category = categories[index] if categories is not None and index < len(categories) else None
        category_id, category_name, source_category_id = _parse_category(category)
        detections.append(
            ObjectDetectionAnnotation(
                bbox=bbox,
                category_id=category_id,
                category_name=category_name,
                source_category_id=source_category_id,
                source_annotation_id=_scalar_or_none(ids, index),
                area=_area_or_none(areas, index),
            )
        )
    return tuple(detections)


def _parse_bbox(value: Any, *, rel_path: str, index: int) -> BBox | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        logger.warning("Skipping Hugging Face bbox %d for %s: expected 4 values", index, rel_path)
        return None
    coords: list[float] = []
    for item in value:
        number = _finite_number(item)
        if number is None:
            logger.warning("Skipping Hugging Face bbox %d for %s with non-numeric value", index, rel_path)
            return None
        coords.append(number)
    bbox: BBox = (coords[0], coords[1], coords[2], coords[3])
    if not has_positive_area(bbox):
        logger.warning("Skipping Hugging Face bbox %d for %s with non-positive area", index, rel_path)
        return None
    return bbox


def _parse_category(value: Any) -> tuple[int | None, str | None, SourceId]:
    """Split one category value into ``(category_id, category_name, source_category_id)``."""
    if isinstance(value, bool) or _is_missing(value):
        return None, None, None
    if isinstance(value, int):
        return value, None, value
    if isinstance(value, float) and value.is_integer():
        return int(value), None, int(value)
    if isinstance(value, str):
        text = value.strip()
        return None, text or None, text or None
    return None, None, None


def _od_taxonomy(samples: Iterable[ImageObjectDetectionSample]) -> Taxonomy | None:
    """Source-preserving taxonomy from observed detection categories."""
    discovered: dict[SourceId, str] = {}
    for sample in samples:
        for detection in sample.detections:
            source_id = (
                detection.source_category_id if detection.source_category_id is not None else detection.category_id
            )
            if source_id is None or source_id in discovered:
                continue
            discovered[source_id] = detection.category_name or str(source_id)
    if not discovered:
        return None
    # Bucket integer ids and sort them NUMERICALLY (0,1,2,...,10,11), not
    # lexicographically (0,1,10,11,2,...): the latter both misorders entries and
    # breaks the contiguous-range density check below.
    int_items: list[tuple[int, str]] = []
    other_items: list[tuple[SourceId, str]] = []
    for source_id, name in discovered.items():
        if isinstance(source_id, int) and not isinstance(source_id, bool):
            int_items.append((source_id, name))
        else:
            other_items.append((source_id, name))
    int_items.sort(key=lambda item: item[0])
    other_items.sort(key=lambda item: str(item[0]))
    ordered: list[tuple[SourceId, str]] = [*int_items, *other_items]
    entries = tuple(CategoryEntry(source_id=source_id, name=name) for source_id, name in ordered)
    int_ids = [source_id for source_id, _ in int_items]
    density = "dense" if int_ids == list(range(len(entries))) else "sparse"
    return Taxonomy(entries=entries, source_dataset="huggingface_vision", id_density=density)


def _as_list(value: Any) -> list[Any] | None:
    if isinstance(value, (list, tuple)):
        return list(value)
    return None


def _scalar_or_none(values: list[Any] | None, index: int) -> int | str | None:
    if values is None or index >= len(values):
        return None
    value = values[index]
    if isinstance(value, bool) or _is_missing(value):
        return None
    if isinstance(value, (int, str)):
        return value
    return None


def _area_or_none(values: list[Any] | None, index: int) -> float | None:
    if values is None or index >= len(values):
        return None
    return _finite_number(values[index])


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _coerce_dimension(value: Any) -> int | None:
    """Coerce a metadata width/height value (int, float, or CSV string) to a positive int."""
    if isinstance(value, bool) or _is_missing(value):
        return None
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            return None
    if isinstance(value, float):
        if not (math.isfinite(value) and value.is_integer()):
            return None
        value = int(value)
    if isinstance(value, int) and value > 0:
        return value
    return None


# ---------------------------------------------------------------------------
# Metadata file reading (mirrors huggingface_video_classification)
# ---------------------------------------------------------------------------


def _metadata_files(root: Path) -> list[Path]:
    """Return root metadata files, or first-level per-directory files when no root file exists.

    The per-directory scan covers every first-level directory, not just
    split-named ones: the matching writer emits ``data/metadata.jsonl`` for
    unsplit samples (Hugging Face's ImageFolder associates metadata files
    per directory tree, so the writer never uses a root-level file when
    image directories exist).
    """
    root_files = _metadata_files_in_dir(root)
    if root_files:
        return root_files
    files: list[Path] = []
    for child in _safe_children(root):
        if child.is_dir():
            files.extend(_metadata_files_in_dir(child))
    return files


def _metadata_files_in_dir(path: Path) -> list[Path]:
    children = {child.name.lower(): child for child in _safe_children(path) if child.is_file()}
    matches = [children[name] for name in _METADATA_FILENAMES if name in children]
    if len(matches) > 1:
        logger.warning("Multiple Hugging Face metadata files found in %s; loading all", path)
    return matches


def _read_metadata_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _read_csv_rows(path)
    if suffix == ".jsonl":
        return _read_jsonl_rows(path)
    if suffix == ".parquet":
        return _read_parquet_rows(path)
    logger.warning("Unsupported Hugging Face metadata file type: %s", path)
    return []


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames or "file_name" not in reader.fieldnames:
                logger.warning("Hugging Face metadata CSV is missing required file_name column: %s", path)
                return []
            rows: list[dict[str, Any]] = []
            for line_no, row in enumerate(reader, start=2):
                if None in row:
                    logger.warning("Ignoring extra CSV columns in Hugging Face metadata row %s:%d", path, line_no)
                    row.pop(None, None)
                rows.append(dict(row))
            return rows
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        logger.warning("Could not read Hugging Face metadata CSV %s: %s", path, exc)
        return []


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed Hugging Face metadata JSONL row %s:%d: %s", path, line_no, exc)
                    continue
                if not isinstance(row, dict):
                    logger.warning("Skipping non-object Hugging Face metadata JSONL row %s:%d", path, line_no)
                    continue
                rows.append(row)
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Could not read Hugging Face metadata JSONL %s: %s", path, exc)
    return rows


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    try:  # pragma: no cover - optional parquet dependency not installed in the core test matrix.
        import pyarrow.parquet as pq  # type: ignore[import-untyped]
    except ImportError:  # pragma: no cover - optional pandas fallback is not installed in the core test matrix.
        try:
            import pandas as pd  # type: ignore[import-untyped]
        except ImportError:
            logger.warning(
                "Cannot read Hugging Face metadata parquet %s; install pyarrow or pandas to enable parquet metadata",
                path,
            )
            return []
        try:
            rows = pd.read_parquet(path).to_dict(orient="records")
        except Exception as exc:
            logger.warning("Could not read Hugging Face metadata parquet %s: %s", path, exc)
            return []
    else:  # pragma: no cover - pyarrow success path needs the optional parquet dependency.
        try:
            rows = pq.read_table(path).to_pylist()
        except Exception as exc:
            logger.warning("Could not read Hugging Face metadata parquet %s: %s", path, exc)
            return []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


# ---------------------------------------------------------------------------
# Shared path/label helpers (mirrors huggingface_video_classification)
# ---------------------------------------------------------------------------


def _parse_file_name(value: object, *, path: Path, row_number: int) -> PurePosixPath | None:
    if not isinstance(value, str) or not value.strip():
        logger.warning("Skipping Hugging Face metadata row %s:%d with missing file_name", path, row_number)
        return None
    if "\\" in value:
        logger.warning("Skipping Hugging Face metadata row %s:%d with unsafe file_name: %r", path, row_number, value)
        return None
    posix = PurePosixPath(value.strip())
    if posix.is_absolute() or not posix.parts or any(part in {"", ".."} or ":" in part for part in posix.parts):
        logger.warning("Skipping Hugging Face metadata row %s:%d with unsafe file_name: %r", path, row_number, value)
        return None
    return posix


def _metadata_base_dir(root: Path, metadata_path: Path, rel_path: PurePosixPath) -> Path:
    metadata_split = _split_for_metadata_path(root, metadata_path)
    if metadata_split is not None and rel_path.parts and _infer_split(rel_path.parts[0]) == metadata_split:
        return root
    return metadata_path.parent


def _split_for_metadata_path(root: Path, metadata_path: Path) -> str | None:
    try:
        parts = metadata_path.parent.relative_to(root).parts
    except ValueError:
        return None
    return _infer_split(parts[0]) if parts else None


def _split_from_parts(parts: tuple[str, ...]) -> str | None:
    return _infer_split(parts[0]) if parts else None


def _label_from_metadata(row: Mapping[str, Any], *, path: Path, row_number: int) -> str | None:
    for column in _LABEL_COLUMNS:
        if column not in row:
            continue
        value = row[column]
        label = _coerce_label(value)
        if label is not None:
            return label
        if not _is_missing(value):
            logger.warning(
                "Ignoring non-scalar Hugging Face label value in %s:%d column %r",
                path,
                row_number,
                column,
            )
    return None


def _label_from_parts(parts: tuple[str, ...]) -> str | None:
    """Class label from an image path relative to its discovery base.

    ``parts`` is relative to the tree base (the split dir when splits are
    present, else the root), so the leading component is the class folder even
    when it is literally named ``train``/``test``/``val``; a bare filename with
    no parent folder has no label.
    """
    return parts[0] if len(parts) >= 2 else None


def _coerce_label(value: object) -> str | None:
    if _is_missing(value) or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else str(value)
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _is_missing(value: object) -> bool:
    return (
        value is None
        or (isinstance(value, str) and not value.strip())
        or (isinstance(value, float) and math.isnan(value))
    )


def _normalize_image_extensions(value: Collection[str] | str | None) -> frozenset[str]:
    if value is None:
        return _IMAGE_EXTENSIONS
    raw_extensions = [value] if isinstance(value, str) else list(value)
    normalized: set[str] = set()
    for raw in raw_extensions:
        ext = str(raw).strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        if ext == "." or "/" in ext or "\\" in ext or ".." in ext:
            raise ValueError(f"image_extensions entries must be safe extensions, got {raw!r}")
        normalized.add(ext)
    if not normalized:
        raise ValueError("image_extensions must include at least one extension")
    return frozenset(normalized)


def _iter_image_files(base: Path, extensions: frozenset[str]) -> list[Path]:
    try:
        return sorted(path for path in base.rglob("*") if path.is_file() and path.suffix.lower() in extensions)
    except OSError as exc:
        logger.warning("Could not list Hugging Face image tree %s: %s", base, exc)
        return []


def _safe_children(path: Path) -> list[Path]:
    try:
        return sorted(path.iterdir(), key=lambda item: item.name)
    except OSError as exc:
        logger.warning("Could not list Hugging Face vision directory %s: %s", path, exc)
        return []


def _infer_split(name: str) -> str | None:
    return _SPLIT_ALIASES.get(name.lower().replace("_", "-"))


def _row_sort_key(row: _ImageRow) -> tuple[int, str, str, str]:
    return (*_split_sort_key(row.split, ""), row.label or "", row.rel_path)


def _split_sort_key(split: str | None, fallback: str) -> tuple[int, str]:
    return (_SPLIT_ORDER.get(split or "", len(_SPLIT_ORDER)), split or fallback)


def _relative_posix(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _is_within_root(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except (OSError, ValueError):
        return False
    return True


def _sample_metadata(row: _ImageRow) -> dict[str, Any]:
    metadata: dict[str, Any] = {"source_format": "huggingface_vision", "source_file_name": row.rel_path}
    if row.metadata_path is not None:
        metadata["metadata_file"] = str(row.metadata_path)
    passthrough = {
        key: value
        for key, value in row.metadata.items()
        if key not in {"file_name", _OBJECTS_COLUMN, "width", "height", *_LABEL_COLUMNS}
    }
    if passthrough:
        metadata["huggingface_metadata"] = passthrough
    return metadata


def _splits_of(samples: Iterable[Any]) -> tuple[str, ...]:
    seen = {sample.split for sample in samples if sample.split is not None}
    return tuple(sorted(seen, key=lambda split: _SPLIT_ORDER.get(split, len(_SPLIT_ORDER))))
