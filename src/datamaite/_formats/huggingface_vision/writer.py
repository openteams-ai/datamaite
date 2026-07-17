"""Hugging Face Vision (still-image) dataset writers (IR-3.2-S-6).

The module mirrors :mod:`datamaite._formats.huggingface_vision.loader`: one
writer is registered for image classification (``Task.IC``) and one for object
detection (``Task.OD``) under the same ``DatasetFormat.HUGGINGFACE_VISION``
family, so the detection/classification selection required by the standard is
the writer registry's task axis (``write(dataset, output_format=
"huggingface_vision")`` dispatches on ``dataset.task``; ``get_writer(...,
task=...)`` selects explicitly).

* IC emits the ImageFolder classification convention the matching loader
  reads back: ``<dest>/<split>/<class>/<image>``.
* OD emits images under ``<dest>/<split>/`` plus a ``metadata.jsonl`` (or
  ``metadata.csv``) *inside each split directory* whose rows carry
  ``file_name``, ``width``/``height`` when known, and the Hugging Face
  ``objects`` column of parallel ``bbox`` / ``categories`` lists (with
  ``id`` / ``area`` when the source detections carry them). Per-directory
  placement is deliberate: once split directories exist,
  ``datasets.load_dataset("imagefolder", ...)`` only associates metadata
  files inside each split's directory tree — a root-level metadata file is
  silently ignored (verified against ``datasets`` 5.x; see
  ``tests/test_huggingface_vision_datasets_compat.py``).

Scope: the output is the *local ImageFolder-compatible layout* that
``datasets.load_dataset("imagefolder", data_dir=...)`` reads — not Hub
repositories, Arrow datasets, dataset scripts, or full feature schemas.
Two deliberate consequences:

* Split directories are limited to the ImageFolder-recognized
  ``train``/``validation``/``test`` (plus aliases such as ``val``, ``dev``,
  ``eval``); anything else would reload as a class folder (IC) or lose its
  split (OD), so unknown split *options* raise and unknown *sample* splits
  fall back with a warning (see ``_normalize_split``).
* ``metadata_format="csv"`` (``objects`` JSON-encoded in one CSV column) is a
  datamaite extension: the matching datamaite loader parses it back, but
  Hugging Face only documents the OD ``objects`` convention for
  ``metadata.jsonl``, so keep the default ``jsonl`` for HF-standard output.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import re
import shutil
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar

from datamaite._types import DatasetFormat, Task
from datamaite.image_classification import ImageClassificationDataset
from datamaite.object_detection import ObjectDetectionDataset
from datamaite.records import (
    ClassificationLabel,
    ImageClassificationSample,
    ImageObjectDetectionSample,
    ObjectDetectionAnnotation,
)
from datamaite.taxonomy import Taxonomy
from datamaite.writers import Writer, WriterCapabilities, register_writer

logger = logging.getLogger(__name__)

_METADATA_FORMATS = frozenset({"csv", "jsonl"})
# The split-directory keywords Hugging Face ImageFolder split inference
# recognizes, normalized to the three canonical split names. Any other split
# directory would NOT round-trip: the IC loader would read it as a class
# folder and the OD loader would drop it, so the writers never emit one.
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
_CANONICAL_SPLITS = frozenset({"train", "validation", "test"})
_RESERVED_COLUMNS = frozenset({"file_name", "objects", "width", "height"})


@register_writer
class HuggingFaceVisionImageClassificationWriter(Writer[ImageClassificationDataset]):
    """Write an IC dataset as a Hugging Face ImageFolder classification repo."""

    task: ClassVar[Task] = Task.IC
    format = DatasetFormat.HUGGINGFACE_VISION
    variant: ClassVar[str] = "default"
    consumes: ClassVar[type] = ImageClassificationDataset
    capabilities: ClassVar[WriterCapabilities] = WriterCapabilities(required_fields=frozenset({"image", "labels"}))

    def validate_options(self, **options: Any) -> None:
        """Validate options that can raise, before write()'s destination policy runs (#55 Fix A1)."""
        if "default_split" in options:
            _normalize_split(str(options["default_split"]), field="default_split")

    def write(
        self,
        dataset: ImageClassificationDataset,
        dest: str | Path,
        *,
        default_split: str = "train",
        **_: Any,
    ) -> list[Path]:
        """Serialise an IC dataset as ``<split>/<class>/<image>`` directories.

        Samples with neither ``image_bytes`` nor ``path_or_uri``, with no
        resolvable class label, or carrying a crop region (this writer copies
        whole files and cannot crop) are skipped with warnings. A folder
        classification format represents one class per image; when a sample
        has multiple labels, the first is used.

        ``default_split`` must be a Hugging Face ImageFolder split
        (``train``/``validation``/``test`` or an alias such as ``val``): any
        other directory name would reload as a class folder, not a split.
        Samples whose own ``split`` is not a recognized split name fall back
        to ``default_split`` with a warning for the same reason.
        """
        dest_path = Path(dest)
        dest_path.mkdir(parents=True, exist_ok=True)
        default_split = _normalize_split(str(default_split), field="default_split")
        taxonomy = dataset.dataset_metadata.taxonomy
        written: list[Path] = []
        used_targets: set[Path] = set()

        for sample in dataset.samples:
            target = _write_ic_sample(
                sample, dest_path, taxonomy=taxonomy, default_split=default_split, used_targets=used_targets
            )
            if target is None:
                continue
            used_targets.add(target)
            written.append(target)
        if not written:
            logger.warning("No Hugging Face vision IC images were written to %s", dest_path)
        return written


@register_writer
class HuggingFaceVisionObjectDetectionWriter(Writer[ObjectDetectionDataset]):
    """Write an OD dataset as a Hugging Face ImageFolder + ``objects`` metadata repo."""

    task: ClassVar[Task] = Task.OD
    format = DatasetFormat.HUGGINGFACE_VISION
    variant: ClassVar[str] = "default"
    consumes: ClassVar[type] = ObjectDetectionDataset
    capabilities: ClassVar[WriterCapabilities] = WriterCapabilities(
        required_fields=frozenset({"image"}),
        lossy_without={
            "score": "the Hugging Face objects convention stores ground-truth boxes, not confidences",
            "segmentation": "the objects column stores bbox/categories lists only",
            "iscrowd": "the objects column stores bbox/categories lists only",
            "source_category_id": "the objects categories list holds one value per box; when a detection "
            "has a category name the name is written (so reloads keep it) and the numeric id is rebuilt "
            "from the reload-side taxonomy",
        },
    )

    def validate_options(self, **options: Any) -> None:
        """Validate options that can raise, before write()'s destination policy runs (#55 Fix A1)."""
        if "split" in options:
            _normalize_optional_split(options["split"])
        if "metadata_format" in options:
            _validate_metadata_format(options["metadata_format"])

    def write(
        self,
        dataset: ObjectDetectionDataset,
        dest: str | Path,
        *,
        split: str | None = None,
        preserve_splits: bool = True,
        metadata_format: str = "jsonl",
        **_: Any,
    ) -> list[Path]:
        """Serialise an OD dataset as images plus a root metadata file.

        Parameters
        ----------
        split
            Optional fallback split for samples without split metadata. Use
            ``None`` (default) to write unsplit images under ``data/``. Must
            be a Hugging Face ImageFolder split (``train``/``validation``/
            ``test``); aliases such as ``"val"`` normalise to
            ``"validation"``. Other names raise: the loader could not read
            the split back.
        preserve_splits
            When True (default), a sample whose ``split`` is a known Hugging
            Face split is written under that split directory; otherwise the
            fallback ``split`` is used.
        metadata_format
            ``"jsonl"`` (default) writes ``metadata.jsonl``, the Hugging
            Face-documented carrier for the nested ``objects`` column;
            ``"csv"`` writes ``metadata.csv`` with ``objects`` JSON-encoded —
            a datamaite extension the matching loader reads back, but not
            documented HF ImageFolder behaviour (``datasets`` loads the
            column as a plain string).

        One metadata file is written inside each emitted image directory
        (``train/metadata.jsonl``, ``data/metadata.jsonl``, ...) with
        ``file_name`` relative to that directory: ``datasets``' ImageFolder
        only associates metadata within a split's directory tree, so a
        root-level metadata file would silently lose the ``objects`` column
        on the Hugging Face side once split directories exist.
        """
        fallback_split = _normalize_optional_split(split)
        metadata_format = _validate_metadata_format(metadata_format)
        dest_path = Path(dest)
        dest_path.mkdir(parents=True, exist_ok=True)

        rows_by_dir: dict[str, list[dict[str, Any]]] = {}
        written: list[Path] = []
        used_paths: set[str] = set()

        for sample in dataset.samples:
            record = _od_record_for_sample(
                sample,
                dest=dest_path,
                fallback_split=fallback_split,
                preserve_splits=preserve_splits,
                used_paths=used_paths,
            )
            if record is None:
                continue
            source, rel_path = record
            out_path = dest_path.joinpath(*PurePosixPath(rel_path).parts)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            _copy_image(source, out_path, image_bytes=sample.image_bytes)
            written.append(out_path)
            parts = PurePosixPath(rel_path).parts
            dir_name, dir_relative = parts[0], PurePosixPath(*parts[1:]).as_posix()
            rows_by_dir.setdefault(dir_name, []).append(
                _od_metadata_row(sample, file_name=dir_relative, metadata_format=metadata_format)
            )

        if rows_by_dir:
            for dir_name in sorted(rows_by_dir):
                metadata_path = dest_path / dir_name / f"metadata.{metadata_format}"
                _write_metadata(metadata_path, rows_by_dir[dir_name], metadata_format=metadata_format)
                written.append(metadata_path)
        else:
            metadata_path = dest_path / f"metadata.{metadata_format}"
            _write_metadata(metadata_path, [], metadata_format=metadata_format)
            written.append(metadata_path)
            logger.warning("No Hugging Face vision OD images were written to %s", dest_path)
        return written


# ---------------------------------------------------------------------------
# Classification writer helpers
# ---------------------------------------------------------------------------


def _write_ic_sample(
    sample: ImageClassificationSample,
    dest_path: Path,
    *,
    taxonomy: Taxonomy | None,
    default_split: str,
    used_targets: set[Path],
) -> Path | None:
    if getattr(sample, "region", None) is not None:
        # Copying the whole source image would emit a full image under a class
        # label that only describes a crop of it. Skip loudly (same guard as the
        # YOLO IC writer).
        logger.warning(
            "Skipping Hugging Face vision IC sample %r: it carries a crop region this image-copying "
            "writer cannot represent",
            sample.image_id,
        )
        return None
    label = _single_label(sample)
    if label is None:
        logger.warning("Skipping Hugging Face vision IC sample %r with no labels", sample.image_id)
        return None
    class_name = _ic_class_name(label, taxonomy)
    if class_name is None:
        logger.warning("Skipping Hugging Face vision IC sample %r with unresolved label %r", sample.image_id, label)
        return None
    split = _ic_split(sample, default_split)
    try:
        safe_class = _safe_path_part(class_name, field="class name")
        file_name = _safe_file_name(sample)
    except ValueError as exc:
        logger.warning("Skipping Hugging Face vision IC sample %r: %s", sample.image_id, exc)
        return None

    source: Path | None = None
    if sample.image_bytes is None:
        if sample.path_or_uri is None:
            logger.warning("Skipping Hugging Face vision IC sample %r with no image source", sample.image_id)
            return None
        source = Path(sample.path_or_uri)
        if not source.is_file():
            logger.warning(
                "Skipping Hugging Face vision IC sample %r with missing image file: %s", sample.image_id, source
            )
            return None

    class_dir = dest_path / split / safe_class
    class_dir.mkdir(parents=True, exist_ok=True)
    target = _unique_target(class_dir / file_name, used_targets)
    if sample.image_bytes is not None:
        target.write_bytes(sample.image_bytes)
    elif source is not None:
        shutil.copy2(source, target)
    return target


def _ic_split(sample: ImageClassificationSample, default_split: str) -> str:
    """Resolve a sample's split directory, falling back on unsafe values with a warning."""
    if not sample.split:
        return default_split
    try:
        return _normalize_split(str(sample.split), field="sample.split")
    except ValueError:
        logger.warning(
            "Sample %r has split %r, which is not a Hugging Face ImageFolder split "
            "(train/validation/test); writing it to %r",
            sample.image_id,
            sample.split,
            default_split,
        )
        return default_split


def _single_label(sample: ImageClassificationSample) -> ClassificationLabel | None:
    if not sample.labels:
        return None
    if len(sample.labels) > 1:
        logger.warning("Hugging Face ImageFolder emits one label per image; using first label for %r", sample.image_id)
    return next(iter(sample.labels))


def _ic_class_name(label: ClassificationLabel, taxonomy: Taxonomy | None) -> str | None:
    source_id = label.source_category_id if label.source_category_id is not None else label.category_id
    if taxonomy is not None:
        entry = taxonomy.by_source_id(source_id)
        if entry is not None:
            return entry.name
        if (
            taxonomy.id_density == "dense"
            and isinstance(source_id, int)
            and not isinstance(source_id, bool)
            and 0 <= source_id < len(taxonomy.entries)
        ):
            return taxonomy.entries[source_id].name
        if label.category_name is not None:
            entry = taxonomy.by_name(label.category_name)
            if entry is not None:
                return entry.name
    return label.category_name


# ---------------------------------------------------------------------------
# Object-detection writer helpers
# ---------------------------------------------------------------------------


def _od_record_for_sample(
    sample: ImageObjectDetectionSample,
    *,
    dest: Path,
    fallback_split: str | None,
    preserve_splits: bool,
    used_paths: set[str],
) -> tuple[Path | None, str] | None:
    source: Path | None = None
    if sample.image_bytes is None:
        if sample.path_or_uri is None:
            logger.warning("Skipping Hugging Face vision OD sample %r with no image source", sample.image_id)
            return None
        source = Path(sample.path_or_uri)
        if not source.is_file():
            logger.warning(
                "Skipping Hugging Face vision OD sample %r with missing image file: %s", sample.image_id, source
            )
            return None

    split = _split_for_sample(sample, fallback=fallback_split, preserve_splits=preserve_splits)
    try:
        file_name = _safe_file_name(sample)
    except ValueError as exc:
        logger.warning("Skipping Hugging Face vision OD sample %r: %s", sample.image_id, exc)
        return None
    stem, suffix = Path(file_name).stem, Path(file_name).suffix
    rel_path = _unique_rel_path(split or "data", stem, suffix, used_paths)
    if source is not None and (dest / rel_path).resolve(strict=False) == source.resolve(strict=False):
        rel_path = _unique_rel_path(split or "data", f"{stem}-copy", suffix, used_paths)
    return source, rel_path


def _od_metadata_row(
    sample: ImageObjectDetectionSample,
    *,
    file_name: str,
    metadata_format: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {"file_name": file_name}
    if sample.width is not None:
        row["width"] = sample.width
    if sample.height is not None:
        row["height"] = sample.height
    objects = _objects_value(sample)
    row["objects"] = json.dumps(objects, sort_keys=True) if metadata_format == "csv" else objects
    return row


def _objects_value(sample: ImageObjectDetectionSample) -> dict[str, Any]:
    """Build the parallel-list ``objects`` value, dropping unrepresentable fields loudly."""
    bboxes: list[list[float]] = []
    categories: list[int | str | None] = []
    ids: list[int | str | None] = []
    areas: list[float | None] = []
    dropped_scores = 0
    dropped_lossy = 0
    for detection in sample.detections:
        bboxes.append([float(value) for value in detection.bbox])
        categories.append(_category_value(detection))
        ids.append(detection.source_annotation_id if isinstance(detection.source_annotation_id, (int, str)) else None)
        areas.append(float(detection.area) if detection.area is not None else None)
        if detection.score is not None:
            dropped_scores += 1
        if detection.segmentation is not None or detection.iscrowd:
            dropped_lossy += 1
    if dropped_scores:
        logger.warning(
            "Dropped score from %d Hugging Face vision OD detection(s) on sample %r: the objects "
            "convention stores ground-truth boxes, not confidences",
            dropped_scores,
            sample.image_id,
        )
    if dropped_lossy:
        logger.warning(
            "Dropped unrepresentable OD fields (segmentation/iscrowd) from %d detection(s) on sample %r",
            dropped_lossy,
            sample.image_id,
        )
    objects: dict[str, Any] = {"bbox": bboxes, "categories": categories}
    if any(value is not None for value in ids):
        objects["id"] = ids
    if any(value is not None for value in areas):
        objects["area"] = areas
    return objects


def _category_value(detection: ObjectDetectionAnnotation) -> int | str | None:
    """One wire value per box: the category name when known, else the source id.

    The ``objects`` convention has a single ``categories`` list and no side
    channel for a features/ClassLabel mapping (that lives in ``datasets``
    metadata this local layout does not carry), so a detection with both
    ``category_id=0`` and ``category_name="person"`` can keep only one.
    Writing the name keeps reloads meaningful ("person", not "0"); the
    numeric id is rebuilt by the reload-side taxonomy (declared in
    ``lossy_without``).
    """
    if detection.category_name is not None:
        return detection.category_name
    source_id = detection.source_category_id if detection.source_category_id is not None else detection.category_id
    if isinstance(source_id, bool):
        return None
    if isinstance(source_id, (int, str)):
        return source_id
    return None


# ---------------------------------------------------------------------------
# Shared writer helpers (mirrors huggingface_video_classification.writer)
# ---------------------------------------------------------------------------


def _write_metadata(path: Path, rows: list[dict[str, Any]], *, metadata_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if metadata_format == "jsonl":
        text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        path.write_text(text, encoding="utf-8")
        return
    fieldnames = ["file_name", "width", "height", "objects"]
    extras = sorted({key for row in rows for key in row} - set(fieldnames) - _RESERVED_COLUMNS)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[*fieldnames, *extras])
        writer.writeheader()
        writer.writerows(rows)


def _copy_image(source: Path | None, dest: Path, *, image_bytes: bytes | None) -> None:
    if image_bytes is not None:
        dest.write_bytes(image_bytes)
        return
    if source is None:  # unreachable by construction; _od_record_for_sample verified one of the two
        raise ValueError(f"no image source for {dest}")
    try:
        same_file = source.resolve(strict=False) == dest.resolve(strict=False)
    except OSError:
        same_file = False
    if not same_file:
        shutil.copy2(source, dest)


def _split_for_sample(sample: ImageObjectDetectionSample, *, fallback: str | None, preserve_splits: bool) -> str | None:
    if not preserve_splits:
        return fallback
    raw = sample.split
    if raw is None:
        return fallback
    try:
        return _normalize_split(str(raw), field="sample.split")
    except ValueError:
        logger.warning(
            "Sample %r has split %r, which is not a Hugging Face ImageFolder split "
            "(train/validation/test); writing it to fallback split %r",
            sample.image_id,
            raw,
            fallback,
        )
        return fallback


def _normalize_optional_split(value: str | None) -> str | None:
    if value is None:
        return None
    return _normalize_split(value, field="split")


def _normalize_split(value: str, *, field: str) -> str:
    """Normalize to a canonical ImageFolder split name, or raise.

    Only ``train``/``validation``/``test`` (via the alias table) round-trip:
    the IC loader reads any other top-level directory as a class folder and
    the OD loader drops it, so unknown split names are rejected here rather
    than silently written into a layout the loader cannot interpret.
    """
    raw = str(value).strip()
    normalized = _SPLIT_ALIASES.get(raw.lower().replace("_", "-"))
    if normalized is None or normalized not in _CANONICAL_SPLITS:
        raise ValueError(
            f"{field} must be a Hugging Face ImageFolder split (train/validation/test or an alias "
            f"such as val/dev/eval); got {value!r}"
        )
    return normalized


def _validate_metadata_format(value: str) -> str:
    metadata_format = str(value).strip().lower()
    if metadata_format not in _METADATA_FORMATS:
        raise ValueError(f"metadata_format must be one of {sorted(_METADATA_FORMATS)!r}; got {value!r}")
    return metadata_format


def _safe_file_name(sample: ImageClassificationSample | ImageObjectDetectionSample) -> str:
    raw = sample.file_name or (Path(sample.path_or_uri).name if sample.path_or_uri else f"{sample.image_id}.jpg")
    name = Path(str(raw)).name
    if not name or name in {".", ".."} or "\x00" in name or "\\" in name:
        raise ValueError(f"unsafe file name: {raw!r}")
    return name


def _safe_path_part(value: str, *, field: str) -> str:
    """Reject only genuinely unsafe path components, preserving spaces/unicode.

    Mirrors the sibling YOLO writer's ``safe_path_part``: a class folder named
    ``traffic light`` or ``café`` is a legitimate ImageFolder label that the IC
    loader reads straight back as the label, so slugging it (as ``_safe_name``
    would) breaks write->reload label identity. Only ``''``, ``.``, ``..``,
    path separators, and NUL are unsafe as a single directory component.
    """
    candidate = str(value).strip()
    if not candidate or candidate in {".", ".."} or "/" in candidate or "\\" in candidate or "\x00" in candidate:
        raise ValueError(f"{field} must be a safe path component; got {value!r}")
    return candidate


def _safe_name(value: str) -> str:
    candidate = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    return candidate or "item"


def _unique_target(target: Path, used: set[Path]) -> Path:
    if target not in used and not target.exists():
        return target
    stem, suffix = target.stem, target.suffix
    for index in range(1, 1_000_000):
        candidate = target.with_name(f"{stem}-{index}{suffix}")
        if candidate not in used and not candidate.exists():
            return candidate
    raise ValueError(f"could not allocate unique Hugging Face vision target near {target}")


def _unique_rel_path(prefix: str, stem: str, suffix: str, used: set[str]) -> str:
    prefix_parts = _safe_prefix_parts(prefix)
    safe_stem = _safe_name(stem)
    rel_path = PurePosixPath(*prefix_parts, f"{safe_stem}{suffix}").as_posix()
    if rel_path not in used:
        used.add(rel_path)
        return rel_path
    index = 1
    while True:
        rel_path = PurePosixPath(*prefix_parts, f"{safe_stem}-{index}{suffix}").as_posix()
        if rel_path not in used:
            used.add(rel_path)
            return rel_path
        index += 1


def _safe_prefix_parts(prefix: str) -> tuple[str, ...]:
    parts = tuple(_safe_name(part) for part in prefix.replace("\\", "/").split("/") if part)
    return parts or ("data",)


def _finite(value: float) -> bool:
    return math.isfinite(value)
