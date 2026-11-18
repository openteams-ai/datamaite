"""YOLO/Ultralytics dataset writers.

The module mirrors :mod:`datamaite._formats.yolo.loader`: one writer is
registered for image classification (``Task.IC``) and one for object detection
(``Task.OD``) under the same ``DatasetFormat.YOLO`` family.
"""

from __future__ import annotations

import json
import logging
import math
import numbers
import os
import posixpath
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar

import yaml

from datamaite._formats.yolo._common import (
    OD_SPLIT_KEYS,
    RESERVED_YAML_KEYS,
    free_target,
    infer_split,
    names_from_yaml,
    safe_path_part,
    safe_relative_path,
    split_sort_key,
    ultralytics_split_path,
    unique_target,
)
from datamaite._io import (
    copy_resource,
    current_write_mode,
    ensure_directory,
    list_files,
    resolve_path,
    same_resource,
    sample_image_dimensions,
    source_path,
)
from datamaite._types import DatasetFormat, Task
from datamaite._upath import is_remote_path, to_dataset_path
from datamaite.geometry import clamp_to_image, has_positive_area, to_yolo
from datamaite.image_classification import ImageClassificationDataset
from datamaite.object_detection import ObjectDetectionDataset
from datamaite.records import (
    ClassificationLabel,
    ImageClassificationSample,
    ImageObjectDetectionSample,
    ObjectDetectionAnnotation,
)
from datamaite.taxonomy import SourceId, Taxonomy
from datamaite.writers import Writer, WriterCapabilities, _validate_include_images, register_writer

logger = logging.getLogger(__name__)

_APPEND_INVENTORY_THRESHOLD = 64
_PLAIN_YAML = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./-]*")


@register_writer
class YoloImageClassificationWriter(Writer[ImageClassificationDataset]):
    """Write YOLO/Ultralytics image-classification folder datasets."""

    task: ClassVar[Task] = Task.IC
    format = DatasetFormat.YOLO
    variant: ClassVar[str] = "default"
    consumes: ClassVar[type] = ImageClassificationDataset
    capabilities: ClassVar[WriterCapabilities] = WriterCapabilities(required_fields=frozenset({"image", "labels"}))

    def write(
        self,
        dataset: ImageClassificationDataset,
        dest: str | Path,
        *,
        default_split: str = "train",
        write_data_yaml: bool = True,
        **_: Any,
    ) -> list[Path]:
        """Serialise an IC dataset as split/class/image directories.

        Samples with neither ``image_bytes`` nor ``path_or_uri`` are skipped with
        a warning. A folder classification format can represent only one class
        per image; when a sample has multiple labels, the first label is used
        and the rest are left in the source model rather than invented on disk.
        """
        dest_path = to_dataset_path(dest, _.get("storage_options"))
        ensure_directory(dest_path)
        taxonomy = dataset.dataset_metadata.taxonomy
        written: list[Path] = []
        used_targets: set[Path] = set()
        inventoried_dirs = _append_inventory(dataset.samples, dest_path)
        splits_seen: set[str] = set()
        classes_seen: set[str] = set()

        for sample in dataset.samples:
            target = _write_ic_sample(
                sample,
                dest_path,
                taxonomy=taxonomy,
                default_split=default_split,
                used_targets=used_targets,
                inventoried_dirs=inventoried_dirs,
            )
            if target is None:
                continue
            used_targets.add(target)
            splits_seen.add(target.parent.parent.name)
            classes_seen.add(target.parent.name)
            written.append(target)

        # data.yaml carries taxonomy/splits even when no sample was writable.
        # Object stores additionally need marker objects because empty prefixes
        # do not exist.
        if taxonomy is not None:
            marker_splits = splits_seen or set(dataset.dataset_metadata.splits) or {default_split}
            for entry in taxonomy.entries:
                safe_class = safe_path_part(entry.name, field="class name")
                classes_seen.add(safe_class)
                for split_name in marker_splits:
                    safe_split = safe_path_part(split_name, field="split")
                    class_dir = dest_path / safe_split / safe_class
                    if is_remote_path(dest_path):
                        if not any(path.parent == class_dir for path in used_targets):
                            marker = class_dir / ".datamaite-empty"
                            ensure_directory(marker.parent)
                            marker.write_bytes(b"")
                            written.append(marker)
                    elif written:
                        # Local filesystems preserve empty directories. Only
                        # materialize them once at least one sample succeeded,
                        # retaining no-stray-output behavior for all-skipped writes.
                        ensure_directory(class_dir)
                    splits_seen.add(safe_split)

        if write_data_yaml:
            data_yaml = dest_path / "data.yaml"
            yaml_text = _ic_data_yaml(
                splits=tuple(sorted(splits_seen, key=split_sort_key)),
                names=sorted(classes_seen),
            )
            data_yaml.write_text(yaml_text, encoding="utf-8")
            written.append(data_yaml)
        return written


@register_writer
class YoloObjectDetectionWriter(Writer[ObjectDetectionDataset]):
    """Write YOLO/Ultralytics object-detection datasets."""

    task: ClassVar[Task] = Task.OD
    format = DatasetFormat.YOLO
    variant: ClassVar[str] = "default"
    consumes: ClassVar[type] = ObjectDetectionDataset
    capabilities: ClassVar[WriterCapabilities] = WriterCapabilities(
        required_fields=frozenset({"image", "width", "height"}),
        lossy_without={
            "score": "standard YOLO training labels do not store confidence",
            "area": "YOLO detection TXT stores boxes/classes only",
            "segmentation": "YOLO detection TXT stores boxes/classes only",
            "iscrowd": "YOLO detection TXT stores boxes/classes only",
        },
        emits_empty_label_files=True,
    )

    def validate_options(self, **options: Any) -> None:
        """Validate options that can raise, before write()'s destination policy runs (#55 Fix A1).

        Mirrors the inline ``precision`` and ``include_images`` checks in ``write()``, but only for
        options that are present, so a ``mode="replace"`` clear never happens
        ahead of an option error. ``write()`` re-validates inline, which also
        covers direct ``Writer.write()`` calls. (``default_split`` is validated
        per-sample inside ``write()`` and only skips offending samples, so it
        cannot raise from ``write()`` and is not pre-checked here.)
        """
        if "precision" in options:
            precision = options["precision"]
            if isinstance(precision, bool) or not isinstance(precision, int) or precision < 1:
                raise ValueError(f"precision must be a non-boolean integer >= 1, got {precision!r}")
        if "include_images" in options:
            _validate_include_images(options["include_images"])

    def write(
        self,
        dataset: ObjectDetectionDataset,
        dest: str | Path,
        *,
        default_split: str = "train",
        include_images: bool = True,
        write_data_yaml: bool = True,
        precision: int = 6,
        include_scores: bool = False,
        **_: Any,
    ) -> list[Path]:
        """Serialise an OD dataset as ``images/<split>`` + ``labels/<split>``.

        YOLO detection labels require image dimensions to normalize canonical
        absolute-pixel boxes. Detections on samples without valid ``width`` and
        ``height`` are skipped with a warning; the image still receives an empty
        label file so the emitted root remains a valid YOLO dataset. Scores are
        omitted by default for training-label compatibility; pass
        ``include_scores=True`` to emit prediction-style six-column labels.
        """
        if isinstance(precision, bool) or not isinstance(precision, int) or precision < 1:
            raise ValueError(f"precision must be a non-boolean integer >= 1, got {precision!r}")
        _validate_include_images(include_images)
        dest_path = to_dataset_path(dest, _.get("storage_options"))
        ensure_directory(dest_path)
        projection = _LabelProjection.from_dataset(dataset)
        # Appending reads the data.yaml there even when it is not rewritten: its class ids are the ones to keep.
        existing = _existing_od_data_yaml(dest_path)
        images_folder = "images"
        if existing is not None:
            # Appending: keep the written classes and their ids, and add new names after them (#130).
            projection = projection.onto(existing.names)
            # Checked now, so an append whose new folders data.yaml can't name is refused before writing.
            images_folder = _images_folder(dest_path, existing.path) if write_data_yaml else "images"
            if not write_data_yaml and len(projection.names) > len(existing.names):
                new = list(projection.names[len(existing.names) :])
                raise ValueError(
                    f"cannot append to YOLO dataset with write_data_yaml=False: classes {new} are not in "
                    f"{dest_path / 'data.yaml'}, which would not be updated to name them"
                )
        written: list[Path] = []
        used_images: set[Path] = set()
        used_labels: set[Path] = set()
        inventoried_dirs = _append_inventory(dataset.samples, dest_path)
        splits_seen: set[str] = set()

        for sample in dataset.samples:
            paths = _write_od_sample(
                sample,
                dest_path,
                projection=projection,
                default_split=default_split,
                include_images=include_images,
                precision=precision,
                include_scores=include_scores,
                used_images=used_images,
                used_labels=used_labels,
                inventoried_dirs=inventoried_dirs,
            )
            if paths is None:
                continue
            image_path, label_path, split = paths
            if image_path is not None:
                written.append(image_path)
            written.append(label_path)
            # The sample's split, not the label's parent folder, which is a subfolder for nested names (#131).
            splits_seen.add(split)

        if not splits_seen and projection.names and existing is None:
            splits_seen.update(dataset.dataset_metadata.splits or (default_split,))
        if write_data_yaml:
            sources = _split_sources(
                splits_seen, existing.sources if existing is not None else {}, images_folder=images_folder
            )
            data_yaml = dest_path / "data.yaml"
            data_yaml.write_text(
                _od_data_yaml(sources=sources, names=list(projection.names), existing=existing), encoding="utf-8"
            )
            written.append(data_yaml)
        return written


# ---------------------------------------------------------------------------
# Classification writer helpers
# ---------------------------------------------------------------------------


def _append_inventory(samples: Any, dest: Path) -> set[Path] | None:
    if is_remote_path(dest) and current_write_mode() == "append" and len(samples) >= _APPEND_INVENTORY_THRESHOLD:
        return set()
    return None


def _inventory_directory(directory: Path, used: set[Path], inventoried: set[Path] | None) -> None:
    if inventoried is None or directory in inventoried:
        return
    used.update(list_files(directory))
    inventoried.add(directory)


def _write_ic_sample(
    sample: ImageClassificationSample,
    dest_path: Path,
    *,
    taxonomy: Taxonomy | None,
    default_split: str,
    used_targets: set[Path],
    inventoried_dirs: set[Path] | None,
) -> Path | None:
    """Write one IC sample to ``<dest>/<split>/<class>/<file>``; return the path."""
    if getattr(sample, "region", None) is not None:
        # This writer copies the whole source image; it cannot crop. A sample with
        # a crop region (e.g. a VisDrone-derived object crop) would be emitted as
        # its full source image under a class label it does not depict, silently
        # producing an incorrect dataset. Skip it loudly instead.
        logger.warning(
            "Skipping YOLO IC sample %r: it carries a crop region this image-copying writer "
            "cannot represent; emitting the full source image would mislabel it",
            sample.image_id,
        )
        return None
    label = _single_label(sample)
    if label is None:
        logger.warning("Skipping YOLO IC sample %r with no labels", sample.image_id)
        return None
    class_name = _ic_class_name(label, taxonomy)
    if class_name is None:
        logger.warning("Skipping YOLO IC sample %r with unresolved label %r", sample.image_id, label)
        return None
    try:
        split = safe_path_part(sample.split or default_split, field="split")
        safe_class_name = safe_path_part(class_name, field="class name")
        file_name = _safe_ic_file_name(sample)
    except ValueError as exc:
        logger.warning("Skipping YOLO IC sample %r: %s", sample.image_id, exc)
        return None

    # Resolve the image source before creating any directory, so a skipped
    # sample never leaves an empty class folder behind.
    source: Path | None = None
    if sample.image_bytes is None:
        if sample.path_or_uri is None:
            logger.warning("Skipping YOLO IC sample %r with no image source", sample.image_id)
            return None
        source = source_path(sample.path_or_uri)
        if not is_remote_path(source) and not source.is_file():
            logger.warning("Skipping YOLO IC sample %r with missing image file: %s", sample.image_id, source)
            return None

    class_dir = dest_path / split / safe_class_name
    created_directories = _ensure_output_directory(class_dir, dest_path)
    _inventory_directory(class_dir, used_targets, inventoried_dirs)
    target = unique_target(class_dir / file_name, used_targets, check_storage=inventoried_dirs is None)
    if sample.image_bytes is not None:
        target.write_bytes(sample.image_bytes)
    elif source is not None and not _copy_ic_source(sample, source, target, created_directories):
        return None
    return target


def _copy_ic_source(
    sample: ImageClassificationSample, source: Path, target: Path, created_directories: list[Path]
) -> bool:
    try:
        copy_resource(source, target)
    except FileNotFoundError:
        _remove_created_output_directories(created_directories)
        logger.warning("Skipping YOLO IC sample %r with missing image file: %s", sample.image_id, source)
        return False
    return True


def _ensure_output_directory(directory: Path, dest: Path) -> list[Path]:
    if is_remote_path(directory):
        return []
    created: list[Path] = []
    current = directory
    while current != dest and not current.exists():
        created.append(current)
        current = current.parent
    ensure_directory(directory)
    return created


def _remove_created_output_directories(directories: list[Path]) -> None:
    for directory in directories:
        try:
            directory.rmdir()
        except OSError:  # noqa: PERF203 - stop at first non-empty dir
            return


def _single_label(sample: ImageClassificationSample) -> ClassificationLabel | None:
    if not sample.labels:
        return None
    if len(sample.labels) > 1:
        logger.warning("YOLO classification can emit one label per image; using first label for %r", sample.image_id)
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


def _safe_ic_file_name(sample: ImageClassificationSample) -> str:
    raw = sample.file_name or (
        PurePosixPath(sample.path_or_uri).name if sample.path_or_uri else f"{sample.image_id}.jpg"
    )
    name = PurePosixPath(str(raw)).name
    if not name or name in {".", ".."} or name != str(raw).replace("\\", "/").rsplit("/", 1)[-1] or "\x00" in name:
        raise ValueError(f"unsafe file name: {raw!r}")
    return name


def _ic_data_yaml(*, splits: tuple[str, ...], names: list[str]) -> str:
    # ``names`` must be the class-folder names actually written, sorted -- the same
    # order the loader (and Ultralytics) derive class indices from. Deriving from
    # the taxonomy's own order instead would mislabel classes for any non-
    # alphabetical taxonomy, since on-disk folders are read back alphabetically.
    lines = ["# Generated by datamaite", "datamaite_task: ic", f"path: {json.dumps('.')}"]
    if splits:
        if "train" in splits:
            lines.append("train: train")
        if "val" in splits:
            lines.append("val: val")
        if "test" in splits:
            lines.append("test: test")
    lines.append(f"names: {json.dumps(names)}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Object-detection writer helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _LabelProjection:
    names: tuple[str, ...]
    by_source_id: Mapping[SourceId, int]
    by_name: Mapping[str, int]

    @classmethod
    def from_dataset(cls, dataset: ObjectDetectionDataset) -> _LabelProjection:
        taxonomy = dataset.dataset_metadata.taxonomy
        if taxonomy is not None and taxonomy.entries:
            by_source_id: dict[SourceId, int] = {}
            by_name: dict[str, int] = {}
            for index, entry in enumerate(taxonomy.entries):
                if entry.source_id not in by_source_id:
                    by_source_id[entry.source_id] = index
                by_name.setdefault(entry.name, index)
            return cls(
                names=tuple(entry.name for entry in taxonomy.entries),
                by_source_id=by_source_id,
                by_name=by_name,
            )

        discovered: list[tuple[SourceId, str]] = []
        seen: set[SourceId] = set()
        for sample in dataset.samples:
            for detection in sample.detections:
                source_id = _detection_source_id(detection)
                if source_id is None or source_id in seen:
                    continue
                seen.add(source_id)
                discovered.append((source_id, detection.category_name or str(source_id)))
        # Integer ids in numeric order (2 before 10), then any others as text (#134).
        discovered.sort(
            key=lambda item: (
                (0, int(item[0]), "")
                if isinstance(item[0], numbers.Integral) and not isinstance(item[0], bool)
                else (1, 0, str(item[0]))
            )
        )
        by_name: dict[str, int] = {}
        for index, (_, name) in enumerate(discovered):
            by_name.setdefault(name, index)
        return cls(
            names=tuple(name for _, name in discovered),
            by_source_id={source_id: index for index, (source_id, _) in enumerate(discovered)},
            by_name=by_name,
        )

    def onto(self, names: Sequence[str]) -> _LabelProjection:
        """This projection renumbered onto ``names``: classes match by name, and new names follow them."""
        merged = list(names)
        position = {}
        for index, name in enumerate(merged):
            position.setdefault(name, index)
        remap: dict[int, int] = {}
        for index, name in enumerate(self.names):
            if name not in position:
                position[name] = len(merged)
                merged.append(name)
            remap[index] = position[name]
        return _LabelProjection(
            names=tuple(merged),
            by_source_id={source_id: remap[index] for source_id, index in self.by_source_id.items()},
            by_name=dict(position),
        )

    def resolve(self, detection: ObjectDetectionAnnotation) -> int | None:
        source_id = _detection_source_id(detection)
        if source_id in self.by_source_id:
            return self.by_source_id[source_id]
        if detection.category_name is not None and detection.category_name in self.by_name:
            return self.by_name[detection.category_name]
        return None


def _detection_source_id(detection: ObjectDetectionAnnotation) -> SourceId:
    return detection.source_category_id if detection.source_category_id is not None else detection.category_id


def _write_od_sample(
    sample: ImageObjectDetectionSample,
    dest: Path,
    *,
    projection: _LabelProjection,
    default_split: str,
    include_images: bool,
    precision: int,
    include_scores: bool,
    used_images: set[Path],
    used_labels: set[Path],
    inventoried_dirs: set[Path] | None,
) -> tuple[Path | None, Path, str] | None:
    try:
        split = safe_path_part(sample.split or default_split, field="split")
        rel_image = _safe_od_image_rel(sample, split=split)
    except ValueError as exc:
        logger.warning("Skipping YOLO OD sample %r: %s", sample.image_id, exc)
        return None

    source: Path | None = None
    if include_images and sample.image_bytes is None:
        if sample.path_or_uri is None:
            logger.warning("Skipping YOLO OD sample %r with no image source", sample.image_id)
            return None
        source = source_path(sample.path_or_uri)
        if not is_remote_path(source) and not source.is_file():
            logger.warning("Skipping YOLO OD sample %r with missing image file: %s", sample.image_id, source)
            return None

    image_root = dest / "images" / split
    label_root = dest / "labels" / split
    image_seed = image_root / rel_image.as_posix()
    label_seed = label_root.joinpath(*rel_image.with_suffix(".txt").parts)
    if include_images:
        _inventory_directory(image_seed.parent, used_images, inventoried_dirs)
    _inventory_directory(label_seed.parent, used_labels, inventoried_dirs)
    image_target, label_target = _unique_od_targets(
        image_seed,
        image_root,
        label_root,
        used_images,
        used_labels,
        require_image_free=include_images,
        check_storage=inventoried_dirs is None,
    )
    label_lines = _od_label_lines(
        _with_image_dimensions(sample), projection=projection, precision=precision, include_scores=include_scores
    )

    image_path: Path | None = None
    if include_images:
        created_directories = _ensure_output_directory(image_target.parent, dest)
        if sample.image_bytes is not None:
            image_target.write_bytes(sample.image_bytes)
        elif source is not None and not same_resource(source, image_target):
            try:
                copy_resource(source, image_target)
            except FileNotFoundError:
                _remove_created_output_directories(created_directories)
                logger.warning("Skipping YOLO OD sample %r with missing image file: %s", sample.image_id, source)
                return None
        used_images.add(image_target)
        image_path = image_target

    ensure_directory(label_target.parent)
    label_target.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
    used_labels.add(label_target)
    return (image_path, label_target, split)


def _safe_od_image_rel(sample: ImageObjectDetectionSample, *, split: str) -> PurePosixPath:
    raw = sample.file_name or (
        PurePosixPath(sample.path_or_uri).name if sample.path_or_uri else f"{sample.image_id}.jpg"
    )
    posix = safe_relative_path(str(raw), field="file name")
    parts = list(posix.parts)
    if parts and parts[0] == "images":
        parts.pop(0)
    if parts and infer_split(parts[0]) == split:
        parts.pop(0)
    if parts and parts[0] == "images":
        parts.pop(0)
    if not parts:
        raise ValueError(f"unsafe file name: {raw!r}")
    if parts[0] == "labels":
        raise ValueError(f"unsafe file name: {raw!r}")
    return PurePosixPath(*parts)


def _unique_od_targets(
    image_target: Path,
    image_root: Path,
    label_root: Path,
    used_images: set[Path],
    used_labels: set[Path],
    *,
    require_image_free: bool,
    check_storage: bool,
) -> tuple[Path, Path]:
    for candidate in _candidate_targets(image_target):
        rel = PurePosixPath(*candidate.relative_to(image_root).parts)
        label_candidate = label_root.joinpath(*rel.with_suffix(".txt").parts)
        image_available = (
            free_target(candidate, used_images, check_storage=check_storage)
            if require_image_free
            else candidate not in used_images
        )
        if image_available and free_target(label_candidate, used_labels, check_storage=check_storage):
            return candidate, label_candidate
    raise ValueError(f"could not allocate unique YOLO OD target near {image_target}")


def _candidate_targets(target: Path) -> Iterable[Path]:
    yield target
    stem = target.stem
    suffix = target.suffix
    for index in range(2, 1_000_000):
        yield target.with_name(f"{stem}_{index}{suffix}")


def _with_image_dimensions(sample: ImageObjectDetectionSample) -> ImageObjectDetectionSample:
    """Fill missing or non-positive ``width``/``height`` from the image header.

    Normalized YOLO boxes need image dimensions. Formats such as VisDrone do not
    store them, so read them instead of dropping every label on the sample.
    """
    if not sample.detections:
        return sample
    dimensions = sample_image_dimensions(sample)
    if dimensions is None or dimensions == (sample.width, sample.height):
        return sample
    return replace(sample, width=dimensions[0], height=dimensions[1])


def _od_label_lines(  # noqa: C901 - per-detection validation is intentionally explicit
    sample: ImageObjectDetectionSample,
    *,
    projection: _LabelProjection,
    precision: int,
    include_scores: bool,
) -> list[str]:
    if not sample.detections:
        return []
    if sample.width is None or sample.height is None or sample.width <= 0 or sample.height <= 0:
        logger.warning("Skipping YOLO OD labels for sample %r: missing/invalid image width/height", sample.image_id)
        return []
    lines: list[str] = []
    dropped_lossy_fields = 0
    dropped_scores = 0
    for detection in sample.detections:
        class_id = projection.resolve(detection)
        if class_id is None:
            logger.warning(
                "Skipping YOLO OD detection on sample %r with unresolved category %r",
                sample.image_id,
                detection,
            )
            continue
        if not has_positive_area(detection.bbox):
            logger.warning(
                "Skipping YOLO OD detection on sample %r with invalid bbox %r",
                sample.image_id,
                detection.bbox,
            )
            continue
        image_width = float(sample.width)
        image_height = float(sample.height)
        try:
            clamped_bbox = clamp_to_image(detection.bbox, image_width, image_height)
        except ValueError as exc:
            logger.warning("Skipping YOLO OD detection on sample %r: %s", sample.image_id, exc)
            continue
        if not has_positive_area(clamped_bbox):
            logger.warning(
                "Skipping YOLO OD detection on sample %r with bbox outside image after clipping %r",
                sample.image_id,
                detection.bbox,
            )
            continue
        try:
            yolo_box = to_yolo(clamped_bbox, image_width, image_height)
        except ValueError as exc:
            logger.warning("Skipping YOLO OD detection on sample %r: %s", sample.image_id, exc)
            continue
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in yolo_box):
            logger.warning(
                "Skipping YOLO OD detection on sample %r with out-of-range normalized bbox %r",
                sample.image_id,
                yolo_box,
            )
            continue
        fields = [str(class_id), *(_format_float(value, precision=precision) for value in yolo_box)]
        if detection.score is not None:
            if include_scores:
                fields.append(_format_float(float(detection.score), precision=precision))
            else:
                dropped_scores += 1
        if detection.area is not None or detection.segmentation is not None or detection.iscrowd:
            dropped_lossy_fields += 1
        lines.append(" ".join(fields))
    if dropped_scores:
        logger.warning(
            "Dropped score from %d YOLO OD detection(s) on sample %r: standard training labels do not store confidence",
            dropped_scores,
            sample.image_id,
        )
    if dropped_lossy_fields:
        logger.warning(
            "Dropped YOLO-unrepresentable OD fields (area/segmentation/iscrowd) from %d detection(s) on sample %r",
            dropped_lossy_fields,
            sample.image_id,
        )
    return lines


def _format_float(value: float, *, precision: int) -> str:
    text = f"{value:.{precision}f}" if precision else f"{value:.0f}"
    text = text.rstrip("0").rstrip(".") if "." in text else text
    if text in {"", "-0"}:
        return "0"
    return text


@dataclass(frozen=True)
class _ExistingDataYaml:
    """What an append must keep from the data.yaml already at the destination."""

    names: tuple[str, ...]
    sources: Mapping[str, tuple[str, ...]]  # data.yaml split key -> image folders
    path: str | None  # its ``path:``, which the split folders are relative to
    has_nc: bool  # whether it has ``nc:``, which is then rewritten with the new class count
    others: Mapping[str, Any]  # every other key, such as Roboflow's ``roboflow:``, kept as it is


def _existing_od_data_yaml(dest: Path) -> _ExistingDataYaml | None:
    """The destination's data.yaml when appending, or ``None`` when not appending or there is none.

    Raises ``ValueError`` before anything is written when the file can't be
    appended to, so a refused append leaves the dataset as it was.
    """
    if current_write_mode() != "append":
        return None
    path = dest / "data.yaml"
    if not path.is_file():
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot append to YOLO dataset: could not read {path}: {exc}") from exc
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"cannot append to YOLO dataset: {path} is not a mapping")
    task = data.get("datamaite_task")
    if task is not None and str(task).lower() != Task.OD.value:
        raise ValueError(f"cannot append object detection data to {path}, which declares task {task!r}")
    pairs = names_from_yaml(data.get("names"))
    if [class_id for class_id, _ in pairs] != list(range(len(pairs))):
        raise ValueError(
            f"cannot append to YOLO dataset: the names in {path} are not numbered 0..n-1, "
            "so new classes have no free ids to take"
        )
    raw_path = data.get("path")
    path = raw_path.strip() if isinstance(raw_path, str) and raw_path.strip() else None
    sources = _yaml_split_sources(data, resolve_path(dest, path))
    others = {str(key): value for key, value in data.items() if key not in sources and key not in _REWRITTEN_YAML_KEYS}
    return _ExistingDataYaml(
        names=tuple(name for _, name in pairs), sources=sources, path=path, has_nc="nc" in data, others=others
    )


#: The data.yaml keys an append writes itself; it keeps every other key as it is.
_REWRITTEN_YAML_KEYS = frozenset({"path", "names", "nc", "datamaite_task"})


def _yaml_split_sources(data: Mapping[Any, Any], base: Path) -> dict[str, tuple[str, ...]]:
    """The data.yaml keys that list image folders or files, which an append keeps as splits.

    As the loader reads them: train, val and test always, any other key only when
    it names folders that exist. A missing ``../x`` that Ultralytics reads as
    ``x`` is kept as ``x``, so it still resolves once the key lists more folders.
    """
    sources: dict[str, tuple[str, ...]] = {}
    for key, value in data.items():
        if not isinstance(key, str) or key in RESERVED_YAML_KEYS:
            continue
        value = ultralytics_split_path(base, key, value)
        values = [value] if isinstance(value, str) else value if isinstance(value, list) else None
        if not values or not all(isinstance(item, str) and item.strip() for item in values):
            continue
        folders = tuple(item.strip() for item in values)
        if key in OD_SPLIT_KEYS or all(resolve_path(base, folder).is_dir() for folder in folders):
            sources[key] = folders
    return sources


def _images_folder(dest: Path, yaml_path: str | None) -> str:
    """``dest/images`` as data.yaml names it: relative to the base its ``path:`` resolves to.

    That is ``images`` when ``path:`` is the destination, and for example
    ``../out/images`` when an append keeps a ``path: ../shared``. Relative, so the
    dataset can still be moved. Raises ``ValueError``, before anything is written,
    when ``dest`` can't be reached from that base by a relative path.
    """
    base = resolve_path(dest, yaml_path)
    if is_remote_path(dest) or is_remote_path(base):
        relative = _remote_relative(dest, base)
    else:
        try:
            relative = os.path.relpath(dest.resolve(), base.resolve()).replace(os.sep, "/")
        except ValueError:  # Windows: another drive
            relative = None
    if relative is None:
        raise ValueError(
            f"cannot append to YOLO dataset: {dest} can't be reached by a relative path from its "
            f"data.yaml path {yaml_path!r}"
        )
    return "images" if relative == "." else f"{relative}/images"


def _remote_relative(dest: Path, base: Path) -> str | None:
    """``dest`` relative to ``base`` on the same remote filesystem and bucket, or ``None``."""
    if getattr(dest, "protocol", None) != getattr(base, "protocol", None):
        return None
    dest_parts = PurePosixPath(str(getattr(dest, "path", dest))).parts
    base_parts = PurePosixPath(str(getattr(base, "path", base))).parts
    if not dest_parts or not base_parts or dest_parts[0] != base_parts[0]:
        return None
    return posixpath.relpath("/" + "/".join(dest_parts), "/" + "/".join(base_parts))


_YEAR_SUFFIX = re.compile(r"(?<=\D)\d{4}$")


def _split_sources(
    splits: Iterable[str], existing: Mapping[str, tuple[str, ...]], *, images_folder: str
) -> dict[str, tuple[str, ...]]:
    """data.yaml split keys and their image folders for every split written, after ``existing``'s.

    A split's key is its train/val/test alias (``validation`` and ``valid`` go
    under ``val``, as Ultralytics reads them), also with a year after it
    (COCO 2017's ``train2017`` goes under ``train``), or else its own name;
    splits that share a key are listed together.
    """
    sources: dict[str, list[str]] = {key: list(values) for key, values in existing.items()}
    for split in sorted(splits, key=split_sort_key):
        key = infer_split(split) or infer_split(_YEAR_SUFFIX.sub("", split)) or split
        if key in RESERVED_YAML_KEYS:
            logger.warning("YOLO split %r is a reserved data.yaml key, so data.yaml does not list it", split)
            continue
        image_folder = f"{images_folder}/{split}"
        folders = sources.setdefault(key, [])
        if image_folder not in folders:
            folders.append(image_folder)
    return {key: tuple(sources[key]) for key in sorted(sources, key=split_sort_key)}


def _yaml_text(value: str) -> str:
    """``value`` as a YAML string: bare when YAML reads it back unchanged, else quoted."""
    if _PLAIN_YAML.fullmatch(value):
        try:
            if yaml.safe_load(value) == value:
                return value
        except yaml.YAMLError:
            pass
    return json.dumps(value)


def _od_data_yaml(
    *, sources: Mapping[str, tuple[str, ...]], names: list[str], existing: _ExistingDataYaml | None = None
) -> str:
    path = existing.path if existing is not None else None
    lines = ["# Generated by datamaite", "datamaite_task: od", f"path: {json.dumps(path or '.')}"]
    for key, folders in sources.items():
        value = _yaml_text(folders[0]) if len(folders) == 1 else f"[{', '.join(_yaml_text(f) for f in folders)}]"
        lines.append(f"{_yaml_text(key)}: {value}")
    if existing is not None and existing.has_nc:
        lines.append(f"nc: {len(names)}")
    lines.append(f"names: {json.dumps(names)}")
    if existing is not None and existing.others:
        # Kept as the appended dataset found them; JSON is YAML.
        lines.extend(
            f"{_yaml_text(key)}: {json.dumps(value, default=str)}"
            for key, value in existing.others.items()
            if key not in sources
        )
    return "\n".join(lines) + "\n"
