"""YOLO/Ultralytics dataset writers.

The module mirrors :mod:`datamaite._formats.yolo.loader`: one writer is
registered for image classification (``Task.IC``) and one for object detection
(``Task.OD``) under the same ``DatasetFormat.YOLO`` family.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar

from datamaite._formats.yolo._common import (
    free_target,
    infer_split,
    safe_path_part,
    safe_relative_path,
    split_sort_key,
    unique_target,
)
from datamaite._types import DatasetFormat, Task
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
from datamaite.writers import Writer, WriterCapabilities, register_writer

logger = logging.getLogger(__name__)


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
        dest_path = Path(dest)
        dest_path.mkdir(parents=True, exist_ok=True)
        taxonomy = dataset.dataset_metadata.taxonomy
        written: list[Path] = []
        used_targets: set[Path] = set()
        splits_seen: set[str] = set()
        classes_seen: set[str] = set()

        for sample in dataset.samples:
            target = _write_ic_sample(
                sample,
                dest_path,
                taxonomy=taxonomy,
                default_split=default_split,
                used_targets=used_targets,
            )
            if target is None:
                continue
            used_targets.add(target)
            splits_seen.add(target.parent.parent.name)
            classes_seen.add(target.parent.name)
            written.append(target)

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

        Mirrors the inline ``precision`` check in ``write()``, but only for
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
        dest_path = Path(dest)
        dest_path.mkdir(parents=True, exist_ok=True)
        projection = _LabelProjection.from_dataset(dataset)
        written: list[Path] = []
        used_images: set[Path] = set()
        used_labels: set[Path] = set()
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
            )
            if paths is None:
                continue
            image_path, label_path = paths
            if image_path is not None:
                written.append(image_path)
            written.append(label_path)
            splits_seen.add(label_path.parent.name)

        if write_data_yaml:
            data_yaml = dest_path / "data.yaml"
            data_yaml.write_text(
                _od_data_yaml(splits=tuple(sorted(splits_seen, key=split_sort_key)), names=list(projection.names)),
                encoding="utf-8",
            )
            written.append(data_yaml)
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
        source = Path(sample.path_or_uri)
        if not source.is_file():
            logger.warning("Skipping YOLO IC sample %r with missing image file: %s", sample.image_id, source)
            return None

    class_dir = dest_path / split / safe_class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    target = unique_target(class_dir / file_name, used_targets)
    if sample.image_bytes is not None:
        target.write_bytes(sample.image_bytes)
    elif source is not None:  # a verified file, set above when image_bytes is None
        shutil.copy2(source, target)
    return target


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
    raw = sample.file_name or (Path(sample.path_or_uri).name if sample.path_or_uri else f"{sample.image_id}.jpg")
    name = Path(str(raw)).name
    if not name or name in {".", ".."} or name != str(raw).replace("\\", "/").rsplit("/", 1)[-1] or "\x00" in name:
        raise ValueError(f"unsafe file name: {raw!r}")
    return name


def _ic_data_yaml(*, splits: tuple[str, ...], names: list[str]) -> str:
    # ``names`` must be the class-folder names actually written, sorted -- the same
    # order the loader (and Ultralytics) derive class indices from. Deriving from
    # the taxonomy's own order instead would mislabel classes for any non-
    # alphabetical taxonomy, since on-disk folders are read back alphabetically.
    lines = ["# Generated by datamaite", f"path: {json.dumps('.')}"]
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
        discovered.sort(
            key=lambda item: (0 if isinstance(item[0], int) and not isinstance(item[0], bool) else 1, str(item[0]))
        )
        by_name: dict[str, int] = {}
        for index, (_, name) in enumerate(discovered):
            by_name.setdefault(name, index)
        return cls(
            names=tuple(name for _, name in discovered),
            by_source_id={source_id: index for index, (source_id, _) in enumerate(discovered)},
            by_name=by_name,
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
) -> tuple[Path | None, Path] | None:
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
        source = Path(sample.path_or_uri)
        if not source.is_file():
            logger.warning("Skipping YOLO OD sample %r with missing image file: %s", sample.image_id, source)
            return None

    image_root = dest / "images" / split
    label_root = dest / "labels" / split
    image_target, label_target = _unique_od_targets(
        image_root / rel_image.as_posix(),
        image_root,
        label_root,
        used_images,
        used_labels,
        require_image_free=include_images,
    )
    label_lines = _od_label_lines(sample, projection=projection, precision=precision, include_scores=include_scores)

    image_path: Path | None = None
    if include_images:
        image_target.parent.mkdir(parents=True, exist_ok=True)
        if sample.image_bytes is not None:
            image_target.write_bytes(sample.image_bytes)
        elif source is not None and source.resolve() != image_target.resolve():
            shutil.copy2(source, image_target)
        used_images.add(image_target)
        image_path = image_target

    label_target.parent.mkdir(parents=True, exist_ok=True)
    label_target.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
    used_labels.add(label_target)
    return (image_path, label_target)


def _safe_od_image_rel(sample: ImageObjectDetectionSample, *, split: str) -> PurePosixPath:
    raw = sample.file_name or (Path(sample.path_or_uri).name if sample.path_or_uri else f"{sample.image_id}.jpg")
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
) -> tuple[Path, Path]:
    for candidate in _candidate_targets(image_target):
        rel = candidate.relative_to(image_root)
        label_candidate = label_root / rel.with_suffix(".txt")
        image_available = free_target(candidate, used_images) if require_image_free else candidate not in used_images
        if image_available and free_target(label_candidate, used_labels):
            return candidate, label_candidate
    raise ValueError(f"could not allocate unique YOLO OD target near {image_target}")


def _candidate_targets(target: Path) -> Iterable[Path]:
    yield target
    stem = target.stem
    suffix = target.suffix
    for index in range(2, 1_000_000):
        yield target.with_name(f"{stem}_{index}{suffix}")


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


def _od_data_yaml(*, splits: tuple[str, ...], names: list[str]) -> str:
    lines = ["# Generated by datamaite", f"path: {json.dumps('.')}"]
    if splits:
        if "train" in splits:
            lines.append("train: images/train")
        if "val" in splits:
            lines.append("val: images/val")
        if "test" in splits:
            lines.append("test: images/test")
    lines.append(f"names: {json.dumps(names)}")
    return "\n".join(lines) + "\n"
