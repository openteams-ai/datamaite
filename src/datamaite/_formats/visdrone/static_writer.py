"""VisDrone Static-Images writers (IR-3.2-S-7): object detection + image classification.

The output-side mirror of :mod:`datamaite._formats.visdrone.static_loader`:
one writer is registered for object detection (``Task.OD``) and one for image
classification (``Task.IC``) under the shared ``DatasetFormat.VISDRONE``
family, so the detection/classification selection required by the standard is
the writer registry's task axis (``write(dataset, output_format="visdrone")``
dispatches on ``dataset.task``; ``get_writer(..., task=...)`` selects
explicitly).

Both emit the official VisDrone-DET split-root layout the static loader
reads back::

    <dest>/
        VisDrone2019-DET-train/
            images/<image>.jpg
            annotations/<image>.txt

Annotation rows are the official eight comma-separated fields
``left,top,width,height,score,category,truncation,occlusion``. The OD writer
emits one image plus one annotation file per sample; the IC writer emits one
image per distinct source that keeps at least one annotation row and one row
per classification sample (the object crop), which is exactly the projection
the IC loader derives — an image whose rows all drop is not copied, so the
emitted root never contains images the loaders would reject or skip.

Everything written must reload: images are copied verbatim (never
transcoded), so only suffixes the static loader reads
(:data:`~datamaite._formats.visdrone.static_loader.IMAGE_EXTENSIONS`) are
emitted; samples with any other suffix are skipped with a warning.

Class ids are resolved with the shared fixed-taxonomy machinery (#55):
explicit ``class_map`` > the ``visdrone_category_id`` attribute the VisDrone
loaders preserve > the generic ``category_id`` fallback with one aggregated
warning per write. A category 0 produced by the generic fallback is *not*
written: VisDrone category 0 is the "ignored regions" pseudo-class (score 0,
excluded from evaluation), and silently reinterpreting an unrelated source
category 0 (e.g. COCO-style dense ids where 0 is a real class) as an ignored
region would corrupt the output (#55 B3).
"""

from __future__ import annotations

import logging
import shutil
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from datamaite._formats._coerce import coerce_finite_float, coerce_int
from datamaite._formats._fixed_taxonomy import ClassIdResolver, validate_class_map
from datamaite._formats.visdrone.static_loader import IMAGE_EXTENSIONS, VISDRONE_STATIC_CLASSES
from datamaite._types import DatasetFormat, Task
from datamaite.geometry import BBox, has_positive_area
from datamaite.image_classification import ImageClassificationDataset
from datamaite.object_detection import ObjectDetectionDataset
from datamaite.records import (
    ClassificationLabel,
    ImageClassificationSample,
    ImageObjectDetectionSample,
    ObjectDetectionAnnotation,
)
from datamaite.writers import Writer, WriterCapabilities, register_writer

logger = logging.getLogger(__name__)

_SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "validation": "val",
    "valid": "val",
    "val": "val",
    "test-dev": "test-dev",
    "test_dev": "test-dev",
    "testdev": "test-dev",
    "test-challenge": "test-challenge",
    "test_challenge": "test-challenge",
    "testchallenge": "test-challenge",
    "test": "test",
}
_MAX_CATEGORY_ID = len(VISDRONE_STATIC_CLASSES) - 1  # 0..11
_IGNORED_REGIONS_ID = 0


@dataclass(frozen=True)
class _ResolvableBox:
    """Duck-typed adapter so :class:`ClassIdResolver` can resolve still-image records."""

    category_id: int
    category_name: str | None
    attributes: Mapping[str, Any]


@dataclass
class _SplitRootState:
    """Per-split-root bookkeeping: unique image stems and pending annotation rows."""

    used_stems: set[str] = field(default_factory=set)
    rows_by_stem: dict[str, list[list[float | int]]] = field(default_factory=dict)


@register_writer
class VisDroneObjectDetectionWriter(Writer[ObjectDetectionDataset]):
    """Write an OD dataset as official VisDrone-DET split roots."""

    task: ClassVar[Task] = Task.OD
    format = DatasetFormat.VISDRONE
    variant: ClassVar[str] = "default"
    consumes: ClassVar[type] = ObjectDetectionDataset
    capabilities: ClassVar[WriterCapabilities] = WriterCapabilities(
        required_fields=frozenset({"image"}),
        lossy_without={
            "score": "the VisDrone GT score column is an evaluation flag, not a confidence",
            "segmentation": "VisDrone-DET rows store boxes/categories only",
            "iscrowd": "VisDrone-DET rows store boxes/categories only",
        },
        forbids_dense_remap=True,
    )

    def validate_options(self, **options: Any) -> None:
        """Validate options that can raise, before write()'s destination policy runs (#55 Fix A1)."""
        if "split" in options:
            _normalize_split(options["split"], field="split")
        validate_class_map(options.get("class_map"), minimum=0, format_label="VisDrone")

    def write(
        self,
        dataset: ObjectDetectionDataset,
        dest: str | Path,
        *,
        split: str = "train",
        preserve_splits: bool = True,
        class_map: Mapping[str | int, int] | None = None,
        **_: Any,
    ) -> list[Path]:
        """Serialise an OD dataset under ``dest`` and return the files written.

        Parameters
        ----------
        split
            Fallback split name for samples without split metadata. Common
            aliases such as ``"validation"`` normalise to VisDrone's ``"val"``.
        preserve_splits
            When True (default), a sample with a known ``split`` is written to
            that split root; otherwise the fallback ``split`` is used.
        class_map
            Optional explicit mapping from source categories to VisDrone
            category ids (0..11). Keys are ``category_name`` strings (matched
            first) or ``category_id`` ints; values must be ``>= 0``. When
            provided it overrides both the ``visdrone_category_id`` attribute
            and the generic ``category_id`` fallback; unmapped boxes are
            dropped and reported in one aggregated warning.

        Notes
        -----
        Every written image receives an annotation file (empty for samples
        with no detections) so the emitted root loads back with the same
        sample count. Images are copied verbatim (never transcoded), so a
        sample whose image suffix the VisDrone static loader cannot read is
        skipped with a warning. Detections with a non-positive box, an
        out-of-range VisDrone category, or a generic ``category_id`` 0 that
        would be reinterpreted as the "ignored regions" pseudo-class are
        dropped with warnings.
        """
        fallback_split = _normalize_split(split, field="split")
        resolver = _resolver(class_map)
        dest_path = Path(dest)
        dest_path.mkdir(parents=True, exist_ok=True)

        written: list[Path] = []
        state: dict[str, _SplitRootState] = {}
        dropped = _DropCounters()
        try:
            for sample in dataset.samples:
                sample_split = _split_for_sample(
                    sample.split, sample.image_id, fallback=fallback_split, preserve_splits=preserve_splits
                )
                root_state = state.setdefault(sample_split, _SplitRootState())
                stem = _write_image(
                    sample,
                    split_root=dest_path / _split_root_name(sample_split),
                    state=root_state,
                    written=written,
                )
                if stem is None:
                    continue
                rows = root_state.rows_by_stem.setdefault(stem, [])
                for detection in sample.detections:
                    row = _od_row(detection, resolver=resolver, dropped=dropped)
                    if row is not None:
                        rows.append(row)
        finally:
            # Aggregated warnings must surface even if a later sample raises
            # mid-write (#55 B2): earlier output is already on disk.
            resolver.emit_warnings()
            dropped.emit(logger)
        written.extend(_write_annotations(dest_path, state))
        if not written:
            logger.warning("No VisDrone static images were written to %s", dest_path)
        return written


@register_writer
class VisDroneImageClassificationWriter(Writer[ImageClassificationDataset]):
    """Write an IC dataset (object crops) as official VisDrone-DET split roots."""

    task: ClassVar[Task] = Task.IC
    format = DatasetFormat.VISDRONE
    variant: ClassVar[str] = "default"
    consumes: ClassVar[type] = ImageClassificationDataset
    capabilities: ClassVar[WriterCapabilities] = WriterCapabilities(
        required_fields=frozenset({"image", "labels"}),
        forbids_dense_remap=True,
    )

    def validate_options(self, **options: Any) -> None:
        """Validate options that can raise, before write()'s destination policy runs (#55 Fix A1)."""
        if "split" in options:
            _normalize_split(options["split"], field="split")
        validate_class_map(options.get("class_map"), minimum=0, format_label="VisDrone")

    def write(
        self,
        dataset: ImageClassificationDataset,
        dest: str | Path,
        *,
        split: str = "train",
        preserve_splits: bool = True,
        class_map: Mapping[str | int, int] | None = None,
        **_: Any,
    ) -> list[Path]:
        """Serialise an IC dataset under ``dest`` and return the files written.

        The IC projection of the VisDrone-DET layout: each sample contributes
        one annotation row (its crop ``region``, or a full-image box when the
        sample has no region but known ``width``/``height``), and each
        distinct source image is copied once. Samples whose box cannot be
        determined, or whose label cannot be resolved to a VisDrone category,
        are skipped with warnings. A source image is only copied once at
        least one of its samples yields an annotation row, so every written
        image has an annotation file and the emitted root reloads cleanly.
        Options match the OD writer.
        """
        fallback_split = _normalize_split(split, field="split")
        resolver = _resolver(class_map)
        dest_path = Path(dest)
        dest_path.mkdir(parents=True, exist_ok=True)

        written: list[Path] = []
        state: dict[str, _SplitRootState] = {}
        stems_by_source: dict[tuple[str, str], str | None] = {}
        dropped = _DropCounters()
        try:
            for sample in dataset.samples:
                bbox = _ic_bbox(sample)
                if bbox is None:
                    logger.warning(
                        "Skipping VisDrone IC sample %r without a crop region or image dimensions",
                        sample.image_id,
                    )
                    continue
                label = _single_label(sample)
                if label is None:
                    logger.warning("Skipping VisDrone IC sample %r with no labels", sample.image_id)
                    continue
                # Resolve the row *before* copying the image: an image whose
                # rows all drop must not be emitted at all, or the root would
                # contain images without annotation files, which the loaders
                # reject (sniff) or silently skip (IC load).
                row = _ic_row(label, bbox, resolver=resolver, dropped=dropped)
                if row is None:
                    continue
                sample_split = _split_for_sample(
                    sample.split, sample.image_id, fallback=fallback_split, preserve_splits=preserve_splits
                )
                root_state = state.setdefault(sample_split, _SplitRootState())
                stem = _ic_image_stem(
                    sample,
                    sample_split=sample_split,
                    split_root=dest_path / _split_root_name(sample_split),
                    state=root_state,
                    stems_by_source=stems_by_source,
                    written=written,
                )
                if stem is None:
                    continue
                root_state.rows_by_stem.setdefault(stem, []).append(row)
        finally:
            resolver.emit_warnings()
            dropped.emit(logger)
        written.extend(_write_annotations(dest_path, state))
        if not written:
            logger.warning("No VisDrone static images were written to %s", dest_path)
        return written


# ---------------------------------------------------------------------------
# Class resolution and row building
# ---------------------------------------------------------------------------


def _resolver(class_map: Mapping[str | int, int] | None) -> ClassIdResolver:
    return ClassIdResolver(
        format_label="VisDrone",
        attribute="visdrone_category_id",
        class_map=validate_class_map(class_map, minimum=0, format_label="VisDrone"),
        logger=logger,
        minimum=0,
    )


@dataclass
class _DropCounters:
    """Per-write tallies for conditions warned once, aggregated (#55 style)."""

    out_of_range: Counter[str] = field(default_factory=Counter)
    fallback_ignored: Counter[str] = field(default_factory=Counter)
    bad_bbox: int = 0
    dropped_scores: int = 0
    dropped_lossy: int = 0

    def emit(self, log: logging.Logger) -> None:
        if self.bad_bbox:
            log.warning("VisDrone static writer: dropped %d annotation(s) with a non-positive box", self.bad_bbox)
        if self.fallback_ignored:
            log.warning(
                "VisDrone static writer: dropped %d annotation(s) whose generic category_id 0 would have "
                "been reinterpreted as VisDrone category 0 ('ignored regions', excluded from evaluation); "
                "pass class_map= to map source categories onto VisDrone ids explicitly: %s",
                sum(self.fallback_ignored.values()),
                ", ".join(f"{label}={count}" for label, count in sorted(self.fallback_ignored.items())),
            )
        if self.out_of_range:
            log.warning(
                "VisDrone static writer: dropped %d annotation(s) whose resolved class id is outside "
                "the VisDrone 0..%d table: %s",
                sum(self.out_of_range.values()),
                _MAX_CATEGORY_ID,
                ", ".join(f"{label}={count}" for label, count in sorted(self.out_of_range.items())),
            )
        if self.dropped_scores:
            log.warning(
                "VisDrone static writer: dropped confidence from %d annotation(s): the GT score column "
                "is an evaluation flag, not a confidence",
                self.dropped_scores,
            )
        if self.dropped_lossy:
            log.warning(
                "VisDrone static writer: dropped VisDrone-unrepresentable fields (segmentation/iscrowd) "
                "from %d annotation(s)",
                self.dropped_lossy,
            )


def _resolve_class_id(
    *,
    category_id: int | str | None,
    source_category_id: int | str | None,
    category_name: str | None,
    attributes: Mapping[str, Any],
    resolver: ClassIdResolver,
    dropped: _DropCounters,
) -> int | None:
    """Resolve one record's VisDrone class id, or ``None`` to drop the record."""
    preferred = source_category_id if source_category_id is not None else category_id
    numeric = preferred if isinstance(preferred, int) and not isinstance(preferred, bool) else -1
    resolved = resolver.resolve(
        _ResolvableBox(category_id=numeric, category_name=category_name, attributes=attributes)  # type: ignore[arg-type]
    )
    if resolved.class_id is None:
        return None  # unmapped under an explicit class_map; already tallied by the resolver
    if resolved.class_id == _IGNORED_REGIONS_ID and resolved.from_generic_fallback:
        # Category 0 is a genuine VisDrone "ignored region" only when it came
        # from a real source (the visdrone_category_id attribute or an explicit
        # class_map); a 0 produced by the generic category_id fallback is just
        # an unrelated source category that happens to be 0 and must not be
        # silently written as an ignored region (#55 B3).
        dropped.fallback_ignored[category_name or f"category_id={preferred!r}"] += 1
        return None
    if not 0 <= resolved.class_id <= _MAX_CATEGORY_ID:
        dropped.out_of_range[category_name or f"category_id={preferred!r}"] += 1
        return None
    return resolved.class_id


def _od_row(
    detection: ObjectDetectionAnnotation,
    *,
    resolver: ClassIdResolver,
    dropped: _DropCounters,
) -> list[float | int] | None:
    if not has_positive_area(detection.bbox):
        dropped.bad_bbox += 1
        return None
    class_id = _resolve_class_id(
        category_id=detection.category_id,
        source_category_id=detection.source_category_id,
        category_name=detection.category_name,
        attributes=detection.attributes,
        resolver=resolver,
        dropped=dropped,
    )
    if class_id is None:
        return None
    if detection.score is not None:
        dropped.dropped_scores += 1
    if detection.segmentation is not None or detection.iscrowd:
        dropped.dropped_lossy += 1
    return _row(detection.bbox, class_id=class_id, attributes=detection.attributes)


def _ic_row(
    label: ClassificationLabel,
    bbox: BBox,
    *,
    resolver: ClassIdResolver,
    dropped: _DropCounters,
) -> list[float | int] | None:
    if not has_positive_area(bbox):
        dropped.bad_bbox += 1
        return None
    class_id = _resolve_class_id(
        category_id=label.category_id,
        source_category_id=label.source_category_id,
        category_name=label.category_name,
        attributes=label.attributes,
        resolver=resolver,
        dropped=dropped,
    )
    if class_id is None:
        return None
    return _row(bbox, class_id=class_id, attributes=label.attributes)


def _row(bbox: BBox, *, class_id: int, attributes: Mapping[str, Any]) -> list[float | int]:
    """One official eight-field row: left,top,width,height,score,category,truncation,occlusion."""
    score = coerce_finite_float(attributes.get("visdrone_score"))
    if score is None:
        # GT convention: 0 flags ignored regions out of evaluation, 1 evaluates.
        score = 0.0 if class_id == _IGNORED_REGIONS_ID else 1.0
    truncation = coerce_int(attributes.get("truncation")) or 0
    occlusion = coerce_int(attributes.get("occlusion")) or 0
    return [*bbox, score, class_id, truncation, occlusion]


def _ic_bbox(sample: ImageClassificationSample) -> BBox | None:
    if sample.region is not None:
        return sample.region
    if sample.width is not None and sample.height is not None and sample.width > 0 and sample.height > 0:
        return (0.0, 0.0, float(sample.width), float(sample.height))
    return None


def _single_label(sample: ImageClassificationSample) -> ClassificationLabel | None:
    if not sample.labels:
        return None
    if len(sample.labels) > 1:
        logger.warning("VisDrone rows carry one category per box; using first label for %r", sample.image_id)
    return next(iter(sample.labels))


# ---------------------------------------------------------------------------
# Image materialisation and annotation emission
# ---------------------------------------------------------------------------


def _split_root_name(split: str) -> str:
    return f"VisDrone2019-DET-{split}"


def _normalize_split(value: object, *, field: str) -> str:
    raw = str(value).strip()
    normalized = _SPLIT_ALIASES.get(raw.lower().replace("_", "-"))
    if normalized is None:
        raise ValueError(f"{field} must be one of {sorted(set(_SPLIT_ALIASES.values()))!r}; got {value!r}")
    return normalized


def _split_for_sample(raw: str | None, sample_id: int | str, *, fallback: str, preserve_splits: bool) -> str:
    if not preserve_splits or raw is None:
        return fallback
    try:
        return _normalize_split(raw, field="sample.split")
    except ValueError:
        logger.warning(
            "Sample %r has unrecognised VisDrone split %r; writing it to fallback split %r",
            sample_id,
            raw,
            fallback,
        )
        return fallback


def _write_image(
    sample: ImageObjectDetectionSample,
    *,
    split_root: Path,
    state: _SplitRootState,
    written: list[Path],
) -> str | None:
    """Copy/write one sample's image under ``images/``; return its unique stem."""
    source: Path | None = None
    if sample.image_bytes is None:
        if sample.path_or_uri is None:
            logger.warning("Skipping VisDrone static sample %r with no image source", sample.image_id)
            return None
        source = Path(sample.path_or_uri)
        if not source.is_file():
            logger.warning("Skipping VisDrone static sample %r with missing image file: %s", sample.image_id, source)
            return None

    raw_name = sample.file_name or (source.name if source is not None else f"{sample.image_id}.jpg")
    name = Path(str(raw_name)).name
    if not name or name in {".", ".."} or "\x00" in name or "\\" in name:
        logger.warning("Skipping VisDrone static sample %r with unsafe file name %r", sample.image_id, raw_name)
        return None
    suffix = Path(name).suffix or ".jpg"
    if suffix.lower() not in IMAGE_EXTENSIONS:
        # Images are copied verbatim (no transcoding), so a suffix the static
        # loader does not read would silently vanish on reload.
        logger.warning(
            "Skipping VisDrone static sample %r: image suffix %r is not readable by the VisDrone "
            "static loader (readable: %s)",
            sample.image_id,
            suffix,
            ", ".join(sorted(IMAGE_EXTENSIONS)),
        )
        return None
    stem = _unique_stem(Path(name).stem or "image", state.used_stems)

    images_dir = split_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    target = images_dir / f"{stem}{suffix}"
    if sample.image_bytes is not None:
        target.write_bytes(sample.image_bytes)
    elif source is not None and source.resolve(strict=False) != target.resolve(strict=False):
        shutil.copy2(source, target)
    written.append(target)
    return stem


def _ic_image_stem(
    sample: ImageClassificationSample,
    *,
    sample_split: str,
    split_root: Path,
    state: _SplitRootState,
    stems_by_source: dict[tuple[str, str], str | None],
    written: list[Path],
) -> str | None:
    """Return the image stem for an IC sample, writing its source image only once per split."""
    if sample.path_or_uri is not None and sample.image_bytes is None:
        key = (sample_split, sample.path_or_uri)
        if key in stems_by_source:
            return stems_by_source[key]
        stem = _write_image(_as_od_sample(sample), split_root=split_root, state=state, written=written)
        stems_by_source[key] = stem
        return stem
    return _write_image(_as_od_sample(sample), split_root=split_root, state=state, written=written)


def _as_od_sample(sample: ImageClassificationSample) -> ImageObjectDetectionSample:
    """Reuse `_write_image` across tasks: only the shared ImageRecord fields matter."""
    return ImageObjectDetectionSample(
        image_id=sample.image_id,
        path_or_uri=sample.path_or_uri,
        image_bytes=sample.image_bytes,
        file_name=sample.file_name,
    )


def _unique_stem(stem: str, used: set[str]) -> str:
    if stem not in used:
        used.add(stem)
        return stem
    index = 1
    while f"{stem}-{index}" in used:
        index += 1
    unique = f"{stem}-{index}"
    used.add(unique)
    return unique


def _write_annotations(dest: Path, state: dict[str, _SplitRootState]) -> list[Path]:
    """Emit one annotation file per written image (empty when it has no rows)."""
    files: list[Path] = []
    for split, root_state in sorted(state.items()):
        annotations_dir = dest / _split_root_name(split) / "annotations"
        for stem, rows in sorted(root_state.rows_by_stem.items()):
            annotations_dir.mkdir(parents=True, exist_ok=True)
            path = annotations_dir / f"{stem}.txt"
            path.write_text(
                "".join(",".join(_format_field(value) for value in row) + "\n" for row in rows),
                encoding="utf-8",
            )
            files.append(path)
    return files


def _format_field(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:.12g}"
