"""COCO object-detection dataset writer.

Serialises an :class:`~datamaite.object_detection.ObjectDetectionDataset` as a
COCO detection root: ``annotations/instances.json`` (the ``info`` / ``licenses``
/ ``categories`` / ``images`` / ``annotations`` arrays) plus the image files
copied under ``dest`` at their ``file_name`` paths. The canonical bbox is
already COCO's ``[x, y, width, height]`` in absolute pixels, so boxes pass
through without conversion -- the exact inverse of the loader.

Best-effort by the writer contract: data the format cannot represent (a
non-integer image or category id, a detection ``score`` -- ground-truth COCO
JSON has no score field) is dropped and logged at WARNING; destination/IO
failures raise. The output is reloadable by the COCO loader, which is pinned by
the load -> write -> load round-trip test.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar

from datamaite._types import DatasetFormat, Task
from datamaite.object_detection import ObjectDetectionDataset
from datamaite.records import ImageObjectDetectionSample, ObjectDetectionAnnotation
from datamaite.taxonomy import Taxonomy
from datamaite.writers import Writer, WriterCapabilities, register_writer

logger = logging.getLogger(__name__)

# Keys the writer emits as typed fields; the attributes round-trip channel must
# not shadow them. Mirrors the loader's _ANNOTATION_CORE_KEYS.
_ANNOTATION_CORE_KEYS = frozenset({"id", "image_id", "category_id", "bbox", "area", "iscrowd", "segmentation"})


@dataclass
class _IdAllocator:
    """Allocate deterministic integer IDs while preserving valid preferred IDs."""

    next_id: int = 1
    used: set[int] = field(default_factory=set)

    def reserve(self, preferred: int | None = None) -> int:
        if preferred is not None and preferred >= 0 and preferred not in self.used:
            self.used.add(preferred)
            if preferred >= self.next_id:
                self.next_id = preferred + 1
            return preferred
        while self.next_id in self.used:
            self.next_id += 1
        value = self.next_id
        self.used.add(value)
        self.next_id += 1
        return value


@register_writer
class CocoWriter(Writer[ObjectDetectionDataset]):
    """Write an :class:`ObjectDetectionDataset` as a COCO detection root."""

    format = DatasetFormat.COCO
    task: ClassVar[Task] = Task.OD
    consumes: ClassVar[type] = ObjectDetectionDataset
    capabilities: ClassVar[WriterCapabilities] = WriterCapabilities(
        lossy_without={"score": "ground-truth COCO JSON has no score field"},
    )

    def write(
        self,
        dataset: ObjectDetectionDataset,
        dest: str | Path,
        *,
        annotation_file_name: str = "instances.json",
        include_images: bool = True,
        **_options: Any,
    ) -> list[Path]:
        """Serialise ``dataset`` under ``dest`` as COCO and return the files written.

        Parameters
        ----------
        annotation_file_name
            Bare file name for the JSON under ``dest/annotations/``. Keep an
            ``instances*.json`` name so the loader's discovery prefers it.
        include_images
            When True (default), copy each sample's image (from
            ``image_bytes`` or ``path_or_uri``) to ``dest/<file_name>``.
            Samples whose image source is missing keep their JSON entry --
            COCO JSON references files by name and stays loadable without
            them -- with a warning per missing source.
        """
        if Path(annotation_file_name).name != annotation_file_name or not annotation_file_name:
            raise ValueError(f"annotation_file_name must be a bare file name, got {annotation_file_name!r}")
        dest = Path(dest)
        ann_dir = dest / "annotations"
        ann_dir.mkdir(parents=True, exist_ok=True)

        written: list[Path] = []
        meta = dataset.dataset_metadata
        images: list[dict[str, Any]] = []
        annotations: list[dict[str, Any]] = []
        annotation_ids = _IdAllocator()
        seen_image_ids: set[int] = set()
        dropped_scores = 0

        for sample in dataset.samples:
            image_obj = _image_object(sample, seen=seen_image_ids)
            if image_obj is None:
                continue
            images.append(image_obj)
            image_id = image_obj["id"]
            for detection in sample.detections:
                row = _annotation_object(detection, image_id=image_id, ids=annotation_ids)
                if row is None:
                    continue
                if detection.score is not None:
                    dropped_scores += 1
                annotations.append(row)
            if include_images:
                copied = _copy_image(sample, dest=dest)
                if copied is not None:
                    written.append(copied)

        if dropped_scores:
            logger.warning(
                "Dropped the score field from %d annotation(s): %s",
                dropped_scores,
                self.capabilities.lossy_without["score"],
            )

        document = {
            "info": dict(meta.info),
            "licenses": [dict(license_obj) for license_obj in meta.licenses],
            "categories": _category_objects(meta.taxonomy),
            "images": images,
            "annotations": annotations,
        }
        ann_path = ann_dir / annotation_file_name
        ann_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
        written.insert(0, ann_path)
        return written


def _category_objects(taxonomy: Taxonomy | None) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for entry in taxonomy.entries if taxonomy is not None else ():
        if isinstance(entry.source_id, bool) or not isinstance(entry.source_id, int):
            logger.warning(
                "Skipping taxonomy entry %r: COCO category ids are integers, got %r",
                entry.name,
                entry.source_id,
            )
            continue
        obj: dict[str, Any] = {"id": entry.source_id, "name": entry.name}
        if entry.supercategory is not None:
            obj["supercategory"] = entry.supercategory
        obj.update({k: v for k, v in entry.attributes.items() if k not in obj})
        objects.append(obj)
    return objects


def _image_object(sample: ImageObjectDetectionSample, *, seen: set[int]) -> dict[str, Any] | None:
    if isinstance(sample.image_id, bool) or not isinstance(sample.image_id, int):
        logger.warning(
            "Skipping sample %r (and its detections): COCO image ids are integers",
            sample.image_id,
        )
        return None
    if sample.image_id in seen:
        logger.warning("Skipping duplicate image id %s (keeping the first)", sample.image_id)
        return None
    file_name = sample.file_name or (PurePosixPath(sample.path_or_uri).name if sample.path_or_uri else None)
    if not file_name:
        logger.warning("Skipping sample %s (and its detections): COCO images require a file_name", sample.image_id)
        return None
    if not _is_safe_file_name(file_name):
        # Don't emit an images[].file_name we would refuse to copy: keep the JSON
        # and the on-disk images consistent.
        logger.warning("Skipping sample %s (and its detections): unsafe file_name %r", sample.image_id, file_name)
        return None
    seen.add(sample.image_id)
    obj: dict[str, Any] = {"id": sample.image_id, "file_name": file_name}
    if sample.width is not None:
        obj["width"] = sample.width
    if sample.height is not None:
        obj["height"] = sample.height
    # Per-image passthrough (license, capture metadata, unparsed width/height
    # source values, ...) -- typed fields above win on collision.
    obj.update({k: v for k, v in sample.metadata.items() if k not in obj})
    return obj


def _annotation_object(
    detection: ObjectDetectionAnnotation,
    *,
    image_id: int,
    ids: _IdAllocator,
) -> dict[str, Any] | None:
    category_id = _int_id(detection.category_id)
    if category_id is None:
        category_id = _int_id(detection.source_category_id)
    if category_id is None:
        logger.warning(
            "Dropping detection %r on image %s: COCO annotations require an integer category_id",
            detection.source_annotation_id,
            image_id,
        )
        return None
    bbox = [float(value) for value in detection.bbox]
    if not all(math.isfinite(value) for value in bbox):
        # A non-finite bbox would serialise to invalid JSON (NaN/Infinity).
        logger.warning(
            "Dropping detection %r on image %s: non-finite bbox %r",
            detection.source_annotation_id,
            image_id,
            bbox,
        )
        return None
    if len(bbox) >= 4 and (bbox[2] <= 0 or bbox[3] <= 0):
        # Non-positive width/height; the loader drops these, so don't emit them.
        logger.warning(
            "Dropping detection %r on image %s: non-positive bbox dimensions %r",
            detection.source_annotation_id,
            image_id,
            bbox,
        )
        return None
    obj: dict[str, Any] = {
        "id": ids.reserve(_int_id(detection.source_annotation_id)),
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,
        "iscrowd": detection.iscrowd,
    }
    if detection.area is not None:
        # Source area verbatim, and only when the source carried one --
        # fabricating w*h would break faithful re-emit and round-trips. Drop a
        # non-finite area rather than emit invalid JSON.
        if isinstance(detection.area, float) and not math.isfinite(detection.area):
            logger.warning(
                "Omitting non-finite area on detection %r (image %s)",
                detection.source_annotation_id,
                image_id,
            )
        else:
            obj["area"] = detection.area
    if detection.segmentation is not None:
        obj["segmentation"] = detection.segmentation
    obj.update({k: v for k, v in detection.attributes.items() if k not in _ANNOTATION_CORE_KEYS})
    return obj


def _int_id(value: int | str | None) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _copy_image(sample: ImageObjectDetectionSample, *, dest: Path) -> Path | None:
    file_name = sample.file_name or (PurePosixPath(sample.path_or_uri).name if sample.path_or_uri else "")
    target = _safe_target(dest, file_name)
    if target is None:
        logger.warning("Not writing image for sample %s: unsafe file_name %r", sample.image_id, file_name)
        return None
    if sample.image_bytes is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(sample.image_bytes)
        return target
    if sample.path_or_uri is not None:
        source = Path(sample.path_or_uri)
        if source.is_file():
            if source.resolve() == target.resolve():
                return None  # writing a dataset over itself; the image is already in place
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            return target
        # The sample names a source; it just isn't on disk -- say so, rather than
        # the misleading "no image source" below.
        logger.warning(
            "Image source for sample %s does not exist: %r; its JSON entry is kept but no file was written",
            sample.image_id,
            sample.path_or_uri,
        )
        return None
    logger.warning(
        "No image source for sample %s (%r); its JSON entry is kept but no file was written",
        sample.image_id,
        file_name,
    )
    return None


def _is_safe_file_name(file_name: str) -> bool:
    """Whether ``file_name`` is a safe relative path (no escapes); dest-independent.

    Shared by ``_image_object`` (so the emitted ``images[].file_name`` is one we
    would actually copy) and ``_safe_target`` (the copy destination), so the JSON
    and the on-disk images can never disagree.
    """
    if "\\" in file_name:
        return False
    posix = PurePosixPath(file_name.strip())
    return (
        bool(posix.parts)
        and not posix.is_absolute()
        and not any(part in {"..", ""} or ":" in part for part in posix.parts)
    )


def _safe_target(dest: Path, file_name: str) -> Path | None:
    """Resolve ``file_name`` under ``dest``, refusing path escapes (mirrors the loader)."""
    if not _is_safe_file_name(file_name):
        return None
    posix = PurePosixPath(file_name.strip())
    return dest.joinpath(*posix.parts)
