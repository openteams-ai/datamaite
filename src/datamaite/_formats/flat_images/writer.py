"""Writer for the flat, label-free still-image format."""

from __future__ import annotations

import logging
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar

from datamaite._io import copy_resource, same_resource, source_path
from datamaite._types import DatasetFormat, Task
from datamaite._upath import to_dataset_path
from datamaite.object_detection import ObjectDetectionDataset
from datamaite.writers import Writer, WriterCapabilities, register_writer

logger = logging.getLogger(__name__)


@register_writer
class FlatImagesWriter(Writer[ObjectDetectionDataset]):
    """Write OD image media into one flat directory, dropping annotations."""

    format = DatasetFormat.FLAT_IMAGES
    task: ClassVar[Task] = Task.OD
    consumes: ClassVar[type] = ObjectDetectionDataset
    capabilities: ClassVar[WriterCapabilities] = WriterCapabilities(
        required_fields=frozenset({"image"}),
        lossy_without={"detections": "flat_images carries image media only"},
    )

    def write(self, dataset: ObjectDetectionDataset, dest: str | Path, **options: Any) -> list[Path]:
        destination = to_dataset_path(dest, options.get("storage_options"))
        destination.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        used: set[str] = set()
        dropped = 0
        for sample in dataset.samples:
            if sample.detections:
                dropped += len(sample.detections)
            raw = sample.file_name or (
                PurePosixPath(sample.path_or_uri).name if sample.path_or_uri else f"{sample.image_id}.jpg"
            )
            name = _safe_unique_name(raw, used)
            if name is None:
                logger.warning("Skipping flat image sample %r with unsafe file name %r", sample.image_id, raw)
                continue
            target = destination / name
            if sample.image_bytes is not None:
                target.write_bytes(sample.image_bytes)
            elif sample.path_or_uri is not None:
                source = source_path(sample.path_or_uri)
                if not source.is_file():
                    logger.warning("Skipping missing flat image source: %s", source)
                    continue
                if not same_resource(source, target):
                    copy_resource(source, target)
            else:
                logger.warning("Skipping flat image sample %r with no image source", sample.image_id)
                continue
            used.add(name)
            written.append(target)
        if dropped:
            logger.warning("Dropped %d detection(s): flat_images carries image media only", dropped)
        return written


def _safe_unique_name(raw: str, used: set[str]) -> str | None:
    name = PurePosixPath(str(raw).replace("\\", "/")).name
    if not name or name in {".", ".."} or "\x00" in name:
        return None
    if name not in used:
        return name
    path = PurePosixPath(name)
    for index in range(2, 1_000_000):
        candidate = f"{path.stem}_{index}{path.suffix}"
        if candidate not in used:
            return candidate
    return None
