"""Writer for the flat MP4 video format."""

from __future__ import annotations

import logging
from pathlib import Path, PurePosixPath
from typing import Any

from datamaite._io import copy_resource, same_resource, source_path
from datamaite._types import DatasetFormat
from datamaite._upath import to_dataset_path
from datamaite.model import BoxTrackDataset
from datamaite.writers import Writer, WriterCapabilities, register_writer

logger = logging.getLogger(__name__)


@register_writer
class FlatMp4Writer(Writer[BoxTrackDataset]):
    """Copy MP4-backed MOT sequences into one flat directory."""

    format = DatasetFormat.FLAT_MP4
    capabilities = WriterCapabilities(
        required_fields=frozenset({"video"}),
        lossy_without={"annotations": "flat_mp4 carries video media only"},
    )

    def write(self, dataset: BoxTrackDataset, dest: str | Path, **options: Any) -> list[Path]:
        destination = to_dataset_path(dest, options.get("storage_options"))
        destination.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        used: set[str] = set()
        dropped = 0
        for sequence in dataset.sequences:
            dropped += len(sequence.boxes)
            if not sequence.video_path:
                logger.warning("Skipping frame-backed sequence %s: flat_mp4 requires an MP4 source", sequence.video_id)
                continue
            source = source_path(sequence.video_path)
            if not source.is_file() or source.suffix.lower() != ".mp4":
                logger.warning("Skipping non-MP4 or missing video source: %s", source)
                continue
            raw = PurePosixPath(sequence.video_path).name or f"video_{sequence.video_id:06d}.mp4"
            name = _unique_name(raw, used)
            target = destination / name
            if not same_resource(source, target):
                copy_resource(source, target)
            used.add(name)
            written.append(target)
        if dropped:
            logger.warning("Dropped %d box annotation(s): flat_mp4 carries video media only", dropped)
        return written


def _unique_name(raw: str, used: set[str]) -> str:
    path = PurePosixPath(raw)
    name = path.name if path.suffix.lower() == ".mp4" else f"{path.stem or 'video'}.mp4"
    if name not in used:
        return name
    for index in range(2, 1_000_000):
        candidate = f"{PurePosixPath(name).stem}_{index}.mp4"
        if candidate not in used:
            return candidate
    raise ValueError("could not allocate a unique flat MP4 filename")
