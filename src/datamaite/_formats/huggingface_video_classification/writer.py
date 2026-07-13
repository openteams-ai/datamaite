"""Hugging Face VideoFolder video classification writer.

The writer emits a local Hugging Face video classification repository using a
metadata file plus copied video files::

    <dest>/
        metadata.csv | metadata.jsonl
        train/<video>.mp4
        validation/<video>.mp4
        test/<video>.mp4

Video classification labels are video-level metadata rather than per-frame
boxes. The writer consumes :class:`~datamaite.model.VideoClassificationDataset`
records from the matching loader, copies each source video, and records
``file_name`` plus ``label`` in metadata so the loader can recover the dataset.
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
from datamaite.model import VideoClassificationDataset, VideoClassificationSample, VisionDataset
from datamaite.writers import Writer, register_writer

logger = logging.getLogger(__name__)

_VIDEO_EXTENSIONS = frozenset({".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"})
_METADATA_FORMATS = frozenset({"csv", "jsonl"})
_SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "validation": "validation",
    "valid": "validation",
    "val": "validation",
    "test": "test",
}
_RESERVED_COLUMNS = frozenset({"file_name", "label"})


@register_writer
class HuggingFaceVideoClassificationWriter(Writer[VideoClassificationDataset]):
    """Write a :class:`VideoClassificationDataset` as a Hugging Face VideoFolder repo."""

    format = DatasetFormat.HUGGINGFACE_VIDEO_CLASSIFICATION
    task: ClassVar[Task] = Task.VC
    consumes: ClassVar[type] = VideoClassificationDataset

    def validate_options(self, **options: Any) -> None:
        """Validate options that can raise, before write()'s destination policy runs (#55 Fix A1).

        Mirrors the inline ``split`` / ``metadata_format`` checks in
        ``write()``, but only for options that are present, so a
        ``mode="replace"`` clear never happens ahead of an option error.
        ``write()`` re-validates inline, which also covers direct
        ``Writer.write()`` calls.
        """
        if "split" in options:
            _normalize_optional_split(options["split"])
        if "metadata_format" in options:
            _validate_metadata_format(options["metadata_format"])

    def write(
        self,
        dataset: VisionDataset,
        dest: str | Path,
        *,
        split: str | None = None,
        preserve_splits: bool = True,
        metadata_format: str = "csv",
        **_options: Any,
    ) -> list[Path]:
        """Serialise ``dataset`` under ``dest`` and return files written.

        Parameters
        ----------
        split
            Optional fallback split for samples without split metadata. Use
            ``None`` (default) to write unsplit videos under ``data/``.
            Common aliases such as ``"val"`` normalise to ``"validation"``.
        preserve_splits
            When True (default), a sample with ``split`` set to a known Hugging
            Face split is written under that split; otherwise the fallback
            ``split`` is used.
        metadata_format
            ``"csv"`` (default) writes ``metadata.csv``; ``"jsonl"`` writes
            ``metadata.jsonl`` and can preserve JSON-safe per-sample metadata
            values more faithfully.

        Notes
        -----
        Samples without a source video path or whose video file is missing are
        skipped with a warning. The format has no per-frame boxes; the input is
        therefore the source-record-only video-classification dataset, not a
        box-track dataset with empty boxes.
        """
        if not isinstance(dataset, VideoClassificationDataset):
            raise TypeError(
                "HuggingFaceVideoClassificationWriter requires a VideoClassificationDataset, "
                f"got {type(dataset).__name__}"
            )

        fallback_split = _normalize_optional_split(split)
        metadata_format = _validate_metadata_format(metadata_format)
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, Any]] = []
        written: list[Path] = []
        written_seen: set[Path] = set()
        used_paths: set[str] = set()

        for sample in dataset.samples:
            record = _record_for_sample(
                sample,
                dest=dest,
                fallback_split=fallback_split,
                preserve_splits=preserve_splits,
                used_paths=used_paths,
            )
            if record is None:
                continue
            source, rel_path, label = record
            out_path = dest.joinpath(*PurePosixPath(rel_path).parts)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            _copy_video(source, out_path)
            _append_written(written, written_seen, out_path)
            rows.append(_metadata_row(sample, file_name=rel_path, label=label, metadata_format=metadata_format))

        metadata_path = dest / f"metadata.{metadata_format}"
        _write_metadata(metadata_path, rows, metadata_format=metadata_format)
        _append_written(written, written_seen, metadata_path)
        if not rows:
            logger.warning("No Hugging Face video classification files were written to %s", dest)
        return written


def _record_for_sample(
    sample: VideoClassificationSample,
    *,
    dest: Path,
    fallback_split: str | None,
    preserve_splits: bool,
    used_paths: set[str],
) -> tuple[Path, str, str | None] | None:
    if not sample.video_path:
        logger.warning("Skipping Hugging Face video classification sample %s with no source video", sample.video_id)
        return None
    source = Path(sample.video_path)
    if not source.is_file():
        logger.warning("Skipping Hugging Face video classification sample with missing video: %s", source)
        return None

    split = _split_for_sample(sample, fallback=fallback_split, preserve_splits=preserve_splits)
    label = _coerce_label(sample.label)
    if sample.label is not None and label is None:
        logger.warning("Ignoring non-scalar Hugging Face label for sample %s: %r", sample.video_id, sample.label)
    suffix = source.suffix.lower()
    if suffix not in _VIDEO_EXTENSIONS:
        logger.warning(
            "Video %s has extension %r outside the common Hugging Face video set; writing it with .mp4 suffix",
            source,
            source.suffix,
        )
        suffix = ".mp4"

    stem = _safe_name(_source_stem(sample, source))
    prefix = split or "data"
    rel_path = _unique_rel_path(prefix, stem, suffix, used_paths)
    if (dest / rel_path).resolve(strict=False) == source.resolve(strict=False):
        rel_path = _unique_rel_path(prefix, f"{stem}-copy", suffix, used_paths)
    return source, rel_path, label


def _source_stem(sample: VideoClassificationSample, source: Path) -> str:
    if sample.file_name and "\\" not in sample.file_name:
        posix = PurePosixPath(sample.file_name.strip())
        if posix.parts and not posix.is_absolute() and all(part not in {"", ".."} for part in posix.parts):
            return posix.stem
    return source.stem or f"video_{sample.video_id:06d}"


def _metadata_row(
    sample: VideoClassificationSample,
    *,
    file_name: str,
    label: str | None,
    metadata_format: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {}
    if metadata_format == "jsonl":
        row.update(_json_object(sample.metadata))
    else:
        row.update(_csv_metadata(sample.metadata))
    row["file_name"] = file_name
    row["label"] = label or ""
    return row


def _write_metadata(path: Path, rows: list[dict[str, Any]], *, metadata_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if metadata_format == "jsonl":
        text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        path.write_text(text, encoding="utf-8")
        return

    fieldnames = _csv_fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _csv_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    extras = sorted({key for row in rows for key in row if key not in _RESERVED_COLUMNS})
    return ["file_name", "label", *extras]


def _copy_video(source: Path, dest: Path) -> None:
    try:
        same_file = source.resolve(strict=False) == dest.resolve(strict=False)
    except OSError:
        same_file = False
    if same_file:
        return
    shutil.copy2(source, dest)


def _append_written(written: list[Path], seen: set[Path], path: Path) -> None:
    if path not in seen:
        seen.add(path)
        written.append(path)


def _split_for_sample(sample: VideoClassificationSample, *, fallback: str | None, preserve_splits: bool) -> str | None:
    if not preserve_splits:
        return fallback
    raw = sample.split
    if raw is None:
        return fallback
    try:
        return _normalize_split(str(raw), field="sample.split")
    except ValueError:
        logger.warning(
            "Sample %s has unsafe Hugging Face split %r; writing it to fallback split %r",
            sample.video_id,
            raw,
            fallback,
        )
        return fallback


def _normalize_optional_split(value: str | None) -> str | None:
    if value is None:
        return None
    return _normalize_split(value, field="split")


def _normalize_split(value: str, *, field: str) -> str:
    raw = str(value).strip()
    normalized = _SPLIT_ALIASES.get(raw.lower().replace("_", "-"), raw)
    if _safe_name(normalized) != normalized:
        raise ValueError(f"{field} must be a safe Hugging Face split name; got {value!r}")
    return normalized


def _validate_metadata_format(value: str) -> str:
    metadata_format = str(value).strip().lower()
    if metadata_format not in _METADATA_FORMATS:
        raise ValueError(f"metadata_format must be one of {sorted(_METADATA_FORMATS)!r}; got {value!r}")
    return metadata_format


def _coerce_label(value: object) -> str | None:
    if _is_missing(value) or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return str(int(value)) if value.is_integer() else str(value)
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _is_missing(value: object) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _unique_rel_path(prefix: str, stem: str, suffix: str, used: set[str]) -> str:
    prefix_parts = _safe_prefix_parts(prefix)
    filename = f"{stem}{suffix}"
    rel_path = PurePosixPath(*prefix_parts, filename).as_posix()
    if rel_path not in used:
        used.add(rel_path)
        return rel_path
    index = 1
    while True:
        rel_path = PurePosixPath(*prefix_parts, f"{stem}-{index}{suffix}").as_posix()
        if rel_path not in used:
            used.add(rel_path)
            return rel_path
        index += 1


def _safe_prefix_parts(prefix: str) -> tuple[str, ...]:
    parts = tuple(_safe_name(part) for part in prefix.replace("\\", "/").split("/") if part)
    return parts or ("data",)


def _safe_name(value: str) -> str:
    candidate = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    return candidate or "item"


def _csv_metadata(metadata: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in sorted(metadata.items()):
        if not isinstance(key, str) or key in _RESERVED_COLUMNS:
            continue
        safe = _json_safe(value)
        if safe is _UNSAFE:
            logger.warning("Dropping non-JSON-serializable Hugging Face metadata field %r", key)
            continue
        result[key] = safe if isinstance(safe, str) else json.dumps(safe, sort_keys=True)
    return result


def _json_object(metadata: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in sorted(metadata.items()):
        if not isinstance(key, str) or key in _RESERVED_COLUMNS:
            continue
        safe = _json_safe(value)
        if safe is _UNSAFE:
            logger.warning("Dropping non-JSON-serializable Hugging Face metadata field %r", key)
            continue
        result[key] = safe
    return result


_UNSAFE = object()


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else _UNSAFE
    if isinstance(value, list | tuple):
        items = [_json_safe(item) for item in value]
        return _UNSAFE if any(item is _UNSAFE for item in items) else items
    if isinstance(value, dict):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                return _UNSAFE
            safe = _json_safe(item)
            if safe is _UNSAFE:
                return _UNSAFE
            result[key] = safe
        return result
    return _UNSAFE
