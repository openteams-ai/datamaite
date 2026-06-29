"""Hugging Face VideoFolder-style video classification dataset loader.

The Hugging Face Hub video dataset guide describes the video classification
layout as the ``VideoFolder`` convention: video files live under class-named
folders, optionally nested under split folders such as ``train/`` and ``test/``.
A dataset may instead provide ``metadata.csv`` or ``metadata.jsonl`` rows with
a required ``file_name`` column and optional ``label`` metadata. Experimental
``metadata.parquet`` reading is attempted when optional ``pyarrow`` or
``pandas`` support is installed; otherwise the loader warns and can fall back
to folder discovery. This loader implements that local repository layout
without requiring the Hugging Face ``datasets`` package.

Video classification has video-level labels rather than per-frame boxes, and
MAITE 0.9.5 has no video-classification protocol. The loader therefore returns
``VideoClassificationDataset`` records rather than masquerading as the MAITE
MOT ``BoxTrackDataset`` surface.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

from datamaite._types import DatasetFormat, Task
from datamaite.loaders import Loader, register_loader
from datamaite.model import VideoClassificationDataset, VideoClassificationSample

logger = logging.getLogger(__name__)

_VIDEO_EXTENSIONS = frozenset({".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"})
_METADATA_FILENAMES = ("metadata.csv", "metadata.jsonl", "metadata.parquet")
_LABEL_COLUMNS = ("label", "labels", "class", "category", "category_name")
_SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "validation": "validation",
    "valid": "validation",
    "val": "validation",
    "test": "test",
}
_SPLIT_ORDER = {"train": 0, "validation": 1, "test": 2}


@dataclass(frozen=True)
class _VideoRecord:
    """One candidate video row discovered from metadata or folder layout."""

    path: Path
    rel_path: str
    split: str | None
    label: str | None
    metadata_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@register_loader
class HuggingFaceVideoClassificationLoader(Loader):
    """Loader for Hugging Face VideoFolder video classification repositories."""

    task = Task.VC
    format = DatasetFormat.HUGGINGFACE_VIDEO_CLASSIFICATION

    def load(
        self,
        root: str | Path,
        *,
        video_extensions: Collection[str] | str | None = None,
        **_: Any,
    ) -> VideoClassificationDataset:
        """Read a Hugging Face video classification dataset root.

        Parameters
        ----------
        root
            Local dataset repository root. Supported shapes are class folders
            (``cat/*.mp4``), split/class folders (``train/cat/*.mp4``), or a
            Hugging Face metadata file with ``file_name`` and optional ``label``
            columns.
        video_extensions
            Optional allowlist of video extensions. Defaults to common Hugging
            Face-compatible video suffixes: ``.mp4``, ``.avi``, ``.mov``,
            ``.mkv``, ``.webm``, ``.m4v``, ``.mpeg``, and ``.mpg``.

        Returns
        -------
        VideoClassificationDataset
            One video-level classification sample per loadable file. This is a
            source-record dataset, not a MAITE dataset.
        """
        root = Path(root)
        if not root.is_dir():
            logger.warning("Hugging Face video classification root is not a directory: %s", root)
            return VideoClassificationDataset(samples=(), categories={})

        extensions = _normalize_video_extensions(video_extensions)
        metadata_files = _metadata_files(root)
        if metadata_files:
            records = _records_from_metadata(root, metadata_files, extensions)
            if not records and all(path.suffix.lower() == ".parquet" for path in metadata_files):
                logger.warning(
                    "Hugging Face parquet metadata produced no loadable rows in %s; falling back to folder discovery",
                    root,
                )
                records = _records_from_folder_layout(root, extensions)
        else:
            records = _records_from_folder_layout(root, extensions)
        samples, categories, labels = _build_dataset_records(records)
        if not samples:
            logger.warning("No loadable Hugging Face video classification files found in %s", root)

        logger.info(
            "Loaded %d Hugging Face video classification video(s), %d label(s) from %s",
            len(samples),
            len(categories),
            root,
        )
        return VideoClassificationDataset(samples=tuple(samples), categories=categories, labels=labels)


def load_huggingface_video_classification(
    root: str | Path,
    *,
    video_extensions: Collection[str] | str | None = None,
) -> VideoClassificationDataset:
    """Load a Hugging Face VideoFolder video classification dataset root.

    Equivalent to ``datamaite.load(root, dataset_format="huggingface_video_classification")``.
    See :meth:`HuggingFaceVideoClassificationLoader.load` for semantics.
    """
    return HuggingFaceVideoClassificationLoader().load(root, video_extensions=video_extensions)


def _records_from_metadata(
    root: Path,
    metadata_files: Iterable[Path],
    extensions: frozenset[str],
) -> list[_VideoRecord]:
    """Read candidate records from Hugging Face metadata files."""
    records: list[_VideoRecord] = []
    for metadata_path in metadata_files:
        for row_number, row in enumerate(_read_metadata_rows(metadata_path), start=1):
            record = _record_from_metadata_row(root, metadata_path, row, row_number, extensions)
            if record is not None:
                records.append(record)
    return records


def _record_from_metadata_row(
    root: Path,
    metadata_path: Path,
    row: Mapping[str, Any],
    row_number: int,
    extensions: frozenset[str],
) -> _VideoRecord | None:
    """Convert one metadata row into a video record, or skip with a warning."""
    file_name = row.get("file_name")
    posix = _parse_file_name(file_name, path=metadata_path, row_number=row_number)
    if posix is None:
        return None
    if posix.suffix.lower() not in extensions:
        logger.warning(
            "Skipping Hugging Face metadata row %s:%d with unsupported video extension: %s",
            metadata_path,
            row_number,
            file_name,
        )
        return None

    base_dir = _metadata_base_dir(root, metadata_path, posix)
    video_path = base_dir.joinpath(*posix.parts)
    if not _is_within_root(video_path, root):
        logger.warning(
            "Skipping Hugging Face metadata row %s:%d whose resolved path escapes the dataset root: %s",
            metadata_path,
            row_number,
            file_name,
        )
        return None

    split = _split_for_metadata_path(root, metadata_path) or _split_from_parts(posix.parts)
    label = _label_from_metadata(row, path=metadata_path, row_number=row_number) or _label_from_parts(posix.parts)
    return _VideoRecord(
        path=video_path,
        rel_path=_relative_posix(video_path, root),
        split=split,
        label=label,
        metadata_path=metadata_path,
        metadata={str(key): value for key, value in row.items() if key is not None},
    )


def _records_from_folder_layout(root: Path, extensions: frozenset[str]) -> list[_VideoRecord]:
    """Discover videos from class folders, optionally under split directories."""
    split_dirs = [child for child in _safe_children(root) if child.is_dir() and _infer_split(child.name) is not None]
    if split_dirs:
        records: list[_VideoRecord] = []
        for split_dir in sorted(split_dirs, key=lambda path: _split_sort_key(_infer_split(path.name), path.name)):
            records.extend(
                _records_from_video_tree(root, split_dir, split=_infer_split(split_dir.name), extensions=extensions)
            )
        return records
    return _records_from_video_tree(root, root, split=None, extensions=extensions)


def _records_from_video_tree(
    root: Path,
    base: Path,
    *,
    split: str | None,
    extensions: frozenset[str],
) -> list[_VideoRecord]:
    """Return video records under ``base`` with labels derived from first folder names."""
    records: list[_VideoRecord] = []
    for video_path in _iter_video_files(base, extensions):
        if not _is_within_root(video_path, root):
            logger.warning("Skipping Hugging Face video whose path escapes the dataset root: %s", video_path)
            continue
        try:
            rel_to_base = video_path.relative_to(base)
        except ValueError:
            rel_to_base = video_path
        records.append(
            _VideoRecord(
                path=video_path,
                rel_path=_relative_posix(video_path, root),
                split=split,
                label=_label_from_parts(rel_to_base.parts),
            )
        )
    return records


def _build_dataset_records(
    records: Iterable[_VideoRecord],
) -> tuple[list[VideoClassificationSample], dict[str, int], dict[int, str]]:
    """Filter candidate records, assign label IDs, and build model samples."""
    valid_records = _dedupe_records(_filter_existing_records(records))
    categories, label_lookup, labels = _category_maps(valid_records)
    samples: list[VideoClassificationSample] = []
    for record in sorted(valid_records, key=_record_sort_key):
        label_id, label_uri = label_lookup.get(record.label or "", (None, None))
        video_meta: dict[str, Any] = {
            "format": "huggingface_video_classification",
            "source_path": str(record.path),
            "file_name": record.rel_path,
        }
        if record.split is not None:
            video_meta["split"] = record.split
        if record.label is not None:
            video_meta.update({"label": record.label, "label_id": label_id, "label_uri": label_uri})
        if record.metadata_path is not None:
            video_meta["metadata_file"] = str(record.metadata_path)

        samples.append(
            VideoClassificationSample(
                video_id=len(samples),
                video_path=str(record.path),
                file_name=record.rel_path,
                label=record.label,
                label_id=label_id,
                label_uri=label_uri,
                split=record.split,
                metadata_path=str(record.metadata_path) if record.metadata_path is not None else None,
                video_meta=video_meta,
                metadata=dict(record.metadata),
                size_bytes=_file_size(record.path),
            )
        )
    return samples, categories, labels


def _filter_existing_records(records: Iterable[_VideoRecord]) -> list[_VideoRecord]:
    """Keep records whose resolved video file exists."""
    valid: list[_VideoRecord] = []
    for record in records:
        if not record.path.is_file():
            logger.warning("Skipping Hugging Face video classification entry with missing file: %s", record.path)
            continue
        valid.append(record)
    return valid


def _dedupe_records(records: Iterable[_VideoRecord]) -> list[_VideoRecord]:
    """Keep the first reference to a video path, preserving discovery order."""
    seen: set[Path] = set()
    deduped: list[_VideoRecord] = []
    for record in records:
        key = record.path.resolve(strict=False)
        if key in seen:
            logger.warning("Skipping duplicate Hugging Face video classification entry: %s", record.path)
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def _category_maps(
    records: Iterable[_VideoRecord],
) -> tuple[dict[str, int], dict[str, tuple[int, str]], dict[int, str]]:
    """Build stable dense class IDs from discovered label names."""
    labels = sorted(
        {record.label for record in records if record.label is not None}, key=lambda value: value.casefold()
    )
    categories: dict[str, int] = {}
    label_lookup: dict[str, tuple[int, str]] = {}
    labels_by_id: dict[int, str] = {}
    for category_id, label in enumerate(labels):
        uri = _label_uri(label)
        categories[uri] = category_id
        label_lookup[label] = (category_id, uri)
        labels_by_id[category_id] = label
    return categories, label_lookup, labels_by_id


def _metadata_files(root: Path) -> list[Path]:
    """Return root metadata files, or per-split metadata files when no root file exists."""
    root_files = _metadata_files_in_dir(root)
    if root_files:
        return root_files

    files: list[Path] = []
    for child in _safe_children(root):
        if child.is_dir() and _infer_split(child.name) is not None:
            files.extend(_metadata_files_in_dir(child))
    return files


def _metadata_files_in_dir(path: Path) -> list[Path]:
    """Return Hugging Face metadata files directly under ``path``."""
    children = {child.name.lower(): child for child in _safe_children(path) if child.is_file()}
    matches = [children[name] for name in _METADATA_FILENAMES if name in children]
    if len(matches) > 1:
        logger.warning("Multiple Hugging Face metadata files found in %s; loading all", path)
    return matches


def _read_metadata_rows(path: Path) -> list[dict[str, Any]]:
    """Read supported Hugging Face metadata file types into dictionaries."""
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
    """Read metadata.csv rows."""
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
    """Read metadata.jsonl rows."""
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
    """Read metadata.parquet rows when an optional parquet reader is installed."""
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


def _parse_file_name(value: object, *, path: Path, row_number: int) -> PurePosixPath | None:
    """Parse a safe Hugging Face metadata file_name value."""
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
    """Return the directory that ``file_name`` is relative to."""
    metadata_split = _split_for_metadata_path(root, metadata_path)
    if metadata_split is not None and rel_path.parts and _infer_split(rel_path.parts[0]) == metadata_split:
        return root
    return metadata_path.parent


def _split_for_metadata_path(root: Path, metadata_path: Path) -> str | None:
    """Infer split from a metadata file's directory relative to the root."""
    try:
        parts = metadata_path.parent.relative_to(root).parts
    except ValueError:
        return None
    return _infer_split(parts[0]) if parts else None


def _split_from_parts(parts: tuple[str, ...]) -> str | None:
    """Infer split from the first relative path component."""
    return _infer_split(parts[0]) if parts else None


def _label_from_metadata(row: Mapping[str, Any], *, path: Path, row_number: int) -> str | None:
    """Return a scalar label from known metadata columns, if present."""
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
    """Infer class label from the first non-split directory in a relative path."""
    start = 1 if parts and _infer_split(parts[0]) is not None else 0
    return parts[start] if len(parts) - start >= 2 else None


def _coerce_label(value: object) -> str | None:
    """Coerce a metadata label scalar to a non-empty string."""
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
    """Return True for empty/NA-like metadata values."""
    return (
        value is None
        or (isinstance(value, str) and not value.strip())
        or (isinstance(value, float) and math.isnan(value))
    )


def _normalize_video_extensions(value: Collection[str] | str | None) -> frozenset[str]:
    """Validate and normalize video extension allowlist values."""
    if value is None:
        return _VIDEO_EXTENSIONS
    raw_extensions = [value] if isinstance(value, str) else list(value)
    normalized: set[str] = set()
    for raw in raw_extensions:
        ext = str(raw).strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        if ext == "." or "/" in ext or "\\" in ext or ".." in ext:
            raise ValueError(f"video_extensions entries must be safe extensions, got {raw!r}")
        normalized.add(ext)
    if not normalized:
        raise ValueError("video_extensions must include at least one extension")
    return frozenset(normalized)


def _iter_video_files(base: Path, extensions: frozenset[str]) -> list[Path]:
    """Return supported video files recursively under ``base``."""
    try:
        return sorted(path for path in base.rglob("*") if path.is_file() and path.suffix.lower() in extensions)
    except OSError as exc:
        logger.warning("Could not list Hugging Face video tree %s: %s", base, exc)
        return []


def _safe_children(path: Path) -> list[Path]:
    """Return sorted immediate children, logging and returning empty on OS errors."""
    try:
        return sorted(path.iterdir(), key=lambda item: item.name)
    except OSError as exc:
        logger.warning("Could not list Hugging Face video classification directory %s: %s", path, exc)
        return []


def _infer_split(name: str) -> str | None:
    """Map common Hugging Face split directory names to canonical split names."""
    return _SPLIT_ALIASES.get(name.lower().replace("_", "-"))


def _record_sort_key(record: _VideoRecord) -> tuple[int, str, str, str]:
    """Sort records in split/label/path order."""
    return (*_split_sort_key(record.split, ""), record.label or "", record.rel_path)


def _split_sort_key(split: str | None, fallback: str) -> tuple[int, str]:
    """Sort common splits in train/validation/test order."""
    return (_SPLIT_ORDER.get(split or "", len(_SPLIT_ORDER)), split or fallback)


def _label_uri(label: str) -> str:
    """Return an injective, stable category URI for a Hugging Face class label."""
    return f"huggingface_video_classification/label/{quote(label, safe='')}"


def _relative_posix(path: Path, root: Path) -> str:
    """Return ``path`` relative to ``root`` using POSIX separators when possible."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _is_within_root(path: Path, root: Path) -> bool:
    """Return True if ``path`` resolves under ``root``, catching symlink escapes."""
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except (OSError, ValueError):
        return False
    return True


def _file_size(path: Path) -> int | None:
    """Return file size in bytes, or None if unavailable."""
    try:
        return path.stat().st_size
    except OSError:
        return None
