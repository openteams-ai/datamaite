"""Shared helpers for YOLO/Ultralytics readers and writers."""

from __future__ import annotations

import logging
from collections.abc import Collection, Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from datamaite._io import current_write_mode, resolve_path
from datamaite._upath import is_remote_path

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
}
SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}
#: Top-level data.yaml keys that are never splits, though their values may look like paths.
RESERVED_YAML_KEYS = frozenset(
    {"path", "names", "nc", "download", "datamaite_task", "kpt_shape", "flip_idx", "channels", "yaml_file"}
)


def normalize_extensions(image_extensions: Collection[str] | str | None) -> frozenset[str]:
    if image_extensions is None:
        return IMAGE_EXTENSIONS
    values = [image_extensions] if isinstance(image_extensions, str) else list(image_extensions)
    normalized = frozenset(ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in values)
    return normalized or IMAGE_EXTENSIONS


def safe_children(path: Path) -> list[Path]:
    try:
        return sorted((child for child in path.iterdir() if not child.name.startswith(".")), key=lambda p: p.name)
    except OSError as exc:
        logger.warning("Could not read directory %s: %s", path, exc)
        return []


#: The data.yaml keys that always declare a split's images.
OD_SPLIT_KEYS = ("train", "val", "test")

#: The data.yaml keys whose missing ``../<path>`` Ultralytics retries as ``<path>`` (``check_det_dataset``),
#: for a single path. Roboflow writes ``train: ../train/images`` beside a ``train/`` folder.
PARENT_FALLBACK_KEYS = frozenset({"train", "val", "test", "minival"})


def ultralytics_split_path(base: Path, key: str, value: Any) -> Any:
    """A data.yaml split value as Ultralytics reads it: a missing ``../x`` under a train, val, test or
    minival key becomes ``x`` when that exists. Any other value is returned unchanged."""
    if key not in PARENT_FALLBACK_KEYS or not isinstance(value, str) or not value.strip().startswith("../"):
        return value
    text = value.strip()
    if resolve_path(base, text).exists() or not resolve_path(base, text[3:]).exists():
        return value
    return text[3:]


def infer_split(name: str) -> str | None:
    return SPLIT_ALIASES.get(name.lower())


def split_sort_key(split: str | None) -> tuple[int, str]:
    key = split or ""
    return (SPLIT_ORDER.get(key, 99), key)


def ordered_unique(values: Iterable[str | None]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in sorted((v for v in values if v is not None), key=split_sort_key):
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def relative_posix(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def within(path: Path, root: Path) -> bool:
    """Whether ``path`` is inside ``root``, resolving local symlinks safely."""
    try:
        if is_remote_path(root):
            # Object stores have no symlinks; lexical containment avoids one or
            # more remote metadata requests per discovered object.
            path.relative_to(root)
            return True
        return path.resolve().is_relative_to(root.resolve())
    except (OSError, RuntimeError, ValueError):
        return False


def safe_path_part(value: str, *, field: str) -> str:
    text = str(value).strip()
    if not text or text in {".", ".."} or "/" in text or "\\" in text or "\x00" in text:
        raise ValueError(f"unsafe {field}: {value!r}")
    return text


def safe_relative_path(value: str, *, field: str) -> PurePosixPath:
    text = str(value).strip()
    posix = PurePosixPath(text)
    if (
        not text
        or "\\" in text
        or "\x00" in text
        or posix.is_absolute()
        or not posix.parts
        or any(part in {"", ".", ".."} or ":" in part for part in posix.parts)
    ):
        raise ValueError(f"unsafe {field}: {value!r}")
    return posix


def unique_target(target: Path, used: set[Path], *, check_storage: bool = True) -> Path:
    # ``is_symlink()`` in addition to ``exists()``: a *dangling* symlink reports
    # ``exists() == False`` but writing through it would land outside ``dest``.
    if free_target(target, used, check_storage=check_storage):
        return target
    stem = target.stem
    suffix = target.suffix
    for index in range(2, 1_000_000):
        candidate = target.with_name(f"{stem}_{index}{suffix}")
        if free_target(candidate, used, check_storage=check_storage):
            return candidate
    raise ValueError(f"could not allocate unique target near {target}")


def free_target(target: Path, used: set[Path], *, check_storage: bool = True) -> bool:
    if target in used:
        return False
    if not check_storage:
        return True
    # Error/replace remote writes use an empty staging prefix; append keeps the
    # immediate candidate checks below to narrow concurrent-writer races.
    if is_remote_path(target) and current_write_mode() in {"error", "replace"}:
        return True
    return not target.exists() and not target.is_symlink()


def names_from_yaml(raw_names: Any) -> tuple[tuple[int, str], ...]:
    """Parse YOLO ``names`` into ``(source_id, name)`` pairs.

    Ultralytics normally writes a dense list, but YAML also commonly appears as
    a mapping (``{0: person, 1: car}``). Preserve mapping keys as source ids so
    sparse/non-contiguous mappings do not silently relabel detections.
    """
    if raw_names is None:
        return ()
    if isinstance(raw_names, str):
        # A quoted collection (``names: "[person, car]"``) arrives as text.
        try:
            parsed = yaml.safe_load(raw_names)
        except yaml.YAMLError:
            parsed = None
        if isinstance(parsed, Mapping | Sequence) and not isinstance(parsed, str):
            return names_from_yaml(parsed)
        name = raw_names.strip()
        return ((0, name),) if name else ()
    if isinstance(raw_names, Mapping):
        items: list[tuple[int, str]] = []
        for key, value in raw_names.items():
            class_id = coerce_yaml_int(key)
            if class_id is None:
                continue
            name = str(value).strip()
            if name:
                items.append((class_id, name))
        return tuple(sorted(items))
    if isinstance(raw_names, Sequence):
        return tuple((idx, name) for idx, value in enumerate(raw_names) if (name := str(value).strip()))
    return ()


def coerce_yaml_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None
