"""Shared helpers for YOLO/Ultralytics readers and writers."""

from __future__ import annotations

import logging
from collections.abc import Collection, Iterable
from pathlib import Path, PurePosixPath

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
    """Whether ``path`` resolves to a location inside ``root`` (symlink-safe)."""
    try:
        return path.resolve().is_relative_to(root.resolve())
    except (OSError, RuntimeError):
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


def unique_target(target: Path, used: set[Path]) -> Path:
    # ``is_symlink()`` in addition to ``exists()``: a *dangling* symlink reports
    # ``exists() == False`` but writing through it would land outside ``dest``.
    if free_target(target, used):
        return target
    stem = target.stem
    suffix = target.suffix
    for index in range(2, 1_000_000):
        candidate = target.with_name(f"{stem}_{index}{suffix}")
        if free_target(candidate, used):
            return candidate
    raise ValueError(f"could not allocate unique target near {target}")


def free_target(target: Path, used: set[Path]) -> bool:
    return target not in used and not target.exists() and not target.is_symlink()
