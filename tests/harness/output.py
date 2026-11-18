"""Checks of a written dataset's files, independent of datamaite's loaders.

:func:`output_problems` reads a COCO or YOLO folder directly and lists what a
consumer would trip over: duplicate or dangling COCO ids, COCO images that are
not on disk, YOLO class indices outside ``names``, malformed YOLO rows, and
YOLO label files with no image.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from datamaite._types import DatasetFormat

YOLO_IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})


def output_problems(root: Path, fmt: DatasetFormat | str) -> list[str]:
    """Every problem found by reading the files under ``root`` as ``fmt``."""
    fmt = DatasetFormat(fmt)
    if fmt == DatasetFormat.COCO:
        return _coco_problems(root)
    if fmt == DatasetFormat.YOLO:
        return _yolo_problems(root)
    raise ValueError(f"no output checks for {fmt.value}")


def _coco_problems(root: Path) -> list[str]:
    problems = []
    files = sorted((root / "annotations").glob("*.json"))
    if not files:
        return ["output: no COCO annotation file under annotations/"]
    for path in files:
        name = path.relative_to(root).as_posix()
        document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        images, annotations = document.get("images", []), document.get("annotations", [])
        categories = document.get("categories", [])
        for what, items in (("image", images), ("annotation", annotations), ("category", categories)):
            repeated = sorted(str(i) for i, n in Counter(item.get("id") for item in items).items() if n > 1)
            if repeated:
                problems.append(f"output: {name}: repeated {what} ids {repeated}")
        image_ids = {image.get("id") for image in images}
        category_ids = {category.get("id") for category in categories}
        for annotation in annotations:
            if annotation.get("image_id") not in image_ids:
                problems.append(f"output: {name}: annotation {annotation.get('id')} refers to a missing image")
            if annotation.get("category_id") not in category_ids:
                problems.append(f"output: {name}: annotation {annotation.get('id')} refers to a missing category")
        for image in images:
            file_name = str(image.get("file_name", ""))
            if not any((base / file_name).is_file() for base in (root, root / "images")):
                problems.append(f"output: {name}: image {file_name!r} is not on disk")
    return problems


def _yolo_problems(root: Path) -> list[str]:
    data_yaml = root / "data.yaml"
    if not data_yaml.is_file():
        return ["output: no data.yaml"]
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = config.get("names")
    if isinstance(names, dict):
        if sorted(names) != list(range(len(names))):
            return [f"output: data.yaml: names keys {sorted(names)} are not 0..{len(names) - 1}"]
    elif not isinstance(names, list):
        return [f"output: data.yaml: names is {type(names).__name__}, not a list or mapping"]
    problems = _yolo_split_path_problems(root, config)
    if "nc" in config and config["nc"] != len(names):
        problems.append(f"output: data.yaml: nc is {config['nc']!r} but names has {len(names)} classes")
    images = {
        p.relative_to(root / "images").with_suffix("").as_posix() for p in (root / "images").rglob("*") if p.is_file()
    }
    for label in sorted((root / "labels").rglob("*.txt")):
        name = label.relative_to(root).as_posix()
        if label.relative_to(root / "labels").with_suffix("").as_posix() not in images:
            problems.append(f"output: {name}: no image for this label file")
        for number, row in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
            problems += [f"output: {name}: line {number}: {problem}" for problem in _yolo_row_problems(row, len(names))]
    return problems


#: data.yaml keys that are not splits.
_YOLO_CONFIG_KEYS = frozenset({"path", "names", "nc", "download", "datamaite_task"})


def _yolo_split_path_problems(root: Path, config: dict[str, Any]) -> list[str]:
    """Each wrong split path in ``config``, resolved from ``path`` as Ultralytics does, and each image none reaches.

    A split path is an image folder or a ``.txt`` list of images. datamaite's
    own loader falls back to the folders it finds, so a wrong path reads back
    fine there and fails only in another tool.
    """
    base = root / str(config.get("path") or ".")
    problems: list[str] = []
    reached: set[Path] = set()
    splits = {split: value for split, value in config.items() if split not in _YOLO_CONFIG_KEYS}
    for split, value in splits.items():
        for entry in value if isinstance(value, list) else [value]:
            problems += _yolo_split_entry_problems(base, split, entry, reached)
    images = [
        p for p in sorted((root / "images").rglob("*")) if p.suffix.lower() in YOLO_IMAGE_SUFFIXES and p.is_file()
    ]
    if images and not splits:
        # A written dataset with images must declare where they are.
        return [*problems, f"output: data.yaml: no split path, though images/ holds {len(images)} image(s)"]
    problems += [
        f"output: data.yaml: no split path reaches {image.relative_to(root).as_posix()}"
        for image in images
        if image.resolve() not in reached
    ]
    return problems


def _yolo_split_entry_problems(base: Path, split: str, entry: Any, reached: set[Path]) -> list[str]:
    """What is wrong with one split path ``entry``; the images it reaches are added to ``reached``."""
    source = base / str(entry)
    if not isinstance(entry, str):
        return [f"output: data.yaml: {split} is {entry!r}, not a path"]
    if not source.exists():
        return [f"output: data.yaml: {split} path {entry!r} is not on disk"]
    if source.is_dir():
        reached.update(p.resolve() for p in source.rglob("*") if p.suffix.lower() in YOLO_IMAGE_SUFFIXES)
        return []
    if source.suffix != ".txt":
        return [f"output: data.yaml: {split} path {entry!r} is not a folder or a .txt list"]
    problems = []
    for line in filter(None, map(str.strip, source.read_text(encoding="utf-8").splitlines())):
        listed = source.parent / line
        if not listed.is_file():
            problems.append(f"output: data.yaml: {split} list {entry!r} names {line!r}, not on disk")
        reached.add(listed.resolve())
    return problems


def _yolo_row_problems(row: str, n_classes: int) -> list[str]:
    fields = row.split()
    if not fields:
        return []
    try:
        index, coords = int(fields[0]), [float(value) for value in fields[1:]]
    except ValueError:
        return [f"not numbers: {row!r}"]
    problems = []
    if not 0 <= index < n_classes:
        problems.append(f"class {index} is outside names (0..{n_classes - 1})")
    if len(coords) != 4 or not all(0.0 <= value <= 1.0 for value in coords):
        problems.append(f"expected 4 coordinates in [0, 1], got {coords}")
    return problems


def relocate(tmp_path: Path) -> Path:
    """Move the written dataset in ``tmp_path / "out"`` and hide the source in ``tmp_path / "src"``.

    Loading from the new place with the source gone shows the output stands on
    its own: every image it lists is read from inside it. Check the output for
    absolute paths of the test folder (#157) before moving it.
    """
    if (tmp_path / "src").exists():
        (tmp_path / "src").rename(tmp_path / "src-gone")
    moved = tmp_path / "moved" / "dataset"
    moved.parent.mkdir()
    return (tmp_path / "out").rename(moved)


def tree_digest(root: Path) -> dict[str, str]:
    """``{relative path: sha256}`` of every file under ``root``."""
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
