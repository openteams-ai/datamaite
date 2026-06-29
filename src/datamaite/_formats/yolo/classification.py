"""YOLO/Ultralytics image-classification folder loader and writer.

The supported IC variant is the standard ImageFolder-style layout used by
Ultralytics YOLO classification datasets::

    root/
      train/cat/0001.jpg
      train/dog/0002.jpg
      val/cat/0003.jpg

The split directory is optional; without it, class folders may live directly
under ``root``. This module intentionally implements only IC. Future YOLO OD
support should register its own ``(Task.OD, DatasetFormat.YOLO, variant)`` key
instead of overloading these records.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Collection, Iterable
from pathlib import Path
from typing import Any, ClassVar

from datamaite._types import DatasetFormat, Task
from datamaite.image_classification import ImageClassificationDataset
from datamaite.loaders import Loader, register_loader
from datamaite.records import ClassificationLabel, DatasetMetadata, ImageClassificationSample
from datamaite.taxonomy import CategoryEntry, Taxonomy
from datamaite.writers import Writer, WriterCapabilities, register_writer

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
_SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
}
_SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}


@register_loader
class YoloImageClassificationLoader(Loader):
    """Load YOLO/Ultralytics image-classification folder datasets."""

    task: ClassVar[Task] = Task.IC
    format = DatasetFormat.YOLO
    variant: ClassVar[str] = "default"

    @classmethod
    def sniff(cls, root: str | Path) -> bool:
        path = Path(root)
        if not path.is_dir():
            return False
        return _looks_like_yolo_classification_root(path, _IMAGE_EXTENSIONS)

    def load(
        self,
        root: str | Path,
        *,
        image_extensions: Collection[str] | str | None = None,
        **_: Any,
    ) -> ImageClassificationDataset:
        """Read a YOLO classification dataset root.

        Images are discovered directly under each class folder (the documented
        flat ``<split>/<class>/<image>`` layout), matching :meth:`sniff`. Class
        indices are derived from the sorted class-folder names; an existing
        ``data.yaml`` is not consulted -- the on-disk folder layout is the
        source of truth for class names and order.
        """
        root_path = Path(root)
        if not root_path.is_dir():
            logger.warning("YOLO image-classification root is not a directory: %s", root_path)
            return ImageClassificationDataset(samples=(), dataset_metadata=DatasetMetadata(source_dataset="yolo"))

        extensions = _normalize_extensions(image_extensions)
        records = _discover_records(root_path, extensions)
        if not records:
            logger.warning("No YOLO image-classification images found in %s", root_path)
            return ImageClassificationDataset(samples=(), dataset_metadata=DatasetMetadata(source_dataset="yolo"))

        class_names = tuple(sorted({record[2] for record in records}))
        class_to_id = {name: idx for idx, name in enumerate(class_names)}
        taxonomy = Taxonomy(
            entries=tuple(CategoryEntry(source_id=idx, name=name) for idx, name in enumerate(class_names)),
            source_dataset="yolo",
            id_density="dense",
            ordered_names=class_names,
        )
        splits = tuple(_ordered_unique(record[1] for record in records if record[1] is not None))
        samples = tuple(
            ImageClassificationSample(
                image_id=rel_path,
                path_or_uri=str(image_path),
                file_name=rel_path,
                split=split,
                labels=(
                    ClassificationLabel(
                        category_id=class_to_id[class_name],
                        source_category_id=class_to_id[class_name],
                        category_name=class_name,
                    ),
                ),
                metadata={"source_format": "yolo", "variant": self.variant},
            )
            for image_path, split, class_name, rel_path in records
        )
        logger.info(
            "Loaded %d YOLO image-classification image(s), %d class(es) from %s",
            len(samples),
            len(class_names),
            root_path,
        )
        return ImageClassificationDataset(
            samples=samples,
            dataset_metadata=DatasetMetadata(taxonomy=taxonomy, source_dataset="yolo", splits=splits),
            dataset_id="yolo",
        )


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
            target = _write_sample(
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
            yaml_text = _data_yaml(
                splits=tuple(sorted(splits_seen, key=_split_sort_key)),
                names=sorted(classes_seen),
            )
            data_yaml.write_text(yaml_text, encoding="utf-8")
            written.append(data_yaml)
        return written


def load_yolo_image_classification(
    root: str | Path,
    *,
    image_extensions: Collection[str] | str | None = None,
) -> ImageClassificationDataset:
    """Convenience helper equivalent to ``load_ic(..., dataset_format='yolo')``."""
    return YoloImageClassificationLoader().load(root, image_extensions=image_extensions)


def _normalize_extensions(image_extensions: Collection[str] | str | None) -> frozenset[str]:
    if image_extensions is None:
        return _IMAGE_EXTENSIONS
    values = [image_extensions] if isinstance(image_extensions, str) else list(image_extensions)
    normalized = frozenset(ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in values)
    return normalized or _IMAGE_EXTENSIONS


def _looks_like_yolo_classification_root(root: Path, extensions: frozenset[str]) -> bool:
    """Shallow, cheap sniff for split/class/image or class/image layouts."""
    for child in _safe_children(root):
        if not child.is_dir():
            continue
        if _is_split_dir(child, extensions):
            return True
        if _infer_split(child.name) is None and _has_direct_image(child, extensions):
            return True
    return False


def _has_direct_image(path: Path, extensions: frozenset[str]) -> bool:
    return any(child.is_file() and child.suffix.lower() in extensions for child in _safe_children(path))


def _is_split_dir(child: Path, extensions: frozenset[str]) -> bool:
    """Whether ``child`` is a split directory in the ``<split>/<class>/<image>`` layout.

    A directory is a split only if its name reads as a split (``train``/``val``/
    ``test``/...) *and* it holds class subdirectories with direct images. This
    keeps the split-vs-flat decision structural and identical between ``sniff``
    and :func:`_discover_records`: a flat-layout class folder that happens to be
    named ``train``/``test`` is NOT mistaken for a split -- which previously made
    discovery silently drop every real class folder and return zero records.
    """
    if _infer_split(child.name) is None:
        return False
    return any(sub.is_dir() and _has_direct_image(sub, extensions) for sub in _safe_children(child))


def _discover_records(root: Path, extensions: frozenset[str]) -> list[tuple[Path, str | None, str, str]]:
    """Return ``(image_path, split, class_name, rel_path)`` rows."""
    split_dirs = [
        (child, _infer_split(child.name))
        for child in _safe_children(root)
        if child.is_dir() and _is_split_dir(child, extensions)
    ]
    records: list[tuple[Path, str | None, str, str]] = []
    if split_dirs:
        for split_dir, split in sorted(split_dirs, key=lambda item: (_split_sort_key(item[1]), item[0].name)):
            records.extend(_records_from_class_dirs(root, split_dir, split=split, extensions=extensions))
    else:
        records.extend(_records_from_class_dirs(root, root, split=None, extensions=extensions))
    return sorted(records, key=lambda row: row[3])


def _records_from_class_dirs(
    root: Path,
    base: Path,
    *,
    split: str | None,
    extensions: frozenset[str],
) -> list[tuple[Path, str | None, str, str]]:
    records: list[tuple[Path, str | None, str, str]] = []
    for class_dir in _safe_children(base):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        # Direct children only -- the documented flat layout, and consistent with
        # the shallow ``sniff``. (``rglob`` would silently flatten nested
        # subdirectories into the class and diverge from autodetect.)
        for image_path in _safe_children(class_dir):
            if not image_path.is_file() or image_path.suffix.lower() not in extensions:
                continue
            if image_path.is_symlink() and not _within(image_path, root):
                logger.warning("Skipping symlinked image escaping the dataset root: %s", image_path)
                continue
            records.append((image_path, split, class_name, _relative_posix(image_path, root)))
    return records


def _safe_children(path: Path) -> list[Path]:
    try:
        return sorted((child for child in path.iterdir() if not child.name.startswith(".")), key=lambda p: p.name)
    except OSError as exc:
        logger.warning("Could not read directory %s: %s", path, exc)
        return []


def _relative_posix(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _within(path: Path, root: Path) -> bool:
    """Whether ``path`` resolves to a location inside ``root`` (symlink-safe).

    Guards against an untrusted dataset smuggling a symlink that points outside
    the dataset root, which a later write would otherwise copy through.
    """
    try:
        return path.resolve().is_relative_to(root.resolve())
    except (OSError, RuntimeError):
        return False


def _infer_split(name: str) -> str | None:
    return _SPLIT_ALIASES.get(name.lower())


def _split_sort_key(split: str | None) -> tuple[int, str]:
    key = split or ""
    return (_SPLIT_ORDER.get(key, 99), key)


def _ordered_unique(values: Iterable[str | None]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in sorted((v for v in values if v is not None), key=_split_sort_key):
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _write_sample(
    sample: ImageClassificationSample,
    dest_path: Path,
    *,
    taxonomy: Taxonomy | None,
    default_split: str,
    used_targets: set[Path],
) -> Path | None:
    """Write one IC sample to ``<dest>/<split>/<class>/<file>``; return the path.

    Returns ``None`` (after a WARNING) for any sample the folder format cannot
    represent: no labels, an unresolved class, an unsafe split/class/file name,
    or a missing image source. Directories are created only once the sample is
    known to be writable, so a skipped sample never leaves an empty class folder.
    """
    label = _single_label(sample)
    if label is None:
        logger.warning("Skipping YOLO IC sample %r with no labels", sample.image_id)
        return None
    class_name = _class_name(label, taxonomy)
    if class_name is None:
        logger.warning("Skipping YOLO IC sample %r with unresolved label %r", sample.image_id, label)
        return None
    try:
        split = _safe_path_part(sample.split or default_split, field="split")
        safe_class_name = _safe_path_part(class_name, field="class name")
        file_name = _safe_file_name(sample)
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
    target = _unique_target(class_dir / file_name, used_targets)
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


def _class_name(label: ClassificationLabel, taxonomy: Taxonomy | None) -> str | None:
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


def _safe_path_part(value: str, *, field: str) -> str:
    text = str(value).strip()
    if not text or text in {".", ".."} or "/" in text or "\\" in text or "\x00" in text:
        raise ValueError(f"unsafe {field}: {value!r}")
    return text


def _safe_file_name(sample: ImageClassificationSample) -> str:
    raw = sample.file_name or (Path(sample.path_or_uri).name if sample.path_or_uri else f"{sample.image_id}.jpg")
    name = Path(str(raw)).name
    if not name or name in {".", ".."} or name != str(raw).replace("\\", "/").rsplit("/", 1)[-1] or "\x00" in name:
        raise ValueError(f"unsafe file name: {raw!r}")
    return name


def _unique_target(target: Path, used: set[Path]) -> Path:
    # ``is_symlink()`` in addition to ``exists()``: a *dangling* symlink reports
    # ``exists() == False`` but writing through it would land outside ``dest``.
    if _free_target(target, used):
        return target
    stem = target.stem
    suffix = target.suffix
    for index in range(2, 1_000_000):
        candidate = target.with_name(f"{stem}_{index}{suffix}")
        if _free_target(candidate, used):
            return candidate
    raise ValueError(f"could not allocate unique target near {target}")


def _free_target(target: Path, used: set[Path]) -> bool:
    return target not in used and not target.exists() and not target.is_symlink()


def _data_yaml(*, splits: tuple[str, ...], names: list[str]) -> str:
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
