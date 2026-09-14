"""YOLO/Ultralytics dataset loaders.

Two task variants are registered under the shared ``DatasetFormat.YOLO`` family:

* ``Task.IC``: the ImageFolder-style classification layout
  (``train/cat/0001.jpg`` or ``cat/0001.jpg``); images may also nest below the
  class directory (``train/cat/sub/0001.jpg``) without the subdirectory
  becoming a class (#90).
* ``Task.OD``: the standard YOLO detection layout with image files mirrored by
  ``.txt`` label files (``images/train/0001.jpg`` + ``labels/train/0001.txt``),
  including the common ``train/images`` + ``train/labels`` variant.

The task axis keeps the two variants independent: use ``load_ic(...,
dataset_format="yolo")`` for classification and ``load_od(...,
dataset_format="yolo")`` for object detection.
"""

from __future__ import annotations

import ast
import logging
import math
import struct
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar

from datamaite._formats.yolo._common import (
    IMAGE_EXTENSIONS,
    SPLIT_ALIASES,
    infer_split,
    normalize_extensions,
    ordered_unique,
    relative_posix,
    safe_children,
    split_sort_key,
    within,
)
from datamaite._io import list_files, resolve_path, resource_key
from datamaite._types import DatasetFormat, Task
from datamaite._upath import is_remote_path, storage_options_for, to_dataset_path
from datamaite.geometry import from_yolo, has_positive_area
from datamaite.image_classification import ImageClassificationDataset
from datamaite.loaders import Loader, register_loader
from datamaite.object_detection import ObjectDetectionDataset
from datamaite.records import (
    ClassificationLabel,
    DatasetMetadata,
    ImageClassificationSample,
    ImageObjectDetectionSample,
    ObjectDetectionAnnotation,
)
from datamaite.taxonomy import CategoryEntry, Taxonomy

logger = logging.getLogger(__name__)

_YOLO_YAML_NAMES = ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml")
_OD_SPLIT_KEYS = ("train", "val", "test")
_SOFS = frozenset({0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF})
_MAX_JPEG_HEADER_SCAN = 256 * 1024


@register_loader
class YoloImageClassificationLoader(Loader):
    """Load YOLO/Ultralytics image-classification folder datasets."""

    task: ClassVar[Task] = Task.IC
    format = DatasetFormat.YOLO
    variant: ClassVar[str] = "default"
    supports_remote: ClassVar[bool] = True

    @classmethod
    def sniff(cls, root: str | Path) -> bool:
        path = to_dataset_path(root)
        declared_task = _declared_yolo_task(path)
        if not path.is_dir() or declared_task not in {None, Task.IC.value}:
            return False
        return _looks_like_yolo_classification_root(path, IMAGE_EXTENSIONS) or (
            declared_task == Task.IC.value and _yaml_has_names(path)
        )

    def load(
        self,
        root: str | Path,
        *,
        image_extensions: Collection[str] | str | None = None,
        split: str | Collection[str] | None = None,
        layout: str = "auto",
        storage_options: Mapping[str, Any] | None = None,
        **_: Any,
    ) -> ImageClassificationDataset:
        """Read a YOLO classification dataset root.

        Images are discovered recursively below each class folder (#90):
        ``<split>/<class>/**/<image>`` keeps the top-level directory as the
        class, and the nested relative path is preserved in the datum ID, so
        nested subdirectories never become classes. :meth:`sniff` deliberately
        stays shallow -- a nested-only root will not autodetect and needs an
        explicit ``dataset_format="yolo"``. Class indices are derived from the
        sorted class-folder names; an existing ``data.yaml`` is not consulted
        -- the on-disk folder layout is the source of truth for class names
        and order.

        ``layout`` selects the directory interpretation. ``"auto"`` (default)
        keeps the structural discriminator: the root uses the split layout
        when some split-named child holds a class subdirectory with an image
        at any depth. A flat root whose class directory is named like a split
        *and* nests its images (``train/roll-01/a.jpg``) is structurally
        indistinguishable from a split, so auto reads it as one -- it warns
        about every non-split sibling it drops. Pass ``layout="flat"`` to read
        every top-level directory as a class, or ``layout="split"`` to force
        the split interpretation.

        ``split`` (#86) loads only the given split(s), with the same alias
        handling and selection semantics as the OD loader's option (#78):
        ``"validation"``/``"valid"`` normalise to ``val``, ``"training"`` to
        ``train``, and an explicit selection that matches nothing selects
        nothing -- it warns and yields an empty dataset, never widening back
        to all splits. The taxonomy is built from the selected splits' class
        directories (including empty ones, #81), so it is split-local exactly
        as if the split directory had been loaded as its own root.

        Symlinked directories -- split, class, or nested -- are never
        descended into, and every discovered image must resolve inside the
        dataset root, so no symlink can smuggle outside files into a dataset.
        """
        if layout not in ("auto", "split", "flat"):
            raise ValueError(f'layout must be "auto", "split", or "flat", got {layout!r}')
        root_path = to_dataset_path(root, storage_options)
        if not root_path.is_dir():
            logger.warning("YOLO image-classification root is not a directory: %s", root_path)
            return ImageClassificationDataset(samples=(), dataset_metadata=DatasetMetadata(source_dataset="yolo"))

        extensions = normalize_extensions(image_extensions)
        selected_splits = _normalize_split_selection(split)
        records, class_name_list, declared_splits = _discover_classification_records(
            root_path, extensions, splits=selected_splits, layout=layout
        )
        if not class_name_list:
            yaml_path = _find_yolo_yaml(root_path)
            yaml_data = _read_data_yaml(yaml_path) if yaml_path is not None else {}
            class_name_list = [name for _, name in _names_from_yaml(yaml_data.get("names"))]
            declared_splits = tuple(
                split_name
                for split_name in _OD_SPLIT_KEYS
                if yaml_data.get(split_name) not in (None, "", [], ())
                and (selected_splits is None or split_name in selected_splits)
            )
        if not records:
            logger.warning("No YOLO image-classification images found in %s", root_path)
            if not class_name_list:
                return ImageClassificationDataset(samples=(), dataset_metadata=DatasetMetadata(source_dataset="yolo"))

        # Union of every class subdirectory (already sorted), including empty
        # ones -- not just classes that contain images -- so dense label indices
        # stay stable across splits (#81).
        class_names = tuple(class_name_list)
        class_to_id = {name: idx for idx, name in enumerate(class_names)}
        taxonomy = Taxonomy(
            entries=tuple(CategoryEntry(source_id=idx, name=name) for idx, name in enumerate(class_names)),
            source_dataset="yolo",
            id_density="dense",
            ordered_names=class_names,
        )
        splits = tuple(ordered_unique(record[1] for record in records if record[1] is not None)) or declared_splits
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
            _storage_options=storage_options_for(root_path),
        )


@register_loader
class YoloObjectDetectionLoader(Loader):
    """Load YOLO/Ultralytics object-detection datasets."""

    task: ClassVar[Task] = Task.OD
    format = DatasetFormat.YOLO
    variant: ClassVar[str] = "default"
    supports_remote: ClassVar[bool] = True

    @classmethod
    def sniff(cls, root: str | Path) -> bool:
        path = to_dataset_path(root)
        declared_task = _declared_yolo_task(path)
        if not path.is_dir() or declared_task not in {None, Task.OD.value}:
            return False
        return _looks_like_yolo_od_root(path, IMAGE_EXTENSIONS) or (
            declared_task == Task.OD.value and _yaml_has_names(path)
        )

    def load(
        self,
        root: str | Path,
        *,
        image_extensions: Collection[str] | str | None = None,
        split: str | Collection[str] | None = None,
        yaml_file: str | Path | None = None,
        ann_dir: str | Path | None = None,
        storage_options: Mapping[str, Any] | None = None,
        **_: Any,
    ) -> ObjectDetectionDataset:
        """Read a YOLO detection dataset root.

        Supported layouts include both common Ultralytics arrangements::

            root/images/train/*.jpg      root/labels/train/*.txt
            root/train/images/*.jpg      root/train/labels/*.txt

        plus the split-less ``root/images`` + ``root/labels`` variant. A
        ``data.yaml``/``data.yml`` file is used for class names and, when it
        declares split paths, for image discovery. Labels are standard YOLO
        ``class cx cy w h`` rows with normalized center boxes; a sixth value is
        accepted as a confidence score for prediction-style TXT files.

        Options (defaults preserve the previous whole-root behavior):

        * ``split`` -- load only the given split(s) (``"train"``/``"val"``/
          ``"test"`` or a collection; aliases like ``"validation"`` normalise).
          Following datamaite's loader contract, an unknown or absent split
          warns and yields an empty dataset rather than raising. A selection
          that matches nothing selects nothing -- it never widens back to all
          splits.
        * ``yaml_file`` -- an explicit ``data.yaml`` path (relative to ``root``
          or absolute) instead of the conventional-name discovery at the root.
          It is authoritative: if it is missing, or declares image sources that
          yield nothing, the result is an empty dataset rather than a fallback
          scan of ``root``. Relative paths inside it resolve against ``path:``,
          or against the YAML's own directory when ``path:`` is absent.
        * ``ann_dir`` -- a label/annotation directory override (relative to
          ``root`` or absolute), for label trees kept outside the conventional
          ``labels/`` sibling of ``images/``; supplying it also removes the need
          for a conventional ``labels/`` directory to exist. The image's
          structure below ``root`` is mirrored under ``ann_dir`` minus any
          ``images`` component, so ``images/train/a.png`` reads
          ``<ann_dir>/train/a.txt`` and equally-named images in different splits
          stay distinct. A flat ``<ann_dir>/<stem>.txt`` layout is also honoured
          where exactly one image claims a given file; contested names are left
          unlabelled with a warning.
        """
        root_path = to_dataset_path(root, storage_options)
        if not root_path.is_dir():
            logger.warning("YOLO object-detection root is not a directory: %s", root_path)
            return ObjectDetectionDataset(samples=(), dataset_metadata=DatasetMetadata(source_dataset="yolo"))

        extensions = normalize_extensions(image_extensions)
        yaml_path = _resolve_yaml_path(root_path, yaml_file)
        if yaml_file is not None and yaml_path is None:
            # Requested config is missing; loading the root's own data.yaml here
            # would quietly hand back a different dataset than the caller asked for.
            return ObjectDetectionDataset(samples=(), dataset_metadata=DatasetMetadata(source_dataset="yolo"))
        yaml_data = _read_data_yaml(yaml_path) if yaml_path is not None else {}
        selected_splits = _normalize_split_selection(split)
        labels_base = resolve_path(root_path, ann_dir) if ann_dir is not None else None
        records = _discover_od_records(
            root_path,
            extensions,
            yaml_data=yaml_data,
            yaml_path=yaml_path,
            splits=selected_splits,
            labels_base=labels_base,
            yaml_is_explicit=yaml_file is not None,
        )
        names = _names_from_yaml(yaml_data.get("names")) if yaml_data else ()
        if not records:
            logger.warning("No YOLO object-detection images found in %s", root_path)
            if not names:
                return ObjectDetectionDataset(samples=(), dataset_metadata=DatasetMetadata(source_dataset="yolo"))
            taxonomy = _build_od_taxonomy(names, ())
            declared_splits = tuple(
                split_name
                for split_name in _OD_SPLIT_KEYS
                if yaml_data.get(split_name) not in (None, "", [], ())
                and (selected_splits is None or split_name in selected_splits)
            )
            return ObjectDetectionDataset(
                samples=(),
                dataset_metadata=DatasetMetadata(taxonomy=taxonomy, source_dataset="yolo", splits=declared_splits),
                dataset_id="yolo",
                _storage_options=storage_options_for(root_path),
            )

        taxonomy = _build_od_taxonomy(names, records)
        names_by_id = taxonomy.index2label()
        samples: list[ImageObjectDetectionSample] = []
        for record in records:
            width, height = _read_image_size(record.image_path)
            detections = (
                ()
                if record.label_ambiguous
                else _load_label_file(
                    record.label_path,
                    image_width=width,
                    image_height=height,
                    names_by_id=names_by_id,
                )
            )
            samples.append(
                ImageObjectDetectionSample(
                    image_id=relative_posix(record.image_path, root_path),
                    path_or_uri=str(record.image_path),
                    file_name=record.file_name,
                    width=width,
                    height=height,
                    split=record.split,
                    detections=detections,
                    metadata={
                        "source_format": "yolo",
                        "variant": self.variant,
                        "source_file_name": relative_posix(record.image_path, root_path),
                        "label_file": relative_posix(record.label_path, root_path),
                    },
                )
            )

        splits = tuple(ordered_unique(record.split for record in records if record.split is not None))
        logger.info(
            "Loaded %d YOLO object-detection image(s), %d annotation(s), %d class(es) from %s",
            len(samples),
            sum(len(sample.detections) for sample in samples),
            len(taxonomy.entries),
            root_path,
        )
        return ObjectDetectionDataset(
            samples=tuple(samples),
            dataset_metadata=DatasetMetadata(taxonomy=taxonomy, source_dataset="yolo", splits=splits),
            dataset_id="yolo",
            _storage_options=storage_options_for(root_path),
        )


# ---------------------------------------------------------------------------
# Classification loader helpers
# ---------------------------------------------------------------------------


def _looks_like_yolo_classification_root(root: Path, extensions: frozenset[str]) -> bool:
    """Shallow, cheap sniff for split/class/image or class/image layouts.

    Deliberately shallower than the recursive discovery in
    :func:`_classification_records_from_class_dirs` (#90): treating any
    directory with images somewhere below it as a class dir would make almost
    any dataset root sniff as YOLO IC and break autodetect with ambiguous
    matches. A nested-only root therefore requires an explicit
    ``dataset_format="yolo"``.
    """
    for child in safe_children(root):
        if not child.is_dir() or child.is_symlink():
            continue
        if _is_classification_split_dir(child, extensions):
            return True
        if infer_split(child.name) is None and _has_direct_image(child, extensions):
            return True
    return False


def _has_direct_image(path: Path, extensions: frozenset[str]) -> bool:
    return any(child.is_file() and child.suffix.lower() in extensions for child in safe_children(path))


def _is_classification_split_dir(child: Path, extensions: frozenset[str]) -> bool:
    """Whether ``child`` is a split directory in the ``<split>/<class>/<image>`` layout.

    Shallow (direct-child images only) -- this is the ``sniff`` contract. The
    load-time layout discriminator goes deep via :class:`_ClassDirScan`
    instead (#90).
    """
    if infer_split(child.name) is None:
        return False
    return any(
        sub.is_dir() and not sub.is_symlink() and _has_direct_image(sub, extensions) for sub in safe_children(child)
    )


class _ClassDirScan:
    """One-pass, symlink-safe image discovery below class directories (#90).

    Directory listings and per-class-dir image walks are memoized, so the
    layout discriminator, the dropped-directory warning, and record building
    together list each directory exactly once per load. Symlinked directories
    are never descended into -- matching ``rglob``'s ``**`` semantics -- so a
    link can neither smuggle an outside tree into a class nor form a cycle,
    and every discovered image must itself resolve inside the dataset root,
    whether or not its final path component is a symlink.
    """

    def __init__(self, root: Path, extensions: frozenset[str]) -> None:
        self._root = root
        self._extensions = extensions
        self._children: dict[Path, list[Path]] = {}
        self._entry_types: dict[Path, str] = {}
        self._empty_markers: set[Path] = set()
        self._images: dict[Path, tuple[Path, ...]] = {}

    def children(self, path: Path) -> list[Path]:
        cached = self._children.get(path)
        if cached is None:
            cached = self._remote_children(path) if is_remote_path(path) else safe_children(path)
            self._children[path] = cached
        return cached

    def _remote_children(self, path: Path) -> list[Path]:
        """List once with detail, retaining entry types to avoid N+1 HEADs."""
        try:
            infos = path.fs.listdir(path.path, detail=True)  # type: ignore[attr-defined]
        except OSError as exc:
            logger.warning("Could not read directory %s: %s", path, exc)
            return []
        children: list[Path] = []
        for info in infos:
            name = str(info["name"]).rstrip("/").rsplit("/", 1)[-1]
            if not name:
                continue
            if name == ".datamaite-empty":
                self._empty_markers.add(path)
                continue
            if name.startswith("."):
                continue
            child = path / name
            self._entry_types[child] = str(info.get("type", ""))
            children.append(child)
        return sorted(children, key=lambda child: child.name)

    def is_file(self, path: Path) -> bool:
        kind = self._entry_types.get(path)
        return kind == "file" if kind is not None else path.is_file()

    def is_dir(self, path: Path) -> bool:
        kind = self._entry_types.get(path)
        return kind in {"directory", "dir"} if kind is not None else path.is_dir()

    def is_symlink(self, path: Path) -> bool:
        return False if path in self._entry_types else path.is_symlink()

    def has_empty_marker(self, class_dir: Path) -> bool:
        if is_remote_path(class_dir):
            # Ensure the detailed listing has populated marker state.
            self.children(class_dir)
            return class_dir in self._empty_markers
        return (class_dir / ".datamaite-empty").is_file()

    def images(self, class_dir: Path) -> tuple[Path, ...]:
        """Every image below ``class_dir``, recursively, in listing order."""
        cached = self._images.get(class_dir)
        if cached is None:
            cached = self._images[class_dir] = tuple(self._walk(class_dir))
        return cached

    def _walk(self, directory: Path) -> Iterator[Path]:
        # Iterative depth-first traversal preserves listing order without
        # failing on deeply nested, untrusted directory trees.
        stack: list[Iterator[Path]] = [iter(self.children(directory))]
        while stack:
            try:
                child = next(stack[-1])
            except StopIteration:
                stack.pop()
                continue
            if self.is_file(child):
                if child.suffix.lower() not in self._extensions:
                    continue
                if not within(child, self._root):
                    logger.warning("Skipping image escaping the dataset root: %s", child)
                    continue
                yield child
            elif self.is_dir(child) and not self.is_symlink(child):
                stack.append(iter(self.children(child)))


def _is_deep_split_dir(child: Path, scan: _ClassDirScan) -> bool:
    """Load-time discriminator: a split holds a class subdir with an image at any depth (#90)."""
    if infer_split(child.name) is None:
        return False
    return any(
        scan.images(sub) or scan.has_empty_marker(sub)
        for sub in scan.children(child)
        if scan.is_dir(sub) and not scan.is_symlink(sub)
    )


def _real_child_dirs(base: Path, scan: _ClassDirScan) -> list[Path]:
    """Child directories of ``base``, skipping symlinked ones with a warning.

    Split and class directories follow the same no-descend symlink policy as
    nested directories; ``scan``'s listing memo means the warning fires once
    per directory per load.
    """
    child_dirs: list[Path] = []
    for child in scan.children(base):
        if not scan.is_dir(child):
            continue
        if scan.is_symlink(child):
            logger.warning("Skipping symlinked directory (symlinked directories are not descended): %s", child)
            continue
        child_dirs.append(child)
    return child_dirs


def _discover_classification_records(
    root: Path, extensions: frozenset[str], *, splits: frozenset[str] | None = None, layout: str = "auto"
) -> tuple[list[tuple[Path, str | None, str, str]], list[str], tuple[str, ...]]:
    """Return ``(rows, class_names)`` where each row is ``(image_path, split, class_name, rel_path)``.

    ``class_names`` is the sorted union of every class subdirectory seen across
    the selected splits, including class dirs that contain no images -- so an
    empty class in one split does not shift dense label indices relative to
    another (#81).

    ``splits`` restricts discovery to the given canonical split names (#86);
    ``None`` means no selection (load every split). An explicit selection never
    widens: a flat (split-less) layout under an explicit selection matches
    nothing, mirroring the OD loader's contract.

    ``layout`` overrides the split-vs-flat discriminator (``"split"`` /
    ``"flat"``); ``"auto"`` decides structurally and warns about non-split
    directories the split interpretation drops.
    """
    scan = _ClassDirScan(root, extensions)
    child_dirs = _real_child_dirs(root, scan)
    if layout == "split":
        uses_split_layout = True
    elif layout == "flat":
        uses_split_layout = False
    else:
        # Deciding split-vs-flat layout stays structural: a split must hold at
        # least one class subdirectory with an image (at any depth, #90). A
        # purely name-based test would misread a flat-layout class legitimately
        # named "train" as the sole split. The converse ambiguity -- a flat
        # class named "train" whose images are all nested -- is structurally
        # identical to a split, so it resolves as one; warn about anything that
        # interpretation drops, and let layout="flat" override it.
        uses_split_layout = any(_is_deep_split_dir(child, scan) for child in child_dirs)
        if uses_split_layout:
            dropped = [child.name for child in child_dirs if infer_split(child.name) is None and scan.images(child)]
            if dropped:
                logger.warning(
                    "Reading %s as a split layout; ignoring non-split directories with images: %s. "
                    'If these are class directories, pass layout="flat".',
                    root,
                    ", ".join(dropped),
                )
    records: list[tuple[Path, str | None, str, str]] = []
    class_names: set[str] = set()
    declared_splits: list[str] = []
    if uses_split_layout:
        # Once the layout is established, every split-named sibling is a split --
        # including one whose class dirs are all empty. Requiring each split to
        # contain an image would drop its declared classes from the union (#81).
        split_dirs = [(child, split) for child in child_dirs if (split := infer_split(child.name)) is not None]
        if splits is not None:
            split_dirs = [(child, split) for child, split in split_dirs if split in splits]
        declared_splits = ordered_unique(split for _, split in split_dirs)
        for split_dir, split in sorted(split_dirs, key=lambda item: (split_sort_key(item[1]), item[0].name)):
            recs, names = _classification_records_from_class_dirs(
                root, _real_child_dirs(split_dir, scan), split=split, scan=scan
            )
            records.extend(recs)
            class_names.update(names)
    elif splits is None:
        recs, names = _classification_records_from_class_dirs(root, child_dirs, split=None, scan=scan)
        records.extend(recs)
        class_names.update(names)
    return sorted(records, key=lambda row: row[3]), sorted(class_names), tuple(declared_splits)


def _classification_records_from_class_dirs(
    root: Path,
    class_dirs: list[Path],
    *,
    split: str | None,
    scan: _ClassDirScan,
) -> tuple[list[tuple[Path, str | None, str, str]], list[str]]:
    """Return ``(records, class_names)`` for one base dir's class directories.

    ``class_names`` lists *every* class subdirectory, including empty ones (no
    images), so the taxonomy is the union of declared classes rather than only
    the classes that happen to contain samples. This keeps dense label indices
    stable across splits when a class is empty in some split (#81). Symlinked
    class directories are already excluded (skipped entirely -- not descended,
    no taxonomy entry): ``class_dirs`` comes from :func:`_real_child_dirs`.
    """
    records: list[tuple[Path, str | None, str, str]] = []
    class_names: list[str] = []
    for class_dir in class_dirs:
        class_name = class_dir.name
        class_names.append(class_name)
        records.extend(
            (image_path, split, class_name, relative_posix(image_path, root)) for image_path in scan.images(class_dir)
        )
    return records, class_names


# ---------------------------------------------------------------------------
# Object-detection loader helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _OdRecord:
    image_path: Path
    label_path: Path
    split: str | None
    file_name: str
    label_ambiguous: bool = False


def _looks_like_yolo_od_root(root: Path, extensions: frozenset[str]) -> bool:
    yaml_path = _find_yolo_yaml(root)
    if yaml_path is not None:
        yaml_data = _read_data_yaml(yaml_path) or {}
        if _discover_od_records(root, extensions, yaml_data=yaml_data, yaml_path=yaml_path, limit=1):
            return True
    return bool(_discover_od_records(root, extensions, yaml_data={}, yaml_path=None, limit=1))


def _yaml_has_names(root: Path) -> bool:
    yaml_path = _find_yolo_yaml(root)
    return bool(yaml_path is not None and _names_from_yaml(_read_data_yaml(yaml_path).get("names")))


def _declared_yolo_task(root: Path) -> str | None:
    yaml_path = _find_yolo_yaml(root)
    if yaml_path is None:
        return None
    value = _read_data_yaml(yaml_path).get("datamaite_task")
    return str(value).lower() if value is not None else None


def _find_yolo_yaml(root: Path) -> Path | None:
    for name in _YOLO_YAML_NAMES:
        path = root / name
        if path.is_file():
            return path
    return None


def _resolve_yaml_path(root: Path, yaml_file: str | Path | None) -> Path | None:
    """Resolve an explicit ``yaml_file`` (relative to ``root`` or absolute).

    An explicit ``yaml_file`` is authoritative: when it does not exist this
    returns ``None`` rather than falling back to conventional discovery, so a
    typo yields an empty dataset instead of silently loading whatever
    ``data.yaml`` happens to sit at ``root``. Conventional-name discovery still
    applies when no ``yaml_file`` was requested.
    """
    if yaml_file is None:
        return _find_yolo_yaml(root)
    candidate = resolve_path(root, yaml_file)
    if candidate.is_file():
        return candidate
    logger.warning("YOLO OD yaml_file not found: %s", candidate)
    return None


def _normalize_split_selection(split: str | Collection[str] | None) -> frozenset[str] | None:
    """Normalize the requested split(s) to canonical keys.

    Returns ``None`` only when no selection was requested (load every split). An
    explicit selection always returns a (possibly empty) set: a selection that
    resolves to nothing must select *nothing*, never everything -- silently
    widening ``split="bogus"`` or ``split=[]`` back to all splits would leak
    training data into an evaluation run.
    """
    if split is None:
        return None
    raw = [split] if isinstance(split, str) else list(split)
    resolved: set[str] = set()
    unknown: list[str] = []
    for value in raw:
        canonical = infer_split(str(value))
        if canonical is None:
            unknown.append(str(value))
        else:
            resolved.add(canonical)
    if unknown:
        logger.warning(
            "YOLO: ignoring unrecognized split(s) %s; recognized values are %s",
            ", ".join(repr(value) for value in unknown),
            ", ".join(sorted(SPLIT_ALIASES)),
        )
    if not resolved:
        logger.warning("YOLO OD: split selection %r matched no known split; loading no images", split)
    return frozenset(resolved)


def _discover_od_records(
    root: Path,
    extensions: frozenset[str],
    *,
    yaml_data: Mapping[str, Any],
    yaml_path: Path | None,
    limit: int | None = None,
    splits: frozenset[str] | None = None,
    labels_base: Path | None = None,
    yaml_is_explicit: bool = False,
) -> list[_OdRecord]:
    records: list[_OdRecord] = []
    seen: set[str] = set()

    _selected_sources, hit_limit = _collect_yaml_records(
        records,
        seen=seen,
        root=root,
        extensions=extensions,
        yaml_data=yaml_data,
        yaml_path=yaml_path,
        limit=limit,
        splits=splits,
        labels_base=labels_base,
    )
    if hit_limit:
        return records

    # Source authority is determined from the whole explicit YAML, before split
    # filtering. If it declares only `train` while the caller requests `val`,
    # falling back to root discovery would load a val set the YAML never named.
    declared_sources = _yaml_declares_image_sources(yaml_data)
    names_only = bool(_names_from_yaml(yaml_data.get("names"))) and not declared_sources
    if yaml_is_explicit and declared_sources:
        if not records:
            logger.warning(
                "YOLO OD: yaml_file %s declares image sources but none of the selected sources yielded images; "
                "not falling back to root discovery",
                yaml_path,
            )
    elif yaml_is_explicit and not names_only:
        # Keep the deliberate names-only carve-out narrow. An unreadable,
        # malformed, non-mapping, or empty explicit config also produces no
        # sources, but must fail closed rather than masquerade as names-only.
        logger.warning(
            "YOLO OD: yaml_file %s declares neither usable image sources nor class names; "
            "not falling back to root discovery",
            yaml_path,
        )
    elif not records:
        for record in _records_from_standard_od_layouts(root, extensions, splits=splits, labels_base=labels_base):
            if _append_unique_record(records, record, seen=seen, limit=limit):
                return records

    records = sorted(
        records, key=lambda record: (split_sort_key(record.split), record.file_name, record.image_path.name)
    )
    if labels_base is not None:
        records = _resolve_ann_dir_labels(records, labels_base=labels_base)
    return records


def _yaml_declares_image_sources(yaml_data: Mapping[str, Any]) -> bool:
    """Whether any split key contains a source declaration, valid or not.

    Invalid source values still count as declarations for fail-closed behavior:
    an explicit ``train: 3`` must not silently turn into a root scan.
    """
    for split in _OD_SPLIT_KEYS:
        if split not in yaml_data:
            continue
        value = yaml_data[split]
        if value is None or value == "" or value == [] or value == ():
            continue
        return True
    return False


def _collect_yaml_records(
    records: list[_OdRecord],
    *,
    seen: set[str],
    root: Path,
    extensions: frozenset[str],
    yaml_data: Mapping[str, Any],
    yaml_path: Path | None,
    limit: int | None,
    splits: frozenset[str] | None,
    labels_base: Path | None,
) -> tuple[bool, bool]:
    """Append records for the YAML-declared splits; return (declared_any, hit_limit)."""
    if yaml_path is None:
        return False, False
    base = _yaml_dataset_base(yaml_path, yaml_data)
    declared = False
    for split in _OD_SPLIT_KEYS:
        if splits is not None and split not in splits:
            continue
        for source in _yaml_split_sources(yaml_data.get(split), base=base, yaml_path=yaml_path, root=root):
            declared = True
            for record in _records_from_image_source(
                source,
                root=root,
                split=split,
                extensions=extensions,
                labels_base=labels_base,
                warn_missing=limit is None,
            ):
                if _append_unique_record(records, record, seen=seen, limit=limit):
                    return declared, True
    return declared, False


def _append_unique_record(
    records: list[_OdRecord],
    record: _OdRecord,
    *,
    seen: set[str],
    limit: int | None,
) -> bool:
    key = resource_key(record.image_path)
    if key in seen:
        return False
    seen.add(key)
    records.append(record)
    return limit is not None and len(records) >= limit


def _yaml_dataset_base(yaml_path: Path, yaml_data: Mapping[str, Any]) -> Path:
    """Base directory that a data.yaml's relative split paths resolve against.

    Ultralytics resolves ``path:`` relative to the YAML's own directory, and
    split paths relative to ``path:`` -- or to the YAML directory when ``path``
    is absent. For a conventional ``data.yaml`` sitting at the dataset root the
    YAML directory *is* the root, so this only differs for a ``yaml_file``
    pointing somewhere nested (``configs/custom.yaml``), where resolving against
    the root would look in the wrong place.
    """
    raw_path = yaml_data.get("path")
    if isinstance(raw_path, str) and raw_path.strip():
        return resolve_path(yaml_path.parent, raw_path.strip())
    return yaml_path.parent


def _yaml_split_sources(raw_value: Any, *, base: Path, yaml_path: Path, root: Path) -> list[Path]:
    if raw_value is None:
        return []
    values = list(raw_value) if isinstance(raw_value, list | tuple) else [raw_value]
    sources: list[Path] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            continue
        path = resolve_path(base, value.strip())
        if is_remote_path(root) and not within(path, root):
            logger.warning("YOLO OD: refusing remote YAML source outside dataset root: %s", path)
            continue
        if path.is_file() and path.suffix.lower() == ".txt":
            sources.extend(_read_image_list(path, base=base, yaml_path=yaml_path, root=root))
        else:
            sources.append(path)
    return sources


def _read_image_list(path: Path, *, base: Path, yaml_path: Path, root: Path) -> list[Path]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Could not read YOLO image-list file %s: %s", path, exc)
        return []
    sources: list[Path] = []
    for raw in lines:
        text = raw.partition("#")[0].strip()
        if not text:
            continue
        candidate = resolve_path(base, text)
        if is_remote_path(root) and not within(candidate, root):
            logger.warning("YOLO OD: refusing remote image-list source outside dataset root: %s", candidate)
            continue
        if not candidate.exists():
            # Ultralytics treats paths in list files as relative to the dataset
            # YAML's ``path`` (or YAML directory when ``path`` is absent).
            alt = resolve_path(yaml_path.parent, text)
            candidate = alt if alt.exists() else candidate
        sources.append(candidate)
    return sources


def _records_from_standard_od_layouts(
    root: Path,
    extensions: frozenset[str],
    *,
    splits: frozenset[str] | None = None,
    labels_base: Path | None = None,
) -> list[_OdRecord]:
    records: list[_OdRecord] = []
    images_dir = root / "images"
    labels_dir = root / "labels"
    # An ann_dir override supplies the labels, so the conventional labels/ tree
    # is no longer required for the layout to be recognized -- otherwise a valid
    # images/ tree plus a separate annotation dir would load zero samples.
    if images_dir.is_dir() and (labels_dir.is_dir() or labels_base is not None):
        for image_path in _iter_images(images_dir, extensions=extensions, root=root):
            rel = image_path.relative_to(images_dir)
            split = infer_split(rel.parts[0]) if len(rel.parts) > 1 else None
            if splits is not None and split not in splits:
                continue
            file_name = PurePosixPath(*rel.parts[1:]).as_posix() if split is not None else rel.as_posix()
            records.append(
                _OdRecord(
                    image_path=image_path,
                    label_path=_label_path_for(image_path, images_dir, root=root, labels_base=labels_base),
                    split=split,
                    file_name=file_name,
                )
            )

    for child in safe_children(root):
        split = infer_split(child.name)
        if split is None or not child.is_dir():
            continue
        if splits is not None and split not in splits:
            continue
        split_images = child / "images"
        split_labels = child / "labels"
        if not split_images.is_dir() or not (split_labels.is_dir() or labels_base is not None):
            continue
        for image_path in _iter_images(split_images, extensions=extensions, root=root):
            rel = image_path.relative_to(split_images)
            records.append(
                _OdRecord(
                    image_path=image_path,
                    label_path=_label_path_for(image_path, split_images, root=root, labels_base=labels_base),
                    split=split,
                    file_name=rel.as_posix(),
                )
            )
    return records


def _label_path_for(image_path: Path, image_base: Path, *, root: Path, labels_base: Path | None) -> Path:
    """Resolve the label ``.txt`` path for an image.

    With ``labels_base`` (an ``ann_dir`` override) the image's structure below
    the dataset root is mirrored under the override, minus any ``images``
    component -- so ``images/train/a.png`` looks for ``<ann_dir>/train/a.txt``
    and images of the same name in different splits stay distinct. A flat
    ``<ann_dir>/<stem>.txt`` layout is still honoured, but only where it is
    unambiguous; see :func:`_resolve_ann_dir_labels`. Without an override, the
    conventional ``images`` -> ``labels`` swap is used.
    """
    if labels_base is not None:
        return (labels_base / _ann_relative_path(image_path, image_base, root)).with_suffix(".txt")
    return _infer_label_path(image_path, root=root)


def _ann_relative_path(image_path: Path, image_base: Path, root: Path) -> Path:
    """Structure to mirror under an ``ann_dir``, most split-preserving first."""
    for base in (root, image_base):
        try:
            rel = image_path.relative_to(base)
        except ValueError:
            continue
        parts = [part for part in rel.parts if part != "images"]
        if parts:
            return to_dataset_path(PurePosixPath(*parts).as_posix())
    return to_dataset_path(image_path.name)


def _resolve_ann_dir_labels(records: list[_OdRecord], *, labels_base: Path) -> list[_OdRecord]:
    """Resolve label candidates while refusing every multiply-claimed file.

    CheckMAITE's historical ``ann_dir`` is a flat directory of ``<stem>.txt``
    files for a *single* split, so that layout must keep working. A structured
    path is preferred; otherwise the flat path is considered. Claims include
    already-resolved structured paths as well as flat fallbacks: absolute YAML
    sources outside ``root`` can otherwise map two same-named images directly to
    the same existing ``<ann_dir>/<stem>.txt`` and bypass collision detection.
    """
    claims: dict[Path, list[int]] = {}
    for index, record in enumerate(records):
        candidate = record.label_path
        if not candidate.is_file():
            flat = labels_base / f"{record.image_path.stem}.txt"
            if not flat.is_file():
                continue
            candidate = flat
        claims.setdefault(candidate, []).append(index)
    if not claims:
        return records
    resolved = list(records)
    for candidate, indices in claims.items():
        if len(indices) == 1:
            resolved[indices[0]] = replace(records[indices[0]], label_path=candidate)
            continue
        for index in indices:
            resolved[index] = replace(records[index], label_path=candidate, label_ambiguous=True)
        logger.warning(
            "YOLO OD: ann_dir label %s is claimed by %d images (%s); leaving them unlabelled rather "
            "than assigning the same annotations to each -- give ann_dir per-split subdirectories, "
            "or load one split at a time with split=",
            candidate,
            len(indices),
            ", ".join(sorted(f"{records[index].split or '-'}/{records[index].file_name}" for index in indices)),
        )
    return resolved


def _records_from_image_source(
    source: Path,
    *,
    root: Path,
    split: str,
    extensions: frozenset[str],
    labels_base: Path | None = None,
    warn_missing: bool = False,
) -> list[_OdRecord]:
    if source.is_dir():
        records: list[_OdRecord] = []
        for image_path in _iter_images(source, extensions=extensions, root=root):
            rel = image_path.relative_to(source)
            records.append(
                _OdRecord(
                    image_path=image_path,
                    label_path=_label_path_for(image_path, source, root=root, labels_base=labels_base),
                    split=split,
                    file_name=rel.as_posix(),
                )
            )
        return records
    if source.is_file() and source.suffix.lower() in extensions:
        return [
            _OdRecord(
                image_path=source,
                label_path=_label_path_for(source, source.parent, root=root, labels_base=labels_base),
                split=split,
                file_name=_relative_image_file_name(source, root=root, split=split),
            )
        ]
    if warn_missing and not source.exists():
        logger.warning("YOLO OD: data.yaml %s source does not exist: %s", split, source)
    return []


def _infer_label_dir(image_dir: Path) -> Path:
    parts = list(image_dir.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "images":
            parts[index] = "labels"
            if is_remote_path(image_dir):
                return image_dir.with_segments(*parts)  # type: ignore[attr-defined]
            return to_dataset_path(PurePosixPath(*parts).as_posix())
    if infer_split(image_dir.name) is not None and image_dir.parent.name == "images":
        return image_dir.parent.parent / "labels" / image_dir.name
    return image_dir.parent / "labels"


def _infer_label_path(image_path: Path, *, root: Path) -> Path:
    try:
        rel = image_path.relative_to(root)
    except ValueError:
        return _infer_label_dir(image_path.parent) / f"{image_path.stem}.txt"
    parts = list(rel.parts)
    for index, part in enumerate(parts):
        if part == "images":
            parts[index] = "labels"
            return root.joinpath(*parts).with_suffix(".txt")
    if len(parts) > 2 and infer_split(parts[0]) is not None and parts[1] == "images":
        parts[1] = "labels"
        return root.joinpath(*parts).with_suffix(".txt")
    return root / "labels" / rel.with_suffix(".txt")


def _relative_image_file_name(image_path: Path, *, root: Path, split: str | None) -> str:
    try:
        rel = image_path.relative_to(root)
    except ValueError:
        return image_path.name
    parts = list(rel.parts)
    if parts and parts[0] == "images":
        parts.pop(0)
    if parts and split is not None and infer_split(parts[0]) == split:
        parts.pop(0)
    if parts and parts[0] == "images":
        parts.pop(0)
    return PurePosixPath(*parts).as_posix() if parts else image_path.name


def _iter_images(directory: Path, *, extensions: frozenset[str], root: Path) -> list[Path]:
    images: list[Path] = []
    try:
        candidates = (
            list_files(directory, recursive=True) if is_remote_path(directory) else sorted(directory.rglob("*"))
        )
    except OSError as exc:
        logger.warning("Could not read YOLO image directory %s: %s", directory, exc)
        return []
    for candidate in candidates:
        if any(part.startswith(".") for part in candidate.relative_to(directory).parts):
            continue
        if candidate.suffix.lower() not in extensions:
            continue
        if not is_remote_path(candidate) and not candidate.is_file():
            continue
        if candidate.is_symlink() and not within(candidate, root):
            logger.warning("Skipping symlinked YOLO image escaping the dataset root: %s", candidate)
            continue
        images.append(candidate)
    return images


def _build_od_taxonomy(names: Sequence[tuple[int, str]], records: Sequence[_OdRecord]) -> Taxonomy:
    if names:
        source_ids = [source_id for source_id, _name in names]
        id_density = "dense" if source_ids == list(range(len(source_ids))) else "sparse"
        return Taxonomy(
            entries=tuple(CategoryEntry(source_id=source_id, name=name) for source_id, name in names),
            source_dataset="yolo",
            id_density=id_density,
            ordered_names=tuple(name for _source_id, name in names),
        )
    class_ids = sorted(_scan_label_class_ids(records))
    id_density = "dense" if class_ids == list(range(len(class_ids))) else "sparse"
    return Taxonomy(
        entries=tuple(CategoryEntry(source_id=class_id, name=str(class_id)) for class_id in class_ids),
        source_dataset="yolo",
        id_density=id_density,
        ordered_names=tuple(str(class_id) for class_id in class_ids),
    )


def _scan_label_class_ids(records: Sequence[_OdRecord]) -> set[int]:
    class_ids: set[int] = set()
    for record in records:
        if record.label_ambiguous or not record.label_path.is_file():
            continue
        try:
            lines = record.label_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("Could not read YOLO label file %s: %s", record.label_path, exc)
            continue
        for raw in lines:
            text = raw.partition("#")[0].strip()
            if not text:
                continue
            fields = text.split()
            class_id = _parse_int_token(fields[0]) if fields else None
            if class_id is not None and class_id >= 0:
                class_ids.add(class_id)
    return class_ids


def _load_label_file(
    label_path: Path,
    *,
    image_width: int | None,
    image_height: int | None,
    names_by_id: Mapping[int, str],
) -> tuple[ObjectDetectionAnnotation, ...]:
    if not label_path.exists():
        return ()
    if not label_path.is_file():
        logger.warning("Skipping YOLO label path that is not a file: %s", label_path)
        return ()
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Could not read YOLO label file %s: %s", label_path, exc)
        return ()
    if image_width is None or image_height is None:
        if any(raw.partition("#")[0].strip() for raw in lines):
            logger.warning(
                "Skipping labels in %s because image dimensions could not be determined",
                label_path,
            )
        return ()
    detections: list[ObjectDetectionAnnotation] = []
    for line_number, raw in enumerate(lines, start=1):
        parsed = _parse_label_line(
            raw,
            line_number=line_number,
            label_path=label_path,
            image_width=image_width,
            image_height=image_height,
            names_by_id=names_by_id,
        )
        if parsed is not None:
            detections.append(parsed)
    return tuple(detections)


def _parse_label_line(
    raw: str,
    *,
    line_number: int,
    label_path: Path,
    image_width: int,
    image_height: int,
    names_by_id: Mapping[int, str],
) -> ObjectDetectionAnnotation | None:
    text = raw.partition("#")[0].strip()
    if not text:
        return None
    fields = text.split()
    if len(fields) not in {5, 6}:
        logger.warning("Skipping malformed YOLO label row %s:%d (expected 5 or 6 fields)", label_path, line_number)
        return None
    class_id = _parse_int_token(fields[0])
    if class_id is None or class_id < 0:
        logger.warning("Skipping YOLO label row %s:%d with invalid class id", label_path, line_number)
        return None
    coords = [_parse_float_token(value) for value in fields[1:5]]
    if any(value is None for value in coords):
        logger.warning("Skipping YOLO label row %s:%d with invalid bbox", label_path, line_number)
        return None
    cx, cy, width, height = (float(value) for value in coords if value is not None)
    if width <= 0 or height <= 0 or not all(0.0 <= value <= 1.0 for value in (cx, cy, width, height)):
        logger.warning("Skipping YOLO label row %s:%d with out-of-range normalized bbox", label_path, line_number)
        return None
    score = _parse_float_token(fields[5]) if len(fields) == 6 else None
    if len(fields) == 6 and (score is None or not 0.0 <= score <= 1.0):
        logger.warning("Skipping YOLO label row %s:%d with invalid confidence", label_path, line_number)
        return None
    bbox = from_yolo(cx, cy, width, height, float(image_width), float(image_height))
    if not has_positive_area(bbox):
        logger.warning("Skipping YOLO label row %s:%d with non-positive bbox area", label_path, line_number)
        return None
    if names_by_id and class_id not in names_by_id:
        logger.warning(
            "YOLO label row %s:%d references class id %d not defined in data.yaml names",
            label_path,
            line_number,
            class_id,
        )
    return ObjectDetectionAnnotation(
        bbox=bbox,
        category_id=class_id,
        category_name=names_by_id.get(class_id),
        source_category_id=class_id,
        score=score,
        attributes={"yolo_bbox": (cx, cy, width, height), "source_line": line_number},
    )


def _parse_int_token(value: object) -> int | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if str(parsed) == value or value == f"+{parsed}" else None


def _parse_float_token(value: object) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


# ---------------------------------------------------------------------------
# data.yaml parsing and image dimensions
# ---------------------------------------------------------------------------


def _read_data_yaml(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Could not read YOLO data YAML %s: %s", path, exc)
        return {}
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        yaml = None  # type: ignore[assignment]
    if yaml is not None:
        try:
            data = yaml.safe_load(text)
        except Exception as exc:
            logger.warning("Could not parse YOLO data YAML %s with PyYAML: %s", path, exc)
            return {}
        if not isinstance(data, dict):
            logger.warning("YOLO data YAML %s must contain a mapping", path)
            return {}
        return dict(data)
    return _parse_simple_yaml(text)


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Small fallback parser for the data.yaml shapes YOLO datasets usually use."""
    lines = text.splitlines()
    data: dict[str, Any] = {}
    index = 0
    while index < len(lines):
        raw = _strip_yaml_comment(lines[index])
        if not raw.strip():
            index += 1
            continue
        if raw[:1].isspace() or ":" not in raw:
            index += 1
            continue
        key, value = raw.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            data[key] = _parse_yaml_scalar(value)
            index += 1
            continue
        block: list[str] = []
        index += 1
        while index < len(lines):
            block_raw = _strip_yaml_comment(lines[index])
            if block_raw.strip() and not block_raw[:1].isspace():
                break
            if block_raw.strip():
                block.append(block_raw.strip())
            index += 1
        data[key] = _parse_yaml_block(block)
    return data


def _strip_yaml_comment(line: str) -> str:
    quote: str | None = None
    escaped = False
    chars: list[str] = []
    for char in line:
        if escaped:
            chars.append(char)
            escaped = False
            continue
        if char == "\\" and quote is not None:
            chars.append(char)
            escaped = True
            continue
        if char in {"'", '"'}:
            quote = None if quote == char else char if quote is None else quote
            chars.append(char)
            continue
        if char == "#" and quote is None:
            break
        chars.append(char)
    return "".join(chars).rstrip()


def _parse_yaml_scalar(value: str) -> Any:
    text = value.strip()
    if not text or text.lower() in {"null", "none", "~"}:
        return None
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text.strip("\"'")


def _parse_yaml_block(block: list[str]) -> Any:
    if not block:
        return None
    if all(line.startswith("-") for line in block):
        return [_parse_yaml_scalar(line[1:].strip()) for line in block]
    if all(":" in line for line in block):
        result: dict[Any, Any] = {}
        for line in block:
            key, value = line.split(":", 1)
            parsed_key = _parse_yaml_scalar(key.strip())
            result[parsed_key] = _parse_yaml_scalar(value.strip())
        return result
    return [_parse_yaml_scalar(line) for line in block]


def _names_from_yaml(raw_names: Any) -> tuple[tuple[int, str], ...]:
    """Parse YOLO ``names`` into ``(source_id, name)`` pairs.

    Ultralytics normally writes a dense list, but YAML also commonly appears as
    a mapping (``{0: person, 1: car}``). Preserve mapping keys as source ids so
    sparse/non-contiguous mappings do not silently relabel detections.
    """
    if raw_names is None:
        return ()
    if isinstance(raw_names, str):
        parsed = _parse_yaml_scalar(raw_names)
        if isinstance(parsed, Mapping | Sequence) and not isinstance(parsed, str):
            return _names_from_yaml(parsed)
        name = raw_names.strip()
        return ((0, name),) if name else ()
    if isinstance(raw_names, Mapping):
        items: list[tuple[int, str]] = []
        for key, value in raw_names.items():
            class_id = _coerce_yaml_int(key)
            if class_id is None:
                continue
            name = str(value).strip()
            if name:
                items.append((class_id, name))
        return tuple(sorted(items))
    if isinstance(raw_names, Sequence):
        return tuple((idx, name) for idx, value in enumerate(raw_names) if (name := str(value).strip()))
    return ()


def _coerce_yaml_int(value: Any) -> int | None:
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


def _read_image_size(path: Path) -> tuple[int | None, int | None]:
    try:
        open_kwargs = {"block_size": 64 * 1024} if is_remote_path(path) else {}
        with path.open("rb", **open_kwargs) as fh:  # type: ignore[call-overload]
            header = fh.read(32)
            if header.startswith(b"\x89PNG\r\n\x1a\n") and header[12:16] == b"IHDR":
                width, height = struct.unpack(">II", header[16:24])
                return _positive_size(width, height)
            if header.startswith(b"GIF87a") or header.startswith(b"GIF89a"):
                width, height = struct.unpack("<HH", header[6:10])
                return _positive_size(width, height)
            if header.startswith(b"BM") and len(header) >= 26:
                width = struct.unpack("<i", header[18:22])[0]
                height = abs(struct.unpack("<i", header[22:26])[0])
                return _positive_size(width, height)
            if header.startswith(b"\xff\xd8"):
                return _read_jpeg_size(fh)
    except (OSError, struct.error) as exc:
        logger.warning("Could not read image dimensions for %s: %s", path, exc)
        return (None, None)
    logger.warning("Could not determine image dimensions for %s", path)
    return (None, None)


def _read_jpeg_size(fh: Any) -> tuple[int | None, int | None]:  # noqa: C901 - JPEG marker scan is branchy
    # The SOI marker was consumed into the initial header; start scanning after it.
    fh.seek(2)
    while fh.tell() < _MAX_JPEG_HEADER_SCAN:
        marker_prefix = fh.read(1)
        if not marker_prefix:
            return (None, None)
        if marker_prefix != b"\xff":
            continue
        marker = fh.read(1)
        while marker == b"\xff":
            marker = fh.read(1)
        if not marker:
            return (None, None)
        marker_value = marker[0]
        if marker_value in {0x01, 0xD8, 0xD9} or 0xD0 <= marker_value <= 0xD7:
            continue
        length_bytes = fh.read(2)
        if len(length_bytes) != 2:
            return (None, None)
        length = struct.unpack(">H", length_bytes)[0]
        if length < 2:
            return (None, None)
        if fh.tell() + length - 2 > _MAX_JPEG_HEADER_SCAN:
            return (None, None)
        if marker_value in _SOFS:
            data = fh.read(length - 2)
            if len(data) < 5:
                return (None, None)
            height, width = struct.unpack(">HH", data[1:5])
            return _positive_size(width, height)
        fh.seek(length - 2, 1)
    return (None, None)


def _positive_size(width: int, height: int) -> tuple[int | None, int | None]:
    if width > 0 and height > 0:
        return (width, height)
    return (None, None)
