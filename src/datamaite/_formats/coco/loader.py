"""COCO object-detection dataset loader.

Reads a COCO detection JSON (the ``images`` / ``annotations`` / ``categories``
arrays, plus ``info`` / ``licenses``) into an
:class:`~datamaite.object_detection.ObjectDetectionDataset`. COCO's ``bbox`` is
``[x, y, width, height]`` in absolute pixels -- exactly datamaite's canonical
``xywh`` -- so boxes pass through without conversion.

Best-effort by the loader contract: malformed *data* is skipped and logged at
WARNING, never raised (a wrong user *argument* -- an explicit ``annotation_file``
that does not exist -- does raise); a (possibly empty) dataset is returned when
nothing loadable is found. ``load_coco`` returns ``ObjectDetectionDataset`` (a
native MAITE object-detection dataset), NOT the MOT ``BoxTrackDataset``. The
loader is registered in the shared registry, so both
``load(root, dataset_format="coco")`` and the task-first
:func:`datamaite.object_detection.load_od` dispatch here.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar

from datamaite._image_header import read_image_header
from datamaite._io import resolve_path
from datamaite._types import DatasetFormat, Task
from datamaite._upath import storage_options_for, to_dataset_path
from datamaite.loaders import Loader, register_loader
from datamaite.object_detection import ObjectDetectionDataset
from datamaite.records import DatasetMetadata, ImageObjectDetectionSample, ObjectDetectionAnnotation
from datamaite.taxonomy import CategoryEntry, SourceId, Taxonomy

logger = logging.getLogger(__name__)

# Keys consumed as typed fields; everything else on an annotation rides in attributes.
_ANNOTATION_CORE_KEYS = frozenset({"id", "image_id", "category_id", "bbox", "area", "iscrowd", "segmentation"})


@register_loader
class CocoLoader(Loader):
    """Loader for COCO detection datasets. Produces an ``ObjectDetectionDataset``."""

    task: ClassVar[Task] = Task.OD
    format: ClassVar[DatasetFormat] = DatasetFormat.COCO
    supports_remote: ClassVar[bool] = True

    @classmethod
    def sniff(cls, root: str | Path) -> bool:
        path = to_dataset_path(root)
        ann_paths = _resolve_annotation_files(path, None)
        if not ann_paths:
            return False
        data = _read_json(ann_paths[0])
        return bool(
            data is not None and "images" in data and "annotations" in data and not ({"videos", "tracks"} & data.keys())
        )

    def load(self, root: str | Path, **options: Any) -> ObjectDetectionDataset:
        return load_coco(root, **options)


def load_coco(
    root: str | Path,
    *,
    annotation_file: str | Path | None = None,
    images_dir: str | Path | None = None,
    storage_options: dict[str, Any] | None = None,
) -> ObjectDetectionDataset:
    """Load a COCO detection dataset into an :class:`ObjectDetectionDataset`.

    Parameters
    ----------
    root
        Either the COCO annotation JSON file itself, or a directory containing
        it (``annotations/instances*.json`` or any top-level ``*.json``). A
        directory with several ``instances*.json`` files loads all of them:
        each ``instances_<split>.json`` holds split ``<split>`` (CVAT's
        ``instances_Train.json`` holds split ``Train``), and ``instances.json``
        holds images with no split, as does CVAT's ``instances_default.json``
        (a task with no subsets).
    annotation_file
        Explicit annotation JSON path, overriding discovery under ``root``. A
        relative path anchors to ``root`` (not the caller's CWD), matching the
        other loaders' override style. Raises ``FileNotFoundError`` when it
        does not exist -- a wrong user argument, unlike malformed data, is not
        a best-effort case. Only this file is loaded; if it is named
        ``instances_<split>.json``, its images get split ``<split>``.
    images_dir
        Directory the ``images[].file_name`` paths are relative to; a relative
        path anchors to ``root``. Defaults to the dataset root: the annotation
        file's parent, or its grandparent when the file lives in an
        ``annotations/`` subdirectory (the standard COCO layout). For an
        ``instances_<split>.json`` file, ``images/<split>/``, ``<split>/`` and
        ``images/`` under that root are also tried (CVAT puts the images of
        ``instances_Train.json`` in ``images/Train/``, COCO 2017 those of
        ``instances_train2017.json`` in ``train2017/``), and the first that
        holds the file's images is used. With ``images_dir`` set, that
        directory and its ``<split>/`` subdirectory are tried.
    """
    root = to_dataset_path(root, storage_options)
    # Relative override paths anchor to ``root`` (its parent when root is the
    # annotation file itself), never the process CWD.
    anchor = root.parent if root.is_file() else root
    if annotation_file is not None:
        annotation_file = resolve_path(anchor, annotation_file)
    ann_paths = _resolve_annotation_files(root, annotation_file)
    if not ann_paths:
        logger.warning("No COCO annotation JSON found under %s", root)
        return ObjectDetectionDataset(samples=(), dataset_metadata=DatasetMetadata(source_dataset="coco"))
    images_override = resolve_path(anchor, images_dir) if images_dir is not None else None

    merged = _MergedFiles()
    for ann_path in ann_paths:
        data = _read_json(ann_path)
        if data is not None:
            merged.add(ann_path, data, images_dir=images_override)
    if not merged.read_any:
        return ObjectDetectionDataset(samples=(), dataset_metadata=DatasetMetadata(source_dataset="coco"))
    return ObjectDetectionDataset(
        samples=tuple(merged.samples), dataset_metadata=merged.metadata(), _storage_options=storage_options_for(root)
    )


def _load_file(
    ann_path: Path, data: dict[str, Any], *, images_dir: Path | None
) -> tuple[list[ImageObjectDetectionSample], Taxonomy]:
    """One annotation file's samples, with its split if it has one, and its categories."""
    if any(key in data for key in ("videos", "tracks")):
        # A TAO-style video-dataset JSON also carries images/annotations/
        # categories, so it would otherwise "load" with tracks silently
        # flattened into independent per-image boxes.
        logger.warning(
            "%s contains video-dataset keys (videos/tracks); this looks like a TAO-style video "
            "dataset, not still-image COCO. Use load_mot(..., dataset_format='tao') instead",
            ann_path,
        )
    file_split = _split_from_file_name(ann_path)
    # CVAT names the file of a task with no subsets instances_default.json: its images are in
    # images/default/, but have no split.
    split = None if file_split is not None and file_split.lower() == _CVAT_NO_SUBSET else file_split
    base_dir = _images_base_dir(ann_path, file_split, data.get("images"), images_dir=images_dir)

    taxonomy, names_by_id = _build_taxonomy(data.get("categories"), ann_path=ann_path)
    boxes_by_image = _annotations_by_image(data.get("annotations"), names_by_id, ann_path=ann_path)
    samples = _build_samples(data.get("images"), boxes_by_image, base_dir=base_dir, ann_path=ann_path)
    sample_ids = {sample.image_id for sample in samples}
    orphaned_ids = {image_id for image_id in boxes_by_image if image_id not in sample_ids}
    if orphaned_ids:
        logger.warning(
            "Dropping %d COCO annotation(s) referencing %d image id(s) missing from images[] in %s",
            sum(len(boxes_by_image[image_id]) for image_id in orphaned_ids),
            len(orphaned_ids),
            ann_path,
        )
    if split is not None:
        samples = [replace(sample, split=split) for sample in samples]
    logger.info("Loaded %d COCO image(s), %d categories from %s", len(samples), len(names_by_id), ann_path)
    return samples, taxonomy


class _MergedFiles:
    """Several annotation files (one per split) as one dataset.

    Samples are concatenated in file order. Each category id keeps the first
    file's entry. A later file that names an id differently has that class
    moved, with a warning, to the id that already has its name, or else to a
    new id after the others, and its boxes follow, so every box's id and name
    agree with the taxonomy. ``info`` is the first file's, and ``licenses``
    lists each distinct license once.
    """

    def __init__(self) -> None:
        self.read_any = False
        self.samples: list[ImageObjectDetectionSample] = []
        self._categories: dict[SourceId, CategoryEntry] = {}
        self._info: dict[str, Any] | None = None
        self._licenses: list[dict[str, Any]] = []
        self._splits: list[str] = []

    def add(self, ann_path: Path, data: dict[str, Any], *, images_dir: Path | None) -> None:
        self.read_any = True
        samples, taxonomy = _load_file(ann_path, data, images_dir=images_dir)
        moved = self._add_categories(taxonomy.entries, ann_path)
        if moved:
            samples = [_with_category_ids(sample, moved) for sample in samples]
        self.samples.extend(samples)
        for split in dict.fromkeys(sample.split for sample in samples if sample.split is not None):
            if split not in self._splits:
                self._splits.append(split)
        if self._info is None and isinstance(data.get("info"), dict):
            self._info = dict(data["info"])
        licenses = data.get("licenses")
        for lic in licenses if isinstance(licenses, list) else []:
            if isinstance(lic, dict) and lic not in self._licenses:
                self._licenses.append(lic)

    def _add_categories(self, entries: tuple[CategoryEntry, ...], ann_path: Path) -> dict[SourceId, SourceId]:
        """Add a file's categories; return the ids of those moved because an earlier file names the id differently."""
        moved: dict[SourceId, SourceId] = {}
        for entry in entries:
            known = self._categories.setdefault(entry.source_id, entry)
            if known.name == entry.name:
                continue
            same_name = next((e.source_id for e in self._categories.values() if e.name == entry.name), None)
            new_id = same_name if same_name is not None else self._next_id()
            if same_name is None:
                self._categories[new_id] = replace(entry, source_id=new_id)
            moved[entry.source_id] = new_id
            logger.warning(
                "COCO category id %s is %r in %s but %r in an earlier annotation file; its boxes in %s get id %s",
                entry.source_id,
                entry.name,
                ann_path,
                known.name,
                ann_path,
                new_id,
            )
        return moved

    def _next_id(self) -> int:
        return max((i for i in self._categories if isinstance(i, int) and not isinstance(i, bool)), default=0) + 1

    def metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            taxonomy=Taxonomy(entries=tuple(self._categories.values()), source_dataset="coco", id_density="sparse"),
            source_dataset="coco",
            splits=tuple(self._splits),
            info=self._info or {},
            licenses=tuple(self._licenses),
        )


def _with_category_ids(
    sample: ImageObjectDetectionSample, moved: dict[SourceId, SourceId]
) -> ImageObjectDetectionSample:
    """``sample`` with each box whose category id is in ``moved`` given its new id."""
    detections = tuple(
        replace(box, category_id=moved[box.category_id], source_category_id=moved[box.category_id])
        if box.category_id in moved
        else box
        for box in sample.detections
    )
    return replace(sample, detections=detections)


# ``instances_<split>.json``: the per-split annotation file name (COCO 2017, CVAT, datamaite's writer).
_SPLIT_FILE_RE = re.compile(r"^instances_(?P<split>.+)\.json$")
_CVAT_NO_SUBSET = "default"


def _split_from_file_name(ann_path: Path) -> str | None:
    """The split an ``instances_<split>.json`` file holds, or None for any other name."""
    match = _SPLIT_FILE_RE.match(ann_path.name)
    split = match.group("split") if match else None
    return split if split and split.strip(".") else None


def _images_base_dir(ann_path: Path, split: str | None, raw_images: object, *, images_dir: Path | None) -> Path:
    """The directory this annotation file's ``file_name`` paths are relative to.

    Tries each candidate in turn and keeps the first that holds one of the
    file's first few images; with none found, the first candidate.
    """
    if images_dir is not None:
        candidates = [images_dir]
        if split is not None:
            candidates.append(images_dir / split)
    else:
        dataset_root = ann_path.parent.parent if ann_path.parent.name == "annotations" else ann_path.parent
        candidates = [dataset_root]
        if split is not None:
            candidates += [dataset_root / "images" / split, dataset_root / split]
        candidates.append(dataset_root / "images")
    if len(candidates) == 1:
        return candidates[0]
    names = [
        raw["file_name"]
        for raw in (raw_images if isinstance(raw_images, list) else [])
        if isinstance(raw, dict) and isinstance(raw.get("file_name"), str) and raw["file_name"].strip()
    ][:_IMAGES_DIR_PROBES]
    for candidate in candidates:
        for name in names:
            path = _resolve_image_path(candidate, name, warn=False)
            if path is not None and path.is_file():
                return candidate
    return candidates[0]


# How many of an annotation file's images are looked for when choosing its images directory.
_IMAGES_DIR_PROBES = 5


def _resolve_annotation_files(root: Path, annotation_file: str | Path | None) -> list[Path]:
    """The annotation files to load: an explicit file, ``root`` itself, or those discovered under it."""
    if annotation_file is not None:
        path = to_dataset_path(annotation_file)
        if not path.is_file():
            # An explicit argument naming a missing file is a caller mistake,
            # not malformed data -- raising beats returning an empty dataset.
            raise FileNotFoundError(f"COCO annotation file does not exist: {path}")
        return [path]
    if root.is_file():
        return [root]
    if not root.is_dir():
        return []
    ann_dir = root / "annotations"
    search_dirs = [ann_dir, root] if ann_dir.is_dir() else [root]
    for directory in search_dirs:
        # Every instances*.json is part of the dataset: one file per split
        # (instances_train2017.json, instances_val2017.json, ...).
        instances = sorted(directory.glob("instances*.json"))
        if instances:
            return instances
        candidates = sorted(directory.glob("*.json"))
        if len(candidates) > 1:
            # Without the instances naming there is no telling which files are
            # detection splits; picking one silently would misrepresent the dataset.
            logger.warning(
                "Multiple COCO annotation JSONs under %s; loading %s and ignoring %s. "
                "Pass annotation_file to select a specific one",
                directory,
                candidates[0].name,
                ", ".join(path.name for path in candidates[1:]),
            )
        if candidates:
            return candidates[:1]
    return []


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        logger.warning("Could not read COCO annotation file %s: %s", path, exc)
        return None
    if not isinstance(data, dict):
        logger.warning("Skipping COCO file whose top-level JSON is not an object: %s", path)
        return None
    return data


def _build_taxonomy(raw_categories: object, *, ann_path: Path) -> tuple[Taxonomy, dict[int, str]]:
    entries: list[CategoryEntry] = []
    names_by_id: dict[int, str] = {}
    for raw in raw_categories if isinstance(raw_categories, list) else []:
        if not isinstance(raw, dict):
            logger.warning("Skipping malformed COCO category in %s", ann_path)
            continue
        cat_id = _parse_int(raw.get("id"))
        name = raw.get("name")
        if cat_id is None or not isinstance(name, str) or not name.strip():
            logger.warning("Skipping COCO category with missing/invalid id or name in %s", ann_path)
            continue
        if cat_id in names_by_id:
            logger.warning("Skipping duplicate COCO category id %s in %s (keeping the first)", cat_id, ann_path)
            continue
        name = name.strip()
        names_by_id[cat_id] = name
        supercategory = raw.get("supercategory")
        entries.append(
            CategoryEntry(
                source_id=cat_id,
                name=name,
                supercategory=supercategory.strip()
                if isinstance(supercategory, str) and supercategory.strip()
                else None,
            )
        )
    return Taxonomy(entries=tuple(entries), source_dataset="coco", id_density="sparse"), names_by_id


def _annotations_by_image(
    raw_annotations: object,
    names_by_id: dict[int, str],
    *,
    ann_path: Path,
) -> dict[int, list[ObjectDetectionAnnotation]]:
    by_image: dict[int, list[ObjectDetectionAnnotation]] = defaultdict(list)
    for raw in raw_annotations if isinstance(raw_annotations, list) else []:
        parsed = _parse_annotation(raw, names_by_id, ann_path=ann_path)
        if parsed is not None:
            image_id, annotation = parsed
            by_image[image_id].append(annotation)
    return dict(by_image)


def _parse_annotation(
    raw: object,
    names_by_id: dict[int, str],
    *,
    ann_path: Path,
) -> tuple[int, ObjectDetectionAnnotation] | None:
    if not isinstance(raw, dict):
        logger.warning("Skipping malformed COCO annotation in %s", ann_path)
        return None
    image_id = _parse_int(raw.get("image_id"))
    bbox = _parse_bbox(raw.get("bbox"))
    if image_id is None or bbox is None:
        logger.warning("Skipping COCO annotation with missing/invalid image_id or bbox in %s", ann_path)
        return None
    category_id = _parse_int(raw.get("category_id"))
    if category_id is None:
        # Kept (best-effort), but it will surface as MAITE label -1, a sentinel
        # absent from index2label -- worth a trace in the log.
        logger.warning(
            "COCO annotation %r in %s has a missing/invalid category_id; it will carry no label",
            raw.get("id"),
            ann_path,
        )
    elif names_by_id and category_id not in names_by_id:
        # The id is kept verbatim (round-trip fidelity; -1 is reserved for "no
        # usable id"), but it will be absent from index2label, so MAITE metric
        # consumers indexing the label map need to know. Only meaningful when
        # categories[] is declared at all -- without it, ids are just opaque.
        logger.warning(
            "COCO annotation %r in %s references category id %s not defined in categories[]; "
            "it will be absent from index2label",
            raw.get("id"),
            ann_path,
            category_id,
        )
    attributes = {k: v for k, v in raw.items() if k not in _ANNOTATION_CORE_KEYS}
    annotation = ObjectDetectionAnnotation(
        bbox=bbox,
        category_id=category_id,
        category_name=names_by_id.get(category_id) if category_id is not None else None,
        source_category_id=category_id,
        source_annotation_id=_parse_int(raw.get("id")),
        area=_parse_float(raw.get("area")),
        segmentation=raw.get("segmentation") if raw.get("segmentation") not in (None, []) else None,
        iscrowd=_parse_iscrowd(raw.get("iscrowd")),
        attributes=attributes,
    )
    return image_id, annotation


def _build_samples(
    raw_images: object,
    boxes_by_image: dict[int, list[ObjectDetectionAnnotation]],
    *,
    base_dir: Path,
    ann_path: Path,
) -> list[ImageObjectDetectionSample]:
    samples: list[ImageObjectDetectionSample] = []
    seen_ids: set[int] = set()
    turned: list[str] = []
    for raw in raw_images if isinstance(raw_images, list) else []:
        if not isinstance(raw, dict):
            logger.warning("Skipping malformed COCO image in %s", ann_path)
            continue
        image_id = _parse_int(raw.get("id"))
        file_name = raw.get("file_name")
        if image_id is None or not isinstance(file_name, str) or not file_name.strip():
            logger.warning("Skipping COCO image with missing/invalid id or file_name in %s", ann_path)
            continue
        if image_id in seen_ids:
            # A second images[] entry with the same id would receive the same
            # detections again, doubling ground truth on re-export.
            logger.warning("Skipping duplicate COCO image id %s in %s (keeping the first)", image_id, ann_path)
            continue
        seen_ids.add(image_id)
        path = _resolve_image_path(base_dir, file_name)
        width = _coerce_positive_int(raw.get("width"))
        height = _coerce_positive_int(raw.get("height"))
        # Only treat width/height as consumed when they parsed; an invalid
        # source value stays visible in metadata instead of vanishing.
        consumed = {"id", "file_name"}
        if width is not None:
            consumed.add("width")
        if height is not None:
            consumed.add("height")
        image_metadata = {k: v for k, v in raw.items() if k not in consumed}
        detections = tuple(boxes_by_image.get(image_id, []))
        displayed = _exif_display_frame(path, width, height, detections)
        if displayed is not None:
            width, height, detections = displayed
            turned.append(file_name)
        samples.append(
            ImageObjectDetectionSample(
                image_id=image_id,
                path_or_uri=str(path) if path is not None else None,
                file_name=file_name,
                width=width,
                height=height,
                detections=detections,
                metadata=image_metadata,
            )
        )
    if turned:
        logger.warning(
            "%d COCO image(s) in %s are EXIF-rotated JPEGs whose stored width/height are the unrotated size, "
            "so their sizes and boxes were turned to the displayed frame that decoding produces: %s",
            len(turned),
            ann_path,
            ", ".join(turned[:5]) + (", ..." if len(turned) > 5 else ""),
        )
    return samples


_JPEG_SUFFIXES = frozenset({".jpg", ".jpeg", ".jpe", ".jfif"})


def _exif_display_frame(
    path: Path | None, width: int | None, height: int | None, detections: tuple[ObjectDetectionAnnotation, ...]
) -> tuple[int, int, tuple[ObjectDetectionAnnotation, ...]] | None:
    """The size and boxes in the displayed frame, for a record written in an EXIF-rotated JPEG's stored frame (#133).

    Decoding applies the EXIF orientation, so a tool that ignored it wrote the
    size and boxes in the wrong frame. That is only detectable when the
    orientation swaps the axes (5-8) and the record's size is the stored,
    unrotated one; orientations 2-4 keep the size and are left alone. Returns
    ``None`` when nothing changes.
    """
    if path is None or width is None or height is None or width == height:
        return None
    if path.suffix.lower() not in _JPEG_SUFFIXES:
        return None
    try:
        header = read_image_header(path)
    except OSError:
        return None  # a missing image is reported where it is decoded
    if header is None or not header.transposed or (header.width, header.height) != (width, height):
        return None
    turn = _orientation_transform(header.orientation, width, height)
    return height, width, tuple(_turn_detection(detection, turn) for detection in detections)


def _orientation_transform(orientation: int, width: int, height: int) -> Callable[[float, float], tuple[float, float]]:
    """Map a point in a ``width`` x ``height`` stored frame to the frame EXIF ``orientation`` 5-8 displays."""
    if orientation == 5:  # transpose
        return lambda x, y: (y, x)
    if orientation == 6:  # turn 90 degrees clockwise
        return lambda x, y: (height - y, x)
    if orientation == 7:  # transverse
        return lambda x, y: (height - y, width - x)
    return lambda x, y: (y, width - x)  # 8: turn 90 degrees anticlockwise


def _turn_detection(
    detection: ObjectDetectionAnnotation, turn: Callable[[float, float], tuple[float, float]]
) -> ObjectDetectionAnnotation:
    x, y, w, h = detection.bbox
    (x1, y1), (x2, y2) = turn(x, y), turn(x + w, y + h)
    bbox = (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
    segmentation = detection.segmentation
    if isinstance(segmentation, list) and all(isinstance(polygon, list) for polygon in segmentation):
        segmentation = [_turn_polygon(polygon, turn) for polygon in segmentation]
    elif segmentation is not None:
        # RLE masks are run-length coded over the stored frame; turning one means decoding it.
        logger.warning(
            "COCO annotation %r: the RLE segmentation stays in the stored (unrotated) frame of its "
            "EXIF-rotated image; only its box was turned",
            detection.source_annotation_id,
        )
    return replace(detection, bbox=bbox, segmentation=segmentation)


def _turn_polygon(polygon: list[Any], turn: Callable[[float, float], tuple[float, float]]) -> list[Any]:
    if len(polygon) % 2 or not all(isinstance(v, int | float) and not isinstance(v, bool) for v in polygon):
        return polygon
    turned: list[Any] = []
    for index in range(0, len(polygon), 2):
        turned.extend(turn(polygon[index], polygon[index + 1]))
    return turned


def _resolve_image_path(base_dir: Path, file_name: str, *, warn: bool = True) -> Path | None:
    """Resolve a COCO ``file_name`` under ``base_dir``, rejecting names that escape it.

    ``:`` and ``\\`` alone do not escape: timestamped frames use ``12:00:00.png``
    and Windows exports use ``cam1\\a.png``. A ``\\`` name is read as written when
    that file exists (POSIX names may contain ``\\``), else as a Windows path.
    """
    if "://" in file_name or _escapes_root(file_name):
        if warn:
            logger.warning("Skipping COCO image with unsafe file_name %r", file_name)
        return None
    name = file_name.strip()
    path = base_dir.joinpath(*PurePosixPath(name).parts)
    if "\\" in name and not path.exists():
        return base_dir.joinpath(*PurePosixPath(name.replace("\\", "/")).parts)
    return path


def _escapes_root(file_name: str) -> bool:
    """Whether ``file_name`` points outside the dataset: absolute, a drive, or ``..``.

    Backslashes count as separators here, so ``..\\x.jpg`` and ``C:\\x.jpg`` are
    caught; a ``\\`` or ``:`` alone does not make a name escape. Any one-letter
    ``<letter>:`` prefix is a drive, even without a separator after it:
    Windows reads ``a:b.png`` as relative to drive ``a:``, not to the images
    directory, so joining it onto that directory would discard the directory.
    """
    normalized = file_name.strip().replace("\\", "/")
    posix = PurePosixPath(normalized)
    return (
        not posix.parts
        or posix.is_absolute()
        or ".." in posix.parts
        or (len(normalized) >= 2 and normalized[1] == ":" and normalized[0].isalpha())
    )


def _parse_bbox(value: object) -> tuple[float, float, float, float] | None:
    """Parse a COCO ``[x, y, w, h]`` bbox (absolute pixels) with positive w/h."""
    if not isinstance(value, list | tuple) or len(value) != 4:
        return None
    x = _parse_float(value[0])
    y = _parse_float(value[1])
    w = _parse_float(value[2])
    h = _parse_float(value[3])
    if x is None or y is None or w is None or h is None:
        return None
    if w <= 0 or h <= 0:
        return None
    return (x, y, w, h)


def _parse_iscrowd(value: object) -> int:
    # COCO uses integer 0/1, but tolerate a JSON boolean rather than silently
    # flipping a truthy crowd flag to 0 (``_parse_int`` rejects bools by design).
    if isinstance(value, bool):
        return int(value)
    return _parse_int(value) or 0


def _parse_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        # Fast path that also keeps ids above 2**53 exact -- the float
        # round-trip below would silently round them to a colliding value.
        return value
    number = _parse_float(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _parse_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _coerce_positive_int(value: object) -> int | None:
    parsed = _parse_int(value)
    return parsed if parsed is not None and parsed > 0 else None
