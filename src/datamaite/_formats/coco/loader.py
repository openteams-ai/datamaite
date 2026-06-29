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
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar

from datamaite._types import DatasetFormat, Task
from datamaite.loaders import Loader, register_loader
from datamaite.object_detection import ObjectDetectionDataset
from datamaite.records import DatasetMetadata, ImageObjectDetectionSample, ObjectDetectionAnnotation
from datamaite.taxonomy import CategoryEntry, Taxonomy

logger = logging.getLogger(__name__)

# Keys consumed as typed fields; everything else on an annotation rides in attributes.
_ANNOTATION_CORE_KEYS = frozenset({"id", "image_id", "category_id", "bbox", "area", "iscrowd", "segmentation"})


@register_loader
class CocoLoader(Loader):
    """Loader for COCO detection datasets. Produces an ``ObjectDetectionDataset``."""

    task: ClassVar[Task] = Task.OD
    format: ClassVar[DatasetFormat] = DatasetFormat.COCO

    def load(self, root: str | Path, **options: Any) -> ObjectDetectionDataset:
        return load_coco(root, **options)


def load_coco(
    root: str | Path,
    *,
    annotation_file: str | Path | None = None,
    images_dir: str | Path | None = None,
) -> ObjectDetectionDataset:
    """Load a COCO detection dataset into an :class:`ObjectDetectionDataset`.

    Parameters
    ----------
    root
        Either the COCO annotation JSON file itself, or a directory containing
        it (``annotations/instances_*.json`` or any top-level ``*.json``).
    annotation_file
        Explicit annotation JSON path, overriding discovery under ``root``. A
        relative path anchors to ``root`` (not the caller's CWD), matching the
        other loaders' override style. Raises ``FileNotFoundError`` when it
        does not exist -- a wrong user argument, unlike malformed data, is not
        a best-effort case.
    images_dir
        Directory the ``images[].file_name`` paths are relative to; a relative
        path anchors to ``root``. Defaults to the annotation file's parent --
        or its grandparent when the file lives in an ``annotations/``
        subdirectory (the standard COCO layout, where file names are relative
        to the dataset root).
    """
    root = Path(root)
    # Relative override paths anchor to ``root`` (its parent when root is the
    # annotation file itself), never the process CWD. ``anchor / p`` leaves
    # absolute overrides intact.
    anchor = root.parent if root.is_file() else root
    if annotation_file is not None:
        annotation_file = anchor / Path(annotation_file)
    ann_path = _resolve_annotation_file(root, annotation_file)
    if ann_path is None:
        logger.warning("No COCO annotation JSON found under %s", root)
        return ObjectDetectionDataset(samples=(), dataset_metadata=DatasetMetadata(source_dataset="coco"))

    data = _read_json(ann_path)
    if data is None:
        return ObjectDetectionDataset(samples=(), dataset_metadata=DatasetMetadata(source_dataset="coco"))
    if any(key in data for key in ("videos", "tracks")):
        # A TAO-style video-dataset JSON also carries images/annotations/
        # categories, so it would otherwise "load" with tracks silently
        # flattened into independent per-image boxes.
        logger.warning(
            "%s contains video-dataset keys (videos/tracks); this looks like a TAO-style video "
            "dataset, not still-image COCO. Use load_mot(..., dataset_format='tao') instead",
            ann_path,
        )

    base_dir = (
        anchor / Path(images_dir)
        if images_dir is not None
        else ann_path.parent.parent
        if ann_path.parent.name == "annotations"
        else ann_path.parent
    )

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

    meta = DatasetMetadata(
        taxonomy=taxonomy,
        source_dataset="coco",
        info=dict(data["info"]) if isinstance(data.get("info"), dict) else {},
        licenses=tuple(lic for lic in data.get("licenses", []) if isinstance(lic, dict)),
    )
    logger.info("Loaded %d COCO image(s), %d categories from %s", len(samples), len(names_by_id), ann_path)
    return ObjectDetectionDataset(samples=tuple(samples), dataset_metadata=meta)


def _resolve_annotation_file(root: Path, annotation_file: str | Path | None) -> Path | None:
    if annotation_file is not None:
        path = Path(annotation_file)
        if not path.is_file():
            # An explicit argument naming a missing file is a caller mistake,
            # not malformed data -- raising beats returning an empty dataset.
            raise FileNotFoundError(f"COCO annotation file does not exist: {path}")
        return path
    if root.is_file():
        return root
    if not root.is_dir():
        return None
    ann_dir = root / "annotations"
    search_dirs = [ann_dir, root] if ann_dir.is_dir() else [root]
    for directory in search_dirs:
        # Prefer instances_*.json (the detection split), else any json.
        candidates = sorted(directory.glob("instances*.json")) or sorted(directory.glob("*.json"))
        if len(candidates) > 1:
            # A real COCO root often has several splits (instances_train2017,
            # instances_val2017, ...); picking one silently would misrepresent
            # the dataset.
            logger.warning(
                "Multiple COCO annotation JSONs under %s; loading %s and ignoring %s. "
                "Pass annotation_file to select a specific one",
                directory,
                candidates[0].name,
                ", ".join(path.name for path in candidates[1:]),
            )
        if candidates:
            return candidates[0]
    return None


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
        samples.append(
            ImageObjectDetectionSample(
                image_id=image_id,
                path_or_uri=str(path) if path is not None else None,
                file_name=file_name,
                width=width,
                height=height,
                detections=tuple(boxes_by_image.get(image_id, [])),
                metadata=image_metadata,
            )
        )
    return samples


def _resolve_image_path(base_dir: Path, file_name: str) -> Path | None:
    """Resolve a COCO ``file_name`` safely under ``base_dir`` (reject path escapes)."""
    if "\\" in file_name:
        logger.warning("Skipping COCO image with unsafe file_name %r", file_name)
        return None
    posix = PurePosixPath(file_name.strip())
    if not posix.parts or posix.is_absolute() or any(part in {"..", ""} or ":" in part for part in posix.parts):
        logger.warning("Skipping COCO image with unsafe file_name %r", file_name)
        return None
    return base_dir.joinpath(*posix.parts)


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
