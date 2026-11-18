"""Compare two dataset snapshots and list every difference.

:func:`compare` returns a list of readable differences (empty when the
snapshots match); :func:`assert_same` fails a test with all of them. A
difference that a format cannot avoid is only ignored when the caller passes
the matching :class:`Loss` in ``allowed``. For data that passed through several
formats, ``allowed`` is one set of losses per format, in order, usually from
:func:`tests.harness.loss_stages`.

Images pair by key (normalised file name) and split, boxes by class, crowd
flag and position, and classes by id. Every field of each pair is then
compared, the fields used for pairing included, and so is every field of the
dataset, its metadata and its taxonomy.
"""

from __future__ import annotations

import enum
import math
from collections import defaultdict
from collections.abc import Callable, Collection, Iterable, Sequence
from dataclasses import replace
from pathlib import PurePosixPath
from typing import Any

from datamaite.records import ImageObjectDetectionSample, ObjectDetectionAnnotation
from datamaite.taxonomy import CategoryEntry

from .snapshot import (
    BOX_TOLERANCE,
    AnyValue,
    BoxSnapshot,
    DatasetSnapshot,
    Near,
    Record,
    SampleSnapshot,
    edit,
    field_defaults,
    remove,
)

# Every number other than box geometry and integers (which are exact) is equal only within
# this relative tolerance, with a tiny absolute floor so that 0.0 and -0.0 or 1e-300 agree.
REL_TOL = 1e-6
ABS_TOL = 1e-9

#: Fields in pixel coordinates, equal within :data:`BOX_TOLERANCE`. Polygon
#: points in ``segmentation`` are pixel coordinates like ``bbox``, and a format
#: that normalises or rounds boxes (YOLO's six decimals) would round polygons
#: the same way; RLE counts and sizes are integers, so the tolerance leaves them exact.
GEOMETRY_FIELDS = frozenset({"bbox", "segmentation"})


class Loss(enum.Enum):
    """A difference a format allows, so a comparison may ignore it.

    Each format says why it allows each one (``tests/harness/formats.py``): the
    format cannot store the value, the format makes the value up, or datamaite
    chose not to keep a value the format could hold.
    """

    #: Source image ids are not kept (YOLO ids are file paths).
    IMAGE_IDS = "image_ids"
    #: Class ids are renumbered 0..n-1 in taxonomy order. Names must still match.
    LABEL_IDS = "label_ids"
    #: Split names are normalised: ``validation``/``valid`` become ``val``, ``training`` becomes ``train``,
    #: ignoring case (CVAT's ``Train`` becomes ``train``).
    SPLIT_ALIASES = "split_aliases"
    #: File names are relative to the split's image folder: a leading ``images/``
    #: and a folder whose name is the image's split (``train/a.jpg`` in split
    #: ``train``, ``Valid/b.jpg`` in split ``val``) are dropped.
    SPLIT_FOLDER_IN_FILE_NAME = "split_folder_in_file_name"
    #: A label file is named after its image without the extension, so of images in one split
    #: whose names differ only in it (``a.png``, ``a.jpg``), the writer must rename all but one.
    LABEL_FILE_STEMS = "label_file_stems"
    #: Images with no split are written to ``train``.
    DEFAULT_SPLIT = "default_split"
    #: Classes have only an id and a name: no supercategory, synset, attributes or other details.
    LABEL_DETAILS = "label_details"
    #: Classes have no flag for being left out of evaluation, so they come back included.
    CLASS_EVAL_EXCLUDED = "class_eval_excluded"
    #: Classes do not record the dataset they came from (set when taxonomies merge).
    CLASS_SOURCE_DATASET = "class_source_dataset"
    #: Boxes have no crowd flag, so crowd boxes come back as ordinary boxes.
    CROWD = "crowd"
    #: Classes that no box uses are dropped.
    UNUSED_LABELS = "unused_labels"
    #: Boxes store a class name, not an id, so the ids are rebuilt on reload. A
    #: class whose name is its id as text (``2: "2"``) keeps its id; the others
    #: are numbered after those ids, in case-insensitive name order. The classes
    #: come back in id order.
    LABEL_IDS_FROM_NAMES = "label_ids_from_names"
    #: A box's source class id is not stored apart from its class id, so it comes back as the class id.
    SOURCE_CATEGORY_IDS = "source_category_ids"
    #: Boxes have no confidence score.
    SCORE = "score"
    #: Boxes have no segmentation (polygons or RLE masks).
    SEGMENTATION = "segmentation"
    #: Boxes have no area.
    AREA = "area"
    #: A box with no area gets its box's ``width * height``.
    AREA_FROM_BOX = "area_from_box"
    #: Boxes have no attributes.
    BOX_ATTRIBUTES = "box_attributes"
    #: Boxes have no annotation ids.
    ANNOTATION_IDS = "annotation_ids"
    #: A box with no annotation id gets a new integer id.
    NEW_ANNOTATION_IDS = "new_annotation_ids"
    #: Images have no metadata.
    IMAGE_METADATA = "image_metadata"
    #: The dataset's split list is its images' splits, in name order.
    SPLIT_LIST_BY_NAME = "split_list_by_name"
    #: The dataset's split list is its images' splits: train, val (or validation) and test first, then the rest by name.
    SPLIT_LIST_FROM_IMAGES = "split_list_from_images"
    #: The dataset has no ``info``.
    INFO = "info"
    #: The dataset has no ``licenses``.
    LICENSES = "licenses"
    #: The dataset's ``extras`` are not kept.
    DATASET_EXTRAS = "dataset_extras"
    #: The YOLO loader records where each image and box came from: image metadata
    #: ``source_format``, ``variant``, ``source_file_name`` and ``label_file``, and box
    #: attributes ``yolo_bbox`` (the normalised row) and ``source_line``.
    YOLO_SOURCE_DETAILS = "yolo_source_details"
    #: The Hugging Face loader records where each image came from: image metadata
    #: ``source_format``, ``source_file_name`` and ``metadata_file``.
    HUGGINGFACE_SOURCE_DETAILS = "huggingface_source_details"
    #: The taxonomy's ``id_density`` is not stored, so it comes back ``'sparse'``.
    ID_DENSITY = "id_density"
    #: The taxonomy's ``id_density`` is worked out from its class ids: ``'dense'``
    #: when they are ``0..n-1`` in taxonomy order, else ``'sparse'``.
    ID_DENSITY_FROM_IDS = "id_density_from_ids"


SPLIT_ALIASES = {"training": "train", "validation": "val", "valid": "val"}
DEFAULT_SPLIT = "train"
SPLIT_ORDER = {"train": 0, "val": 1, "validation": 1, "test": 2}

_BOXES = "samples[*].detections[*]"
_ENTRIES = "dataset_metadata.taxonomy.entries"


#: The losses a comparison allows: one set, or one set per format the data
#: passed through, in order.
Allowed = Collection[Loss] | Sequence[Collection[Loss]]


def compare(expected: DatasetSnapshot, actual: DatasetSnapshot, *, allowed: Allowed = ()) -> list[str]:
    """Every difference between ``expected`` and ``actual``, plus ``actual``'s own problems.

    Each allowed loss turns ``expected`` into what the format can give back,
    for example class ids renumbered ``0..n-1``. With one set of losses per
    format, each format's losses apply in turn, so a conversion gives back what
    the second format makes of the first one's output. ``actual`` is never
    changed, so a loss can't hide a change the format did not need to make.
    """
    expected = expected_after_losses(expected, allowed)
    differences = [f"problem: {problem}" for problem in actual.problems]
    differences += _compare_labels(expected.labels, actual.labels)
    differences += _compare_classes(expected.entries, actual.entries)
    differences += record_differences("dataset", _without_entries(expected.fields), _without_entries(actual.fields))
    differences += _compare_samples(expected.samples, actual.samples)
    return differences


def assert_same(expected: DatasetSnapshot, actual: DatasetSnapshot, *, allowed: Allowed = ()) -> None:
    """Fail with every difference between ``expected`` and ``actual``."""
    differences = compare(expected, actual, allowed=allowed)
    if differences:
        lines = "\n".join(f"  - {difference}" for difference in differences)
        raise AssertionError(f"{len(differences)} difference(s) from the expected dataset:\n{lines}")


def without_classes(snap: DatasetSnapshot) -> DatasetSnapshot:
    """``snap`` with no taxonomy and no class on any box: only the boxes' places and other fields are left."""
    tree = snap.tree
    tree["dataset_metadata"]["taxonomy"] = None
    for field in ("category_id", "category_name", "source_category_id"):
        edit(tree, f"samples[*].detections[*].{field}", lambda _value: None)
    return snap.with_tree(tree)


def without_class_names(snap: DatasetSnapshot) -> DatasetSnapshot:
    """``snap`` with no taxonomy and no class name on any box: the boxes keep their class ids."""
    tree = snap.tree
    tree["dataset_metadata"]["taxonomy"] = None
    edit(tree, "samples[*].detections[*].category_name", lambda _value: None)
    return snap.with_tree(tree)


# ---------------------------------------------------------------------------
# Normalisation for allowed losses
# ---------------------------------------------------------------------------


def expected_after_losses(expected: DatasetSnapshot, allowed: Allowed) -> DatasetSnapshot:
    """``expected`` as the formats with these losses must give it back, each format's losses in turn."""
    for stage in _stages(allowed):
        expected = _expected_after_losses(expected, stage)
    return expected


def _stages(allowed: Allowed) -> tuple[frozenset[Loss], ...]:
    items = list(allowed)
    if all(isinstance(item, Loss) for item in items):
        return (frozenset(items),)  # type: ignore[arg-type]
    if any(isinstance(item, Loss) for item in items):
        raise TypeError("allowed is a set of losses or a sequence of sets, not a mix")
    return tuple(frozenset(stage) for stage in items)  # type: ignore[arg-type]


def _expected_after_losses(expected: DatasetSnapshot, allowed: frozenset[Loss]) -> DatasetSnapshot:
    """``expected`` as a format with these losses must give it back. Every change is an edit of its tree."""
    tree = expected.tree
    _class_fields_after_losses(tree, allowed)
    if Loss.CROWD in allowed:
        edit(tree, f"{_BOXES}.iscrowd", lambda _: 0)
    _class_ids_after_losses(tree, allowed)
    if Loss.IMAGE_IDS in allowed:
        edit(tree, "samples[*].image_id", lambda _: AnyValue(object))
    for sample in tree["samples"]:
        _image_fields_after_losses(sample, allowed)
        for box in sample.get("detections", ()):
            _box_fields_after_losses(box, sample, allowed)
    _image_splits_after_losses(tree, allowed)
    _file_names_after_losses(tree, allowed)
    _dataset_fields_after_losses(tree, allowed)
    return expected.with_tree(tree)


def _image_splits_after_losses(tree: dict[str, Any], allowed: frozenset[Loss]) -> None:
    """Each image's split, and its file name's split folder, after the losses."""
    if Loss.SPLIT_FOLDER_IN_FILE_NAME in allowed:
        # The writer drops the folder before it renames the split.
        for sample in tree["samples"]:
            sample["file_name"] = _without_split_folder(sample.get("file_name"), sample.get("split") or DEFAULT_SPLIT)
    if Loss.SPLIT_ALIASES in allowed:
        edit(tree, "samples[*].split", _split_alias)
    if Loss.DEFAULT_SPLIT in allowed:
        edit(tree, "samples[*].split", lambda split: split or DEFAULT_SPLIT)


def _class_ids_after_losses(tree: dict[str, Any], allowed: frozenset[Loss]) -> None:
    """The classes and the boxes' class ids after the losses that drop or renumber classes."""
    if Loss.UNUSED_LABELS in allowed:
        used = {box.get("category_id") for sample in tree["samples"] for box in sample.get("detections", ())}
        edit(tree, _ENTRIES, lambda entries: [entry for entry in entries if entry.get("source_id") in used])
    if Loss.LABEL_IDS_FROM_NAMES in allowed:
        by_name = _ranks_from_names(_entries(tree))
        _renumber(tree, by_name)
        # The reload lists its classes by their rebuilt ids.
        edit(tree, _ENTRIES, lambda entries: sorted(entries, key=lambda entry: _id_sort_key(entry.get("source_id"))))
    if Loss.LABEL_IDS in allowed:
        _renumber(tree, _dense_ranks(tree))


def _renumber(tree: dict[str, Any], rank: dict[Any, Any]) -> None:
    """Replace each class id in the classes and the boxes with ``rank``'s, where it has one."""
    edit(tree, f"{_ENTRIES}[*].source_id", lambda label_id: rank.get(label_id, label_id))
    edit(tree, f"{_BOXES}.category_id", lambda label_id: rank.get(label_id, label_id))


def _file_names_after_losses(tree: dict[str, Any], allowed: frozenset[Loss]) -> None:
    """Each image's file name after the losses: any name, for images the format must rename.

    A renamed image is still paired with its own by its pixels, and its boxes
    and every other field are compared as usual.
    """
    if Loss.LABEL_FILE_STEMS in allowed:
        by_label_file: dict[tuple[Any, str], list[dict[str, Any]]] = defaultdict(list)
        for sample in tree["samples"]:
            name = sample.get("file_name")
            if isinstance(name, str):
                by_label_file[(sample.get("split"), PurePosixPath(name).with_suffix("").as_posix())].append(sample)
        for samples in by_label_file.values():
            if len(samples) > 1:
                for sample in samples:
                    sample["file_name"] = AnyValue(str)


def _class_fields_after_losses(tree: dict[str, Any], allowed: frozenset[Loss]) -> None:
    """The taxonomy's class entries after the losses; their ids are renumbered separately."""
    defaults = field_defaults(CategoryEntry)
    if Loss.LABEL_DETAILS in allowed:
        edit(
            tree,
            f"{_ENTRIES}[*]",
            lambda entry: Record({k: v if k in ("source_id", "name") else defaults[k] for k, v in entry.items()}),
        )
    if Loss.CLASS_EVAL_EXCLUDED in allowed:
        edit(tree, f"{_ENTRIES}[*].eval_excluded", lambda _: defaults["eval_excluded"])
    if Loss.CLASS_SOURCE_DATASET in allowed:
        edit(tree, f"{_ENTRIES}[*].source_dataset", lambda _: defaults["source_dataset"])


def _dataset_fields_after_losses(tree: dict[str, Any], allowed: frozenset[Loss]) -> None:
    """The dataset's own fields after the losses; its samples already have their splits as the format gives them."""
    splits = {s.get("split") for s in tree["samples"]} - {None}
    if Loss.SPLIT_LIST_BY_NAME in allowed:
        edit(tree, "dataset_metadata.splits", lambda _: sorted(splits))
    if Loss.SPLIT_LIST_FROM_IMAGES in allowed:
        listed = sorted(splits, key=lambda split: (SPLIT_ORDER.get(split, len(SPLIT_ORDER)), split))
        edit(tree, "dataset_metadata.splits", lambda _: listed)
    if Loss.INFO in allowed:
        edit(tree, "dataset_metadata.info", lambda _: {})
    if Loss.LICENSES in allowed:
        edit(tree, "dataset_metadata.licenses", lambda _: [])
    if Loss.DATASET_EXTRAS in allowed:
        edit(tree, "dataset_metadata.extras", lambda _: {})
    if Loss.ID_DENSITY in allowed:
        edit(tree, "dataset_metadata.taxonomy.id_density", lambda _: "sparse")
    if Loss.ID_DENSITY_FROM_IDS in allowed:
        ids = [entry.get("source_id") for entry in _entries(tree)]
        density = "dense" if ids == list(range(len(ids))) else "sparse"
        edit(tree, "dataset_metadata.taxonomy.id_density", lambda _: density)


def _image_fields_after_losses(sample: dict[str, Any], allowed: frozenset[Loss]) -> None:
    if Loss.IMAGE_METADATA in allowed:
        sample["metadata"] = field_defaults(ImageObjectDetectionSample)["metadata"]
    if Loss.YOLO_SOURCE_DETAILS in allowed:
        # source_file_name is the image's path under the root, as is its YOLO image id.
        sample["metadata"] = {
            **sample.get("metadata", {}),
            "source_format": "yolo",
            "variant": "default",
            "source_file_name": AnyValue(str),
            "label_file": AnyValue(str),
        }
    if Loss.HUGGINGFACE_SOURCE_DETAILS in allowed:
        # metadata_file is the absolute path of the metadata.jsonl or .parquet the row came from.
        sample["metadata"] = {
            **sample.get("metadata", {}),
            "source_format": "huggingface_vision",
            "source_file_name": AnyValue(str),
            "metadata_file": AnyValue(str),
        }


def _box_fields_after_losses(box: dict[str, Any], sample: dict[str, Any], allowed: frozenset[Loss]) -> None:
    defaults = field_defaults(ObjectDetectionAnnotation)
    if Loss.SOURCE_CATEGORY_IDS in allowed:
        box["source_category_id"] = box.get("category_id")
    for loss, name in (
        (Loss.SCORE, "score"),
        (Loss.SEGMENTATION, "segmentation"),
        (Loss.AREA, "area"),
        (Loss.BOX_ATTRIBUTES, "attributes"),
        (Loss.ANNOTATION_IDS, "source_annotation_id"),
    ):
        if loss in allowed:
            box[name] = defaults[name]
    if Loss.AREA_FROM_BOX in allowed and box.get("area") is None:
        # The box's width and height may each come back within BOX_TOLERANCE, so its area
        # within (w + t)(h + t) - wh = t(w + h) + t².
        _x, _y, w, h = box["bbox"]
        box["area"] = Near(w * h, BOX_TOLERANCE * (w + h) + BOX_TOLERANCE**2)
    if Loss.NEW_ANNOTATION_IDS in allowed and box.get("source_annotation_id") is None:
        box["source_annotation_id"] = AnyValue(int)
    if Loss.YOLO_SOURCE_DETAILS in allowed:
        box["attributes"] = {
            **box.get("attributes", {}),
            "yolo_bbox": _normalised_box(box["bbox"], sample.get("width"), sample.get("height")),
            "source_line": AnyValue(int),
        }


def _normalised_box(bbox: list[float], width: Any, height: Any) -> Any:
    """A YOLO label row's ``[centre x, centre y, width, height]``, as fractions of the image size.

    Each fraction is known within what the box tolerance allows: a centre moves
    by up to one and a half times it (corner plus half the size), a size by up to once.
    """
    if not width or not height:
        return AnyValue(list)
    x, y, w, h = bbox
    return [
        Near((x + w / 2) / width, 1.5 * BOX_TOLERANCE / width),
        Near((y + h / 2) / height, 1.5 * BOX_TOLERANCE / height),
        Near(w / width, BOX_TOLERANCE / width),
        Near(h / height, BOX_TOLERANCE / height),
    ]


def _split_alias(split: str | None) -> str | None:
    """``split`` as a format that normalises split names gives it back."""
    if not split:
        return None
    lowered = split.lower()
    if lowered in SPLIT_ALIASES:
        return SPLIT_ALIASES[lowered]
    return lowered if lowered in SPLIT_ALIASES.values() else split


def _without_split_folder(file_name: Any, split: str) -> Any:
    """``file_name`` without a leading ``images/`` or a folder named for ``split``, as YOLO writes it."""
    if not isinstance(file_name, str):
        return file_name
    parts = list(PurePosixPath(file_name).parts)
    if parts and parts[0] == "images":
        parts.pop(0)
    if parts and split in ("train", "val", "test") and _split_alias(parts[0]) == split:
        parts.pop(0)
    if parts and parts[0] == "images":
        parts.pop(0)
    stripped = PurePosixPath(*parts).as_posix() if parts else file_name
    # A name the writer keeps is kept as it is (``./a.png`` too).
    return stripped if len(parts) < len(PurePosixPath(file_name).parts) else file_name


def _ranks_from_names(entries: list[dict[str, Any]]) -> dict[Any, Any]:
    """Each class id as a reload rebuilds it from the class names (see ``Loss.LABEL_IDS_FROM_NAMES``)."""
    labels = {entry.get("source_id"): entry.get("name") for entry in entries}
    kept = [label_id for label_id, name in labels.items() if isinstance(label_id, int) and name == str(label_id)]
    named = sorted((i for i in labels if i not in kept), key=lambda i: (labels[i].casefold(), labels[i]))
    start = max(kept, default=-1) + 1
    rank: dict[Any, Any] = {label_id: label_id for label_id in kept}
    rank.update({label_id: start + offset for offset, label_id in enumerate(named)})
    return rank


def _entries(tree: dict[str, Any]) -> list[dict[str, Any]]:
    taxonomy = (tree.get("dataset_metadata") or {}).get("taxonomy")
    return list(taxonomy.get("entries", ())) if isinstance(taxonomy, dict) else []


def _dense_ranks(tree: dict[str, Any]) -> dict[Any, int]:
    """Each class id's position in ``0..n-1``.

    The labels come first, in their (taxonomy) order, which is the order a YOLO
    writer must follow. Class ids used by boxes but missing from the labels
    follow in numeric order, the fallback for a dataset with no taxonomy.
    """
    labels = list(dict.fromkeys(entry.get("source_id") for entry in _entries(tree)))
    box_ids = {box.get("category_id") for sample in tree["samples"] for box in sample.get("detections", ())}
    unlisted = box_ids - {None} - set(labels)
    ordered = [*labels, *sorted(unlisted, key=_id_sort_key)]
    return {label_id: index for index, label_id in enumerate(ordered)}


def _without_entries(fields: dict[str, Any]) -> dict[str, Any]:
    """The dataset's own fields without its class entries, which are compared class by class."""
    tree = dict(fields)
    if isinstance((tree.get("dataset_metadata") or {}).get("taxonomy"), dict):
        tree["dataset_metadata"] = Record(tree["dataset_metadata"])
        tree["dataset_metadata"]["taxonomy"] = Record(tree["dataset_metadata"]["taxonomy"])
        remove(tree, _ENTRIES)
    return tree


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _close(want: float, got: float) -> bool:
    return math.isclose(want, got, rel_tol=REL_TOL, abs_tol=ABS_TOL)


def _close_in_pixels(want: float, got: float) -> bool:
    return want == got or abs(want - got) <= BOX_TOLERANCE


def same_value(want: Any, got: Any, close: Callable[[float, float], bool] = _close) -> bool:
    """Whether canonical ``got`` matches ``want``, recursively.

    Two integers are equal only if they are equal, so ids are exact: 10000000
    is not 10000001, though they are within :data:`REL_TOL`. Box geometry
    compares in pixels, integers included. Any other numbers are ``close``,
    NaN equals NaN, and True is not 1.
    """
    if isinstance(want, AnyValue):
        return isinstance(got, want.kind) and not (want.kind is int and isinstance(got, bool))
    if isinstance(want, Near):
        return _is_number(got) and abs(got - want.value) <= want.abs_tol
    if _is_number(want) and _is_number(got):
        if isinstance(want, int) and isinstance(got, int) and close is _close:
            return want == got
        if math.isnan(want) or math.isnan(got):
            return math.isnan(want) and math.isnan(got)
        return close(want, got)
    if isinstance(want, list) and isinstance(got, list):
        return len(want) == len(got) and all(same_value(w, g, close) for w, g in zip(want, got, strict=True))
    if isinstance(want, dict) and isinstance(got, dict):
        return want.keys() == got.keys() and all(same_value(want[key], got[key], close) for key in want)
    if isinstance(want, bool) or isinstance(got, bool):
        return type(want) is type(got) and want == got  # True is not 1
    return bool(want == got)


_MISSING = object()


def record_differences(what: str, expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    """Each field of ``expected`` or ``actual`` that the other lacks or holds a different value for.

    Records (from dataclasses) and mappings are compared field by field and key
    by key, so a lost metadata key is reported on its own (``metadata['license']
    expected 1, got <missing>``), as is a changed taxonomy field
    (``dataset_metadata.taxonomy.id_density``). Any other value, such as a
    polygon, is reported whole. Fields in :data:`GEOMETRY_FIELDS` are equal
    within :data:`BOX_TOLERANCE`; two integers anywhere else exactly; every
    other number within :data:`REL_TOL`.
    """
    differences = []
    for name in _sorted_keys(expected.keys() | actual.keys()):
        close = _close_in_pixels if name in GEOMETRY_FIELDS else _close
        differences += _value_differences(str(name), expected.get(name, _MISSING), actual.get(name, _MISSING), close)
    return [f"{what}: {difference}" for difference in differences]


def _value_differences(path: str, want: Any, got: Any, close: Callable[[float, float], bool]) -> list[str]:
    if isinstance(want, dict) and isinstance(got, dict):
        differences = []
        for key in _sorted_keys(want.keys() | got.keys()):
            child = f"{path}.{key}" if isinstance(want, Record) or isinstance(got, Record) else f"{path}[{key!r}]"
            differences += _value_differences(child, want.get(key, _MISSING), got.get(key, _MISSING), close)
        return differences
    if same_value(want, got, close):
        return []
    return [f"{path} expected {_show(want)}, got {_show(got)}"]


def _sorted_keys(keys: Iterable[Any]) -> list[Any]:
    return sorted(keys, key=lambda key: (type(key).__name__, str(key)))


def _id_sort_key(value: Any) -> tuple[int, Any]:
    """Class ids in numeric order, then text ids, then anything else (``None``) by its text."""
    if _is_number(value):
        return (0, value)
    return (1, value) if isinstance(value, str) else (2, str(value))


def _show(value: Any) -> str:
    return "<missing>" if value is _MISSING else repr(value)


# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------


def _compare_labels(expected: dict[Any, Any], actual: dict[Any, Any]) -> list[str]:
    if expected == actual:
        return []
    return [f"labels {{id: name}}: expected {_by_id(expected)}, got {_by_id(actual)}"]


def _by_id(labels: dict[Any, Any]) -> dict[Any, Any]:
    return {label_id: labels[label_id] for label_id in sorted(labels, key=_id_sort_key)}


def _compare_classes(expected: list[dict[str, Any]], actual: list[dict[str, Any]]) -> list[str]:
    """Every field of each class in both snapshots, paired by id, and their order.

    A class in only one of them shows up in the labels. The taxonomy's order is
    the class order of formats whose class id is a position (YOLO), so it is
    compared too, over the ids both have.
    """
    want_by_id, got_by_id = _by_class_id(expected), _by_class_id(actual)
    differences = []
    for label_id in sorted(want_by_id, key=_id_sort_key):
        wants, gots = want_by_id[label_id], got_by_id.get(label_id, [])
        if not gots:
            continue
        if len(wants) != len(gots):
            differences.append(f"class id {label_id!r}: expected {len(wants)} entries, got {len(gots)}")
        for want, got in zip(wants, gots, strict=False):
            differences += record_differences(f"class {want.get('name')!r} (id {label_id!r})", want, got)
    shared = want_by_id.keys() & got_by_id.keys()
    want_order = [label_id for label_id in want_by_id if label_id in shared]
    got_order = [label_id for label_id in got_by_id if label_id in shared]
    if want_order != got_order:
        differences.append(f"class order: expected ids {want_order}, got {got_order}")
    return differences


def _by_class_id(entries: list[dict[str, Any]]) -> dict[Any, list[dict[str, Any]]]:
    grouped: dict[Any, list[dict[str, Any]]] = {}
    for entry in entries:
        grouped.setdefault(entry.get("source_id"), []).append(entry)
    return grouped


# ---------------------------------------------------------------------------
# Samples and boxes
# ---------------------------------------------------------------------------


def _compare_samples(expected: tuple[SampleSnapshot, ...], actual: tuple[SampleSnapshot, ...]) -> list[str]:
    differences: list[str] = []
    missing, extra = [], []
    expected_by_key, actual_by_key = _by_key(expected), _by_key(actual)
    for key in sorted(expected_by_key.keys() | actual_by_key.keys()):
        pairs, lost, gained = _pair(expected_by_key.get(key, []), actual_by_key.get(key, []))
        missing += lost
        extra += gained
        for want, got in pairs:
            differences += _compare_sample(want, got)
    colliding = {key for key, samples in expected_by_key.items() if len(samples) > 1}
    differences += _missing_and_extra(missing, extra, colliding)
    return differences


def _by_key(samples: Iterable[SampleSnapshot]) -> dict[str, list[SampleSnapshot]]:
    grouped: dict[str, list[SampleSnapshot]] = defaultdict(list)
    for sample in samples:
        grouped[sample.key].append(sample)
    return grouped


def _pair(
    expected: list[SampleSnapshot], actual: list[SampleSnapshot]
) -> tuple[list[tuple[SampleSnapshot, SampleSnapshot]], list[SampleSnapshot], list[SampleSnapshot]]:
    """Pair samples that share a key: within a split, the closest first, when the key is used more than once."""
    if len(expected) == 1 and len(actual) == 1:
        return [(expected[0], actual[0])], [], []
    return _closest_pairs(expected, actual, lambda want, got: want.split == got.split)


def _closest_pairs(
    expected: list[SampleSnapshot],
    actual: list[SampleSnapshot],
    eligible: Callable[[SampleSnapshot, SampleSnapshot], bool],
) -> tuple[list[tuple[SampleSnapshot, SampleSnapshot]], list[SampleSnapshot], list[SampleSnapshot]]:
    """Pair ``eligible`` samples one-to-one, those that agree on the most preserved fields first.

    Images with the same pixels are told apart by their split, size, boxes and
    image id, so each pairs with its own counterpart whatever the order.
    """
    candidates = sorted(
        ((i, j) for i, want in enumerate(expected) for j, got in enumerate(actual) if eligible(want, got)),
        key=lambda ij: _agreement(expected[ij[0]], actual[ij[1]]),
        reverse=True,
    )
    pairs, paired_want, paired_got = [], set(), set()
    for i, j in candidates:
        if i not in paired_want and j not in paired_got:
            paired_want.add(i)
            paired_got.add(j)
            pairs.append((expected[i], actual[j]))
    missing = [want for i, want in enumerate(expected) if i not in paired_want]
    extra = [got for j, got in enumerate(actual) if j not in paired_got]
    return pairs, missing, extra


def _agreement(want: SampleSnapshot, got: SampleSnapshot) -> tuple[bool, ...]:
    """Which preserved fields ``got`` shares with ``want``, most telling first."""
    return (
        want.image_sha256 == got.image_sha256,
        want.split == got.split,
        (want.width, want.height) == (got.width, got.height),
        not _compare_boxes("", want.boxes, got.boxes),
        same_value(want.image_id, got.image_id),
    )


def _missing_and_extra(missing: list[SampleSnapshot], extra: list[SampleSnapshot], colliding: set[str]) -> list[str]:
    """Report unpaired samples.

    A renamed image is a difference unless its name collided with another's,
    or a loss gave it any name because its name collides in the format's own
    files (:attr:`Loss.LABEL_FILE_STEMS`).
    """
    missing_by_hash: dict[str, list[SampleSnapshot]] = defaultdict(list)
    extra_by_hash: dict[str, list[SampleSnapshot]] = defaultdict(list)
    for want in missing:
        if want.image_sha256:
            missing_by_hash[want.image_sha256].append(want)
    for got in extra:
        if got.image_sha256:
            extra_by_hash[got.image_sha256].append(got)
    renamed_to: dict[int, SampleSnapshot] = {}
    for sha, wants in missing_by_hash.items():
        pairs, _, _ = _closest_pairs(wants, extra_by_hash.get(sha, []), lambda _want, _got: True)
        renamed_to.update((id(want), got) for want, got in pairs)
    differences = []
    for want in missing:
        got = renamed_to.get(id(want))
        if got is None:
            differences.append(f"{_name(want)}: missing ({len(want.boxes)} box(es))")
        elif want.key in colliding:
            # Several images shared this name, so a writer must rename all but one of them.
            differences += _compare_sample(replace(want, fields={**want.fields, "file_name": AnyValue(str)}), got)
        elif isinstance(want.fields.get("file_name"), AnyValue):
            # A loss says the format must rename this image (its name collides in the format's own files).
            differences += _compare_sample(want, got)
        else:
            differences.append(f"{_name(want)}: image was renamed to {_name(got)}")
    renamed = {id(got) for got in renamed_to.values()}
    differences += [
        f"{_name(got)}: unexpected sample ({len(got.boxes)} box(es))" for got in extra if id(got) not in renamed
    ]
    return differences


def _name(sample: SampleSnapshot) -> str:
    return f"{sample.key} [split {sample.split}]" if sample.split else sample.key


def _compare_sample(want: SampleSnapshot, got: SampleSnapshot) -> list[str]:
    """Every field of a paired image, its key's file name and split included, then its boxes."""
    name = _name(want)
    differences = record_differences(name, _without_boxes(want), _without_boxes(got))
    return differences + _compare_boxes(name, want.boxes, got.boxes)


def _without_boxes(sample: SampleSnapshot) -> dict[str, Any]:
    return {key: value for key, value in sample.fields.items() if key != "detections"}


def _compare_boxes(name: str, expected: tuple[BoxSnapshot, ...], actual: tuple[BoxSnapshot, ...]) -> list[str]:
    """Match boxes one-to-one, as many as possible, so the result doesn't depend on their order.

    Boxes match on their class, position and crowd flag. Among boxes that do,
    those whose other fields agree too are paired first, so two boxes that
    differ only in, say, their polygons each pair with their own. Every field
    of each matched pair, those it matched on included, is then compared.
    """
    want = sorted(expected, key=box_sort_key)
    got = sorted(actual, key=box_sort_key)
    identical = _max_matching(
        [[j for j, g in enumerate(got) if _same_box(w, g) and _same_fields(w, g)] for w in want], len(got)
    )
    match_of_got = _max_matching([[j for j, g in enumerate(got) if _same_box(w, g)] for w in want], len(got), identical)
    matched_want = set(match_of_got.values())
    differences = [f"{name}: missing box {w.describe()}" for i, w in enumerate(want) if i not in matched_want]
    differences += [f"{name}: unexpected box {g.describe()}" for j, g in enumerate(got) if j not in match_of_got]
    for j, i in sorted(match_of_got.items(), key=lambda pair: pair[1]):
        differences += record_differences(f"{name}: box {want[i].describe()}", want[i].fields, got[j].fields)
    return differences


def box_sort_key(box: BoxSnapshot) -> tuple[str, tuple[float, ...]]:
    return (str(box.label or ""), tuple(round(float(value), 1) for value in box.bbox))


def _same_fields(want: BoxSnapshot, got: BoxSnapshot) -> bool:
    return not record_differences("", want.fields, got.fields)


def _max_matching(candidates: list[list[int]], n_right: int, start: dict[int, int] | None = None) -> dict[int, int]:
    """A maximum matching from left items to the right items in their ``candidates``; ``{right: left}``.

    ``start`` is a matching to grow from: its pairs are kept where a maximum
    matching allows, and its matched left items stay matched.
    """
    match: dict[int, int] = dict(start or {})

    def augment(left: int, seen: set[int]) -> bool:
        # A free right item first, so pairs from ``start`` are only moved when they must be.
        for right in candidates[left]:
            if right not in match and right not in seen:
                match[right] = left
                return True
        for right in candidates[left]:
            if right in seen:
                continue
            seen.add(right)
            if right not in match or augment(match[right], seen):
                match[right] = left
                return True
        return False

    matched = set(match.values())
    for left in range(len(candidates)):
        if left not in matched:
            augment(left, set())
    assert all(0 <= right < n_right for right in match)
    return match


def _same_box(want: BoxSnapshot, got: BoxSnapshot) -> bool:
    return (
        want.label == got.label
        and same_value(want.label_id, got.label_id)
        and want.iscrowd == got.iscrowd
        and len(want.bbox) == len(got.bbox)
        and all(_close_in_pixels(a, b) for a, b in zip(want.bbox, got.bbox, strict=True))
    )
