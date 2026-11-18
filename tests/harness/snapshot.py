"""A format-neutral snapshot of a loaded object-detection dataset.

:func:`snapshot` walks the dataset from its root with :func:`canonical`, which
turns everything reachable into plain data: every dataclass at any depth
(samples, boxes, the dataset metadata, the taxonomy and its entries, and any
class added later) becomes a :class:`Record` of all its fields. Then it
removes the paths in :data:`EXCLUDED_PATHS`, each listed with the reason. So
a field added to any record is compared without any change here, and a value
of a type the harness does not know fails loudly instead of being compared
with ``==``.

Each image's content is kept as ``image_sha256``, hashed from its
``image_bytes`` or, for a file-backed image, from the file at ``path_or_uri``.

:func:`problems` lists where a dataset disagrees with itself, such as a box
label that is not in the dataset's labels, a MAITE target whose boxes, labels
or scores don't match the records, a box outside its image, or a taxonomy
with a repeated or missing class id. A snapshot keeps the problems of its
dataset, and :func:`tests.harness.compare` reports ``actual``'s.
"""

from __future__ import annotations

import copy
import dataclasses
import enum
import hashlib
import math
import re
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

import numpy as np

from datamaite.object_detection import ObjectDetectionDataset
from datamaite.records import ImageObjectDetectionSample, ImageRecord

# Box coordinates (``bbox``) are equal within this many pixels. YOLO stores
# boxes normalised to six decimals, which is well inside it.
BOX_TOLERANCE = 0.01

#: Paths left out of every snapshot, each with the reason. ``[*]`` stands for
#: every item of a list. Only values that are meant to differ whatever the
#: format belong here; a field that carries the user's data is compared, and a
#: format that cannot store it declares a :class:`~tests.harness.Loss`.
#: ``test_every_excluded_path_matches_a_real_snapshot`` keeps the list current.
EXCLUDED_PATHS: dict[str, str] = {
    "samples[*].path_or_uri": "where the image is stored, not its data; its content is hashed into image_sha256",
    "dataset_id": "the MAITE dataset id, set to the name of the loader that ran",
    "image_color_mode": "how MAITE decodes images, a runtime option set after loading",
    "dataset_metadata.source_dataset": "the name of the loader that ran ('coco', 'yolo', 'datamaite')",
    "dataset_metadata.taxonomy.source_dataset": "the name of the loader that ran ('coco', 'yolo', 'datamaite')",
    "dataset_metadata.taxonomy.ordered_names": (
        "the entries' names in order, so it would report a class change twice; "
        "problems() flags a taxonomy whose ordered_names are not its entries' names"
    ),
}


class Record(dict):  # type: ignore[type-arg]
    """A dataclass as a dict of all its fields; differences name its fields as ``.name``, not ``['name']``."""


def canonical(value: Any, path: str = "dataset") -> Any:
    """``value`` as plain data that compares the same however a format stored it.

    Every dataclass becomes a :class:`Record` of all its fields, mappings
    become dicts, lists and tuples become lists, NumPy scalars and arrays
    become Python values, enums become their values and bytes become a hash,
    all recursively. An image record's ``image_bytes`` becomes
    ``image_sha256``, the hash of its content wherever it is stored. Any other
    type raises :class:`TypeError` naming ``path``, so it is handled here on
    purpose rather than compared with ``==``.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (np.generic, np.ndarray)):
        return canonical(value.tolist(), path)
    if isinstance(value, enum.Enum):
        return canonical(value.value, path)
    if isinstance(value, Mapping):
        return {_canonical_key(key, path): canonical(item, f"{path}[{key!r}]") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        record = Record()
        for f in dataclasses.fields(value):
            if isinstance(value, ImageRecord) and f.name == "image_bytes":
                record["image_sha256"] = _image_hash(value)
            else:
                record[f.name] = canonical(getattr(value, f.name), f"{path}.{f.name}")
        return record
    if isinstance(value, bytes):
        return f"<{len(value)} bytes, sha256 {hashlib.sha256(value).hexdigest()[:12]}>"
    raise TypeError(f"{path}: the harness does not know how to compare a {type(value).__qualname__}")


def _canonical_key(key: Any, path: str) -> Any:
    """A mapping key as a Python value that stays hashable (a tuple key stays a tuple)."""
    if isinstance(key, np.generic):
        return key.item()
    if isinstance(key, enum.Enum):
        return key.value
    if key is None or isinstance(key, (bool, int, float, str, tuple)):
        return key
    raise TypeError(f"{path}: the harness does not know how to compare a {type(key).__qualname__} key")


def field_defaults(cls: type) -> dict[str, Any]:
    """The canonical default of each field of ``cls`` that has one: what a format that drops the field gives back."""
    defaults = {}
    for f in dataclasses.fields(cls):
        if f.default is not dataclasses.MISSING:
            defaults[f.name] = canonical(f.default)
        elif f.default_factory is not dataclasses.MISSING:
            defaults[f.name] = canonical(f.default_factory())
    return defaults


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _segments(pattern: str) -> list[str]:
    """``"samples[*].path_or_uri"`` as ``["samples", "[*]", "path_or_uri"]``."""
    segments = re.findall(r"\[\*\]|[^.\[\]]+", pattern)
    if "".join(s if s == "[*]" else f".{s}" for s in segments).lstrip(".") != pattern:
        raise ValueError(f"not a path pattern: {pattern!r}")
    return segments


def _locations(node: Any, segments: list[str]) -> Iterator[tuple[Any, Any]]:
    """Each ``(container, key)`` that ``segments`` names under ``node``."""
    head, rest = segments[0], segments[1:]
    if head == "[*]":
        children = list(enumerate(node)) if isinstance(node, list) else []
    else:
        children = [(head, node[head])] if isinstance(node, dict) and head in node else []
    for key, child in children:
        if rest:
            yield from _locations(child, rest)
        else:
            yield node, key


def matches(tree: Any, pattern: str) -> int:
    """How many values ``pattern`` names in ``tree``."""
    return sum(1 for _ in _locations(tree, _segments(pattern)))


def remove(tree: Any, pattern: str) -> None:
    """Remove every value ``pattern`` names from ``tree``, in place."""
    for container, key in list(_locations(tree, _segments(pattern))):
        del container[key]


def edit(tree: Any, pattern: str, change: Callable[[Any], Any]) -> None:
    """Replace every value ``pattern`` names in ``tree`` with ``change(value)``, in place."""
    for container, key in list(_locations(tree, _segments(pattern))):
        container[key] = change(container[key])


# ---------------------------------------------------------------------------
# Expected values a format makes up
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnyValue:
    """An expected value that matches any value of ``kind``, for one a format makes up, such as a new id."""

    kind: type

    def __repr__(self) -> str:
        return f"<any {self.kind.__name__}>"


@dataclass(frozen=True)
class Near:
    """An expected number a format derives, such as an area from a box, known only within ``abs_tol``.

    It reads as its value, so a difference looks like any other number's.
    """

    value: float
    abs_tol: float

    def __repr__(self) -> str:
        return repr(self.value)


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoxSnapshot:
    """One box: every field of its record (class, ``bbox``, crowd flag, polygon, attributes, ...)."""

    fields: dict[str, Any]

    @property
    def label(self) -> Any:
        return self.fields.get("category_name")

    @property
    def label_id(self) -> Any:
        return self.fields.get("category_id")

    @property
    def bbox(self) -> tuple[float, ...]:
        return tuple(self.fields.get("bbox") or ())

    @property
    def iscrowd(self) -> Any:
        return self.fields.get("iscrowd")

    def describe(self) -> str:
        coords = ", ".join(f"{value:g}" for value in self.bbox)
        crowd = " (crowd)" if self.iscrowd else ""
        return f"{self.label!r} (id {self.label_id!r}) at [{coords}]{crowd}"


@dataclass(frozen=True)
class SampleSnapshot:
    """One image: its key (normalised file name) and every field of its record, its boxes in ``detections``.

    With ``snapshot(..., decode=True)`` the fields also hold ``decoded_size``,
    the ``(width, height)`` of the image as MAITE decodes it, and
    ``decoded_sha256``, a hash of its decoded pixels.
    """

    key: str
    fields: dict[str, Any]

    @property
    def split(self) -> Any:
        return self.fields.get("split")

    @property
    def image_id(self) -> Any:
        return self.fields.get("image_id")

    @property
    def image_sha256(self) -> Any:
        return self.fields.get("image_sha256")

    @property
    def width(self) -> Any:
        return self.fields.get("width")

    @property
    def height(self) -> Any:
        return self.fields.get("height")

    @property
    def boxes(self) -> tuple[BoxSnapshot, ...]:
        return tuple(BoxSnapshot(box) for box in self.fields.get("detections", ()))


@dataclass(frozen=True)
class DatasetSnapshot:
    """A whole dataset as plain data: its own fields, its samples, and its problems.

    ``fields`` is the canonical dataset without its samples (its task, its
    metadata with the taxonomy, split list, info and licenses, ...);
    ``samples`` keeps each sample's key next to its fields. :attr:`tree` puts
    them back together.
    """

    fields: dict[str, Any]
    samples: tuple[SampleSnapshot, ...]
    problems: tuple[str, ...] = ()

    @property
    def tree(self) -> dict[str, Any]:
        """A copy of the whole canonical dataset, samples included, to read or edit by path."""
        return copy.deepcopy({**self.fields, "samples": [sample.fields for sample in self.samples]})

    def with_tree(self, tree: dict[str, Any]) -> DatasetSnapshot:
        """This snapshot with its data replaced by ``tree``, which has the same samples in the same order."""
        samples = tree.pop("samples")
        assert len(samples) == len(self.samples)
        return DatasetSnapshot(
            fields=tree,
            samples=tuple(SampleSnapshot(s.key, fields) for s, fields in zip(self.samples, samples, strict=True)),
            problems=self.problems,
        )

    @property
    def entries(self) -> list[dict[str, Any]]:
        """The taxonomy's class entries, in order (none without a taxonomy)."""
        taxonomy = (self.fields.get("dataset_metadata") or {}).get("taxonomy")
        return list(taxonomy.get("entries", ())) if isinstance(taxonomy, dict) else []

    @property
    def labels(self) -> dict[Any, Any]:
        """``{class id: name}`` from the taxonomy, a readable summary of its entries."""
        return {entry.get("source_id"): entry.get("name") for entry in self.entries}


def default_key(sample: ImageObjectDetectionSample) -> str:
    """The sample's file name as a POSIX path, falling back to its source path's name, then its id."""
    raw = sample.file_name or (PurePosixPath(sample.path_or_uri).name if sample.path_or_uri else None)
    if raw is None:
        return str(sample.image_id)
    return PurePosixPath(str(raw).replace("\\", "/")).as_posix()


def snapshot(
    dataset: ObjectDetectionDataset,
    *,
    decode: bool = False,
    key: Callable[[ImageObjectDetectionSample], str] = default_key,
) -> DatasetSnapshot:
    """Take a format-neutral snapshot of ``dataset``, with its :func:`problems`.

    ``decode=True`` also decodes every image through the MAITE interface,
    keeps its size and a hash of its pixels, and checks the size against the
    stored size: the same file bytes are not yet the same pixels to a model.
    It needs the image extra (``datamaite[od]``). ``key`` names each sample;
    the default is the file name, which YOLO and COCO both keep relative to
    their image folders.
    """
    tree = canonical(dataset)
    for pattern in EXCLUDED_PATHS:
        remove(tree, pattern)
    keys = [key(sample) for sample in dataset.samples]
    decoded: list[tuple[int, int]] | None = None
    if decode:
        decoded = []
        # One image at a time: only its size and hash are kept, not its pixels.
        for index, fields in enumerate(tree["samples"]):
            image = dataset.get_input(index)
            decoded.append((int(image.shape[2]), int(image.shape[1])))
            fields["decoded_size"] = list(decoded[-1])
            fields["decoded_sha256"] = _pixels_hash(image)
    samples = tuple(SampleSnapshot(k, fields) for k, fields in zip(keys, tree.pop("samples"), strict=True))
    return DatasetSnapshot(fields=tree, samples=samples, problems=tuple(_problems(dataset, keys, decoded)))


# ---------------------------------------------------------------------------
# Problems
# ---------------------------------------------------------------------------


def problems(
    dataset: ObjectDetectionDataset,
    *,
    decode: bool = False,
    key: Callable[[ImageObjectDetectionSample], str] = default_key,
) -> list[str]:
    """Every place ``dataset`` disagrees with itself.

    A taxonomy entry with no class id or a class id used by two entries, a
    taxonomy whose ``ordered_names`` are not its entries' names, a box label
    that is not in the dataset's labels, a MAITE target whose boxes, labels or
    scores don't match the records or whose label is not in the public MAITE
    ``index2label``, a box outside its image, and (with ``decode=True``) an
    image whose decoded size is not the stored size.
    """
    keys = [key(sample) for sample in dataset.samples]
    decoded = [_decoded_size(dataset, index) for index in range(len(dataset.samples))] if decode else None
    return _problems(dataset, keys, decoded)


def _problems(dataset: ObjectDetectionDataset, keys: list[str], decoded: list[tuple[int, int]] | None) -> list[str]:
    taxonomy = dataset.dataset_metadata.taxonomy
    entries = taxonomy.entries if taxonomy is not None else ()
    found = _taxonomy_problems(dataset)
    labels: dict[int | str, str] = {
        entry.source_id: entry.name  # type: ignore[misc]
        for entry in entries
        if entry.source_id is not None
    }
    index2label = dict(dataset.metadata["index2label"])
    found += _index2label_problems(entries, index2label)
    for index in range(len(dataset.samples)):
        size = decoded[index] if decoded is not None else None
        found.extend(_sample_problems(dataset, index, keys[index], labels, index2label, size))
    return found


def _index2label_problems(entries: tuple[Any, ...], index2label: dict[int, str]) -> list[str]:
    """Where the public MAITE ``index2label`` is not every class with an integer id, unused classes included."""
    want = {
        int(entry.source_id): entry.name
        for entry in entries
        if isinstance(entry.source_id, (int, np.integer)) and not isinstance(entry.source_id, bool)
    }
    found = [
        f"taxonomy: class {name!r} (id {label_id}) is not in the MAITE index2label as {name!r}"
        for label_id, name in want.items()
        if index2label.get(label_id) != name
    ]
    found += [
        f"taxonomy: MAITE index2label maps {label_id} to {name!r}, which is no class"
        for label_id, name in index2label.items()
        if label_id not in want
    ]
    return found


def _taxonomy_problems(dataset: ObjectDetectionDataset) -> list[str]:
    taxonomy = dataset.dataset_metadata.taxonomy
    if taxonomy is None:
        return []
    found = [f"taxonomy: class {entry.name!r} has no id" for entry in taxonomy.entries if entry.source_id is None]
    counts = Counter(entry.source_id for entry in taxonomy.entries if entry.source_id is not None)
    for source_id, n in counts.items():
        if n > 1:
            names = [entry.name for entry in taxonomy.entries if entry.source_id == source_id]
            found.append(f"taxonomy: class id {source_id!r} is used by {n} entries {names}")
    names = tuple(entry.name for entry in taxonomy.entries)
    if tuple(taxonomy.ordered_names) != names:
        found.append(f"taxonomy: ordered_names {list(taxonomy.ordered_names)} are not the entries' names {list(names)}")
    return found


def _image_hash(sample: ImageRecord) -> str | None:
    data = sample.image_bytes
    if data is None and sample.path_or_uri is not None:
        from datamaite._upath import to_dataset_path

        try:
            data = to_dataset_path(sample.path_or_uri).read_bytes()
        except OSError:
            return None
    return hashlib.sha256(data).hexdigest() if data is not None else None


def _pixels_hash(image: Any) -> str:
    """A hash of a decoded image's pixels, with its shape and type."""
    array = np.ascontiguousarray(np.asarray(image))
    header = f"{array.dtype.str}{array.shape}".encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def _decoded_size(dataset: ObjectDetectionDataset, index: int) -> tuple[int, int]:
    image = dataset.get_input(index)
    return int(image.shape[2]), int(image.shape[1])


def _sample_problems(
    dataset: ObjectDetectionDataset,
    index: int,
    key: str,
    labels: dict[int | str, str],
    index2label: dict[int, str],
    decoded_size: tuple[int, int] | None,
) -> list[str]:
    sample = dataset.samples[index]
    found: list[str] = []
    width, height = sample.width, sample.height
    if decoded_size is not None and (width, height) != decoded_size:
        found.append(
            f"{key}: stored size {width}x{height} but the image decodes to {decoded_size[0]}x{decoded_size[1]}"
        )
    target = dataset.get_target(index)
    n = len(sample.detections)
    shapes = {"boxes": target.boxes.shape, "labels": target.labels.shape, "scores": target.scores.shape}
    expected_shapes = {"boxes": (n, 4), "labels": (n,), "scores": (n,)}
    if shapes != expected_shapes:
        found.append(f"{key}: MAITE target shapes {shapes} but the sample has {n} box(es)")
        return found
    maite_labels: list[Any] = target.labels.tolist()
    rows = zip(sample.detections, maite_labels, target.boxes.tolist(), target.scores.tolist(), strict=True)
    for detection, maite_label, maite_box, maite_score in rows:
        what = f"{key}: box {detection.category_name!r} (id {detection.category_id!r})"
        if detection.category_id is None or labels.get(detection.category_id) != detection.category_name:
            found.append(f"{what} does not match the dataset labels {labels}")
        if maite_label != detection.category_id:
            found.append(f"{what} has MAITE target label {maite_label}")
        elif index2label.get(maite_label) != detection.category_name:
            found.append(f"{what}: MAITE index2label maps {maite_label} to {index2label.get(maite_label)!r}")
        x, y, w, h = (float(value) for value in detection.bbox)
        corners = (x, y, x + w, y + h)
        if not all(abs(a - b) <= BOX_TOLERANCE for a, b in zip(corners, maite_box, strict=True)):
            found.append(f"{what}: MAITE target box {maite_box} is not the record's corners {list(corners)}")
        # Ground truth has no score; MAITE targets give it 1.0.
        record_score = 1.0 if detection.score is None else float(detection.score)
        # NaN compares false with everything, so check finiteness before closeness.
        if not (
            math.isfinite(maite_score)
            and math.isfinite(record_score)
            and math.isclose(maite_score, record_score, rel_tol=0.0, abs_tol=1e-6)
        ):
            found.append(f"{what}: MAITE target score {maite_score} but the record's is {record_score}")
        if (
            width
            and height
            and not (
                all(math.isfinite(value) for value in (x, y, w, h))
                and x >= -BOX_TOLERANCE
                and y >= -BOX_TOLERANCE
                and x + w <= width + BOX_TOLERANCE
                and y + h <= height + BOX_TOLERANCE
            )
        ):
            found.append(f"{what} at {detection.bbox} lies outside the {width}x{height} image")
    return found
