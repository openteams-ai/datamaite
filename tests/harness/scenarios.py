"""Small synthetic datasets, each built around one edge case.

Every scenario gives an ``input`` dataset (what is written or converted) and
an ``expected`` dataset (what should come back). They are the same except
where the edge case is the input itself, such as NumPy class ids. Images are
generated in code and every image has different pixels, so the harness can
tell images apart by their content hash.
"""

from __future__ import annotations

import struct
import zlib
from collections.abc import Callable
from dataclasses import dataclass, replace
from itertools import count

from datamaite.object_detection import ObjectDetectionDataset
from datamaite.records import DatasetMetadata, ImageObjectDetectionSample, ObjectDetectionAnnotation
from datamaite.taxonomy import CategoryEntry, Taxonomy

WIDTH, HEIGHT = 64, 48

# Operations (see tests/test_conversion_harness.py). Appends are AppendCase below.
MEMORY, LOAD, ROUNDTRIP, CONVERT = "memory", "load", "roundtrip", "convert"
ALL_OPERATIONS = (MEMORY, LOAD, ROUNDTRIP, CONVERT)

_shade = count(1)


def png(width: int = WIDTH, height: int = HEIGHT, shade: int | None = None) -> bytes:
    """A tiny valid RGB PNG, built without an image library. Each call has a new shade unless one is given."""
    value = next(_shade) % 256 if shade is None else shade

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    rows = b"".join(b"\x00" + bytes([value, 0, 0]) * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def box(category_id: int, name: str, x: float, y: float, w: float = 10.0, h: float = 8.0) -> ObjectDetectionAnnotation:
    return ObjectDetectionAnnotation(bbox=(x, y, w, h), category_id=category_id, category_name=name)


def image(
    image_id: int,
    file_name: str,
    *boxes: ObjectDetectionAnnotation,
    split: str | None = "train",
    data: bytes | None = None,
    size: tuple[int, int] = (WIDTH, HEIGHT),
) -> ImageObjectDetectionSample:
    return ImageObjectDetectionSample(
        image_id=image_id,
        file_name=file_name,
        image_bytes=data if data is not None else png(*size),
        width=size[0],
        height=size[1],
        split=split,
        detections=tuple(boxes),
    )


def dataset(*samples: ImageObjectDetectionSample, labels: dict[int, str] | None = None) -> ObjectDetectionDataset:
    """An OD dataset; ``labels`` defaults to the classes its boxes use, in id order."""
    if labels is None:
        used = {d.category_id: d.category_name for s in samples for d in s.detections}
        labels = {int(i): str(used[i]) for i in sorted(used)}  # type: ignore[call-overload]
    taxonomy = Taxonomy(entries=tuple(CategoryEntry(source_id=i, name=n) for i, n in labels.items()))
    return ObjectDetectionDataset(samples=tuple(samples), dataset_metadata=DatasetMetadata(taxonomy=taxonomy))


@dataclass(frozen=True)
class Scenario:
    name: str
    build: Callable[[], tuple[ObjectDetectionDataset, ObjectDetectionDataset]]  # (input, expected)
    operations: tuple[str, ...] = ALL_OPERATIONS
    needs_image_stack: bool = False  # builds images with OpenCV and decodes them

    def __str__(self) -> str:
        return self.name


def _same(
    build: Callable[[], ObjectDetectionDataset],
) -> Callable[[], tuple[ObjectDetectionDataset, ObjectDetectionDataset]]:
    def both() -> tuple[ObjectDetectionDataset, ObjectDetectionDataset]:
        ds = build()
        return ds, ds

    return both


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def rich() -> ObjectDetectionDataset:
    """The healthy dataset every operation and format pair runs on.

    Several classes, one used by no box, a class repeated on one image, images
    with no boxes, and boxes at fractional pixel positions. Every image has its
    own pixels.
    """
    return dataset(
        image(
            1, "a.png", box(0, "person", 2.5, 3.25, 10.5, 8.75), box(1, "car", 20, 10), box(1, "car", 30.75, 30, 12, 6)
        ),
        image(2, "b.png", box(2, "bus", 5, 5, 40.25, 30.5)),
        image(3, "c.png"),
        image(4, "d.png", box(0, "person", 50, 30, 10, 15), box(2, "bus", 0, 0, 8, 8)),
        image(5, "e.png"),
        labels={0: "person", 1: "car", 2: "bus", 3: "truck"},
    )


def multi_class() -> ObjectDetectionDataset:
    """Several classes per image, a repeated class, and an image with no boxes."""
    return dataset(
        image(1, "a.png", box(0, "person", 2, 3), box(1, "car", 20, 10), box(1, "car", 30, 30, 12, 6)),
        image(2, "b.png", box(2, "bus", 5, 5, 40, 30)),
        image(3, "c.png"),
        image(4, "d.png", box(0, "person", 50, 30, 10, 15), box(2, "bus", 0, 0, 8, 8)),
    )


def standard_splits() -> ObjectDetectionDataset:
    """train, val and test, with flat file names."""
    return dataset(
        image(1, "a.png", box(0, "person", 2, 3), split="train"),
        image(2, "b.png", box(1, "car", 20, 10), split="train"),
        image(3, "c.png", box(1, "car", 4, 4), split="val"),
        image(4, "d.png", split="test"),
    )


def unsplit() -> ObjectDetectionDataset:
    """Images with no split at all, as built in memory or loaded from an unsplit folder."""
    return dataset(
        image(1, "a.png", box(0, "person", 2, 3), split=None),
        image(2, "b.png", box(1, "car", 20, 10), split=None),
    )


def single_split(split: str) -> Callable[[], ObjectDetectionDataset]:
    """Every image in one split other than train, e.g. a validation set exported on its own."""

    def build() -> ObjectDetectionDataset:
        return dataset(
            image(1, "a.png", box(0, "person", 2, 3), split=split),
            image(2, "b.png", box(1, "car", 20, 10), split=split),
        )

    build.__name__ = f"single_{split}"
    return build


def nested_split() -> ObjectDetectionDataset:
    """A nested file name outside train (#131)."""
    return dataset(
        image(1, "a.png", box(0, "person", 2, 3), split="train"),
        image(2, "cam1/c.png", box(1, "car", 4, 4), split="val"),
    )


def named_splits() -> ObjectDetectionDataset:
    """Split names from other tools: Hugging Face ``validation`` and a custom ``holdout``."""
    return dataset(
        image(1, "a.png", box(0, "person", 2, 3), split="train"),
        image(2, "b.png", box(1, "car", 20, 10), split="validation"),
        image(3, "c.png", box(0, "person", 9, 9), split="holdout"),
    )


def valid_split() -> ObjectDetectionDataset:
    """Roboflow's ``valid`` split name."""
    return dataset(
        image(1, "a.png", box(0, "person", 2, 3), split="train"),
        image(2, "b.png", box(1, "car", 20, 10), split="valid"),
    )


def nested_names() -> ObjectDetectionDataset:
    """File names in subfolders, as from multi-camera captures."""
    return dataset(
        image(1, "cam1/a.png", box(0, "person", 2, 3)),
        image(2, "cam2/day/b.png", box(1, "car", 20, 10)),
        image(3, "c.png", box(1, "car", 4, 4)),
    )


def filename_alias() -> ObjectDetectionDataset:
    """Two different images named ``a.png`` and ``./a.png``: one file on disk, so a writer must rename one."""
    return dataset(
        image(1, "a.png", box(0, "person", 2, 3)),
        image(2, "./a.png", box(1, "car", 20, 10)),
    )


def category_details() -> ObjectDetectionDataset:
    """COCO class details: supercategories, a class attribute, and a crowd box."""
    crowd = replace(box(1, "car", 20, 10, 30, 20), iscrowd=1)
    samples = (image(1, "a.png", box(0, "person", 2, 3), crowd), image(2, "b.png", box(1, "car", 4, 4)))
    entries = (
        CategoryEntry(source_id=0, name="person", supercategory="human"),
        CategoryEntry(source_id=1, name="car", supercategory="vehicle", attributes={"keypoints": ["front", "back"]}),
    )
    return ObjectDetectionDataset(samples=samples, dataset_metadata=DatasetMetadata(taxonomy=Taxonomy(entries=entries)))


# A polygon inside the car box of record_details, and a crowd mask as uncompressed RLE.
CAR_POLYGON = [[22.0, 11.0, 29.5, 11.0, 29.5, 17.25, 22.0, 17.25]]
CROWD_RLE = {"counts": [490, 4, 60, 4, 60, 4, 60, 4, 2486], "size": [HEIGHT, WIDTH]}


def record_details() -> ObjectDetectionDataset:
    """Every other record field with a value: what COCO keeps and YOLO declares lost.

    A box with a polygon, an area, attributes and a source annotation id; a box
    with a score and a separate source class id; a crowd box with an RLE mask;
    image metadata, nested included; classes with supercategories and synsets,
    one left out of evaluation and from another dataset; a dense taxonomy; and
    the dataset's split list, info, licenses and extras.
    """
    car = replace(
        box(1, "car", 20, 10),
        segmentation=CAR_POLYGON,
        area=46.875,
        source_annotation_id=42,
        attributes={"occluded": True, "colour": "red"},
    )
    person = replace(box(0, "person", 2, 3), attributes={"truncated": 0.25}, score=0.9, source_category_id=100)
    crowd = replace(box(0, "person", 10, 10, 4, 4), iscrowd=1, segmentation=CROWD_RLE, area=16.0)
    metadata = {"license": 1, "date_captured": "2024-05-01 10:00:00", "capture": {"camera": "cam1", "exposure": 0.008}}
    samples = (
        replace(image(1, "a.png", car, person), metadata=metadata),
        image(2, "b.png", replace(box(1, "car", 4, 4), source_annotation_id=7)),
        replace(image(3, "c.png", crowd), metadata={"license": 1}),
    )
    entries = (
        CategoryEntry(source_id=0, name="person", supercategory="human", synset="person.n.01"),
        CategoryEntry(
            source_id=1,
            name="car",
            supercategory="vehicle",
            synset="car.n.01",
            eval_excluded=True,
            source_dataset="coco",
        ),
    )
    return ObjectDetectionDataset(
        samples=samples,
        dataset_metadata=DatasetMetadata(
            taxonomy=Taxonomy(entries=entries, id_density="dense"),
            extras={"note": "harness"},
            splits=("train",),
            info={"description": "harness record details", "version": "1.0", "year": 2024},
            licenses=({"id": 1, "name": "CC BY 4.0", "url": "https://creativecommons.org/licenses/by/4.0/"},),
        ),
    )


def sparse_ids() -> ObjectDetectionDataset:
    """Class ids with gaps and ids of 10 and above, which sort differently as text."""
    return dataset(
        image(1, "a.png", box(3, "car", 2, 3), box(10, "bus", 20, 10)),
        image(2, "b.png", box(12, "truck", 4, 4), box(27, "person", 30, 20)),
        image(3, "c.png", box(10, "bus", 1, 1), box(3, "car", 40, 30)),
    )


def unordered_taxonomy() -> ObjectDetectionDataset:
    """A taxonomy listing class 10 before class 2. YOLO numbers classes in taxonomy order."""
    return dataset(
        image(1, "a.png", box(2, "car", 2, 3), box(10, "bus", 20, 10)),
        image(2, "b.png", box(10, "bus", 4, 4)),
        labels={10: "bus", 2: "car"},
    )


def no_taxonomy() -> tuple[ObjectDetectionDataset, ObjectDetectionDataset]:
    """Class ids 2 and 10 on boxes, with no dataset-level class list."""
    expected = dataset(
        image(1, "a.png", box(2, "two", 2, 3), box(10, "ten", 20, 10)),
        image(2, "b.png", box(10, "ten", 4, 4)),
    )
    return replace(expected, dataset_metadata=DatasetMetadata()), expected


def numpy_ids() -> tuple[ObjectDetectionDataset, ObjectDetectionDataset]:
    """Class ids as NumPy integers, as built from NumPy or pandas columns."""
    import numpy as np

    expected = multi_class()
    samples = tuple(
        replace(s, detections=tuple(replace(d, category_id=np.int64(d.category_id)) for d in s.detections))
        for s in expected.samples
    )
    return replace(expected, samples=samples), expected


def _encode(ext: str, width: int, height: int, shade: int) -> bytes:
    import cv2
    import numpy as np

    ok, data = cv2.imencode(ext, np.full((height, width, 3), shade, np.uint8))
    assert ok
    return data.tobytes()


# Colours of the EXIF fixtures: a marker under each box on a plain background.
MARKER = (200, 40, 40)
BACKGROUND = (60, 60, 160)


def exif_jpeg(orientation: int, display_size: tuple[int, int], marker_box: tuple[int, int, int, int]) -> bytes:
    """A JPEG with EXIF ``orientation`` that displays as ``display_size`` with ``MARKER`` at ``marker_box``.

    ``marker_box`` is ``(x, y, width, height)`` in the displayed frame. The stored
    pixels are the displayed image turned back, so only a decoder that applies the
    orientation puts the marker under the box. Orientation 3 (180 degrees) keeps
    the size, so it can only be checked through the pixels.
    """
    import cv2
    import numpy as np

    width, height = display_size
    shown = np.empty((height, width, 3), np.uint8)
    shown[:] = BACKGROUND
    x, y, w, h = marker_box
    shown[y : y + h, x : x + w] = MARKER
    # Orientation 6 shows the stored image turned 90 degrees clockwise; 3 turns it 180 degrees.
    stored = {6: np.rot90(shown, k=1), 3: np.rot90(shown, k=2), 1: shown}[orientation]
    ok, data = cv2.imencode(".jpg", np.ascontiguousarray(stored[:, :, ::-1]), [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    jpeg = data.tobytes()
    tag = struct.pack(">HHII", 0x0112, 3, 1, orientation << 16)
    exif = b"Exif\0\0MM\0*" + struct.pack(">IH", 8, 1) + tag + b"\0\0\0\0"
    return jpeg[:2] + b"\xff\xe1" + struct.pack(">H", len(exif) + 2) + exif + jpeg[2:]


def exif_photo() -> ObjectDetectionDataset:
    """EXIF-rotated JPEGs next to a PNG, each box on its marker in the displayed frame.

    ``a.jpg`` is stored 64x32 and shown 32x64 (orientation 6); ``e.jpg`` is shown
    upside down at its stored 64x48 (orientation 3). Boxes are in the displayed
    frame, as Ultralytics and labelling tools use.
    """
    a_box, e_box = (2, 40, 20, 16), (40, 4, 16, 12)
    return dataset(
        image(1, "a.jpg", box(0, "person", *a_box), data=exif_jpeg(6, (32, 64), a_box), size=(32, 64)),
        image(2, "d.png", box(0, "person", 9, 9)),
        image(3, "e.jpg", box(0, "person", *e_box), data=exif_jpeg(3, (WIDTH, HEIGHT), e_box)),
    )


def webp_tiff() -> ObjectDetectionDataset:
    """A WebP and a TIFF next to a PNG."""
    return dataset(
        image(1, "b.webp", box(1, "car", 20, 10), data=_encode(".webp", WIDTH, HEIGHT, 80)),
        image(2, "c.tif", box(1, "car", 4, 4), data=_encode(".tif", WIDTH, HEIGHT, 120)),
        image(3, "d.png", box(0, "person", 9, 9)),
    )


def annotation_ids() -> ObjectDetectionDataset:
    """Boxes with and without annotation ids, a box with none both before and after one with an id.

    A writer that gives a box with no id a new one must not take an id another box already has.
    """
    return dataset(
        image(1, "a.png", box(0, "person", 2, 3), replace(box(1, "car", 20, 10), source_annotation_id=1)),
        image(2, "b.png", replace(box(1, "car", 4, 4), source_annotation_id=2), box(0, "person", 30, 20)),
    )


def same_stem() -> ObjectDetectionDataset:
    """a.png and a.jpg, different images in one split: YOLO names both label files a.txt, so one must be renamed."""
    return dataset(
        image(1, "a.png", box(0, "person", 2, 3)),
        image(2, "a.jpg", box(1, "car", 20, 10), data=_encode(".jpg", WIDTH, HEIGHT, 120)),
        image(3, "b.png", box(1, "car", 4, 4)),
    )


def numpy_taxonomy_ids() -> tuple[ObjectDetectionDataset, ObjectDetectionDataset]:
    """Taxonomy ids as NumPy integers, with Python-int ids on the boxes."""
    import numpy as np

    expected = multi_class()
    taxonomy = expected.dataset_metadata.taxonomy
    assert taxonomy is not None
    entries = tuple(replace(entry, source_id=np.int64(entry.source_id)) for entry in taxonomy.entries)
    metadata = replace(expected.dataset_metadata, taxonomy=Taxonomy(entries=entries))
    return replace(expected, dataset_metadata=metadata), expected


# The rich dataset runs every operation; each edge case runs only the operations it is about.
_SPLIT_OPS = (LOAD, ROUNDTRIP, CONVERT)
SCENARIOS: tuple[Scenario, ...] = (
    Scenario("rich", _same(rich)),
    # Split discovery and writing.
    Scenario("standard_splits", _same(standard_splits), operations=_SPLIT_OPS),
    Scenario("unsplit", _same(unsplit), operations=(ROUNDTRIP, CONVERT)),
    # COCO loads one split unnamed; converted to YOLO, it is then written to train.
    Scenario("single_val", _same(single_split("val")), operations=_SPLIT_OPS),
    Scenario("single_test", _same(single_split("test")), operations=_SPLIT_OPS),
    Scenario("nested_split", _same(nested_split), operations=(LOAD, ROUNDTRIP)),
    Scenario("named_splits", _same(named_splits), operations=_SPLIT_OPS),
    Scenario("valid_split", _same(valid_split), operations=_SPLIT_OPS),
    # File names.
    Scenario("nested_names", _same(nested_names), operations=(LOAD, ROUNDTRIP)),
    Scenario("filename_alias", _same(filename_alias), operations=(MEMORY, ROUNDTRIP)),
    # Class ids and the taxonomy projection.
    Scenario("sparse_ids", _same(sparse_ids), operations=(ROUNDTRIP, CONVERT)),
    Scenario("unordered_taxonomy", _same(unordered_taxonomy), operations=(ROUNDTRIP, CONVERT)),
    Scenario("category_details", _same(category_details), operations=(LOAD, ROUNDTRIP, CONVERT)),
    # Every other record field: polygons, masks, box attributes, image metadata, synsets, info and licenses.
    Scenario("record_details", _same(record_details)),
    Scenario("no_taxonomy", no_taxonomy, operations=(ROUNDTRIP,)),
    # Input dtypes.
    Scenario("numpy_ids", numpy_ids, operations=(MEMORY, ROUNDTRIP)),
    Scenario("numpy_taxonomy_ids", numpy_taxonomy_ids, operations=(MEMORY, ROUNDTRIP)),
    # Image types and orientation.
    Scenario("exif_photo", _same(exif_photo), operations=(LOAD, ROUNDTRIP), needs_image_stack=True),
    Scenario("webp_tiff", _same(webp_tiff), operations=(LOAD, ROUNDTRIP), needs_image_stack=True),
    Scenario("annotation_ids", _same(annotation_ids), operations=(LOAD, ROUNDTRIP)),
    Scenario("same_stem", _same(same_stem), operations=(ROUNDTRIP,), needs_image_stack=True),
)


# ---------------------------------------------------------------------------
# Append cases
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AppendCase:
    """A dataset written first, a second written into it with ``mode="append"``, and what should load back.

    ``expected`` is the whole dataset after the append. When ``names_only`` is
    set, the two class lists disagree, so class ids may change; everything else,
    including each box's class name, must be kept. A writer may also refuse such
    an append with ``ValueError``, if it leaves the written dataset unchanged.
    """

    name: str
    build: Callable[[], tuple[ObjectDetectionDataset, ObjectDetectionDataset, ObjectDetectionDataset]]
    names_only: bool = False

    def __str__(self) -> str:
        return self.name


_PEOPLE_AND_CARS = {0: "person", 1: "car"}


def _first() -> tuple[ImageObjectDetectionSample, ...]:
    return (image(1, "a.png", box(0, "person", 2, 3), box(1, "car", 20, 10)), image(2, "b.png"))


def append_same_classes() -> tuple[ObjectDetectionDataset, ObjectDetectionDataset, ObjectDetectionDataset]:
    first, second = _first(), (image(3, "c.png", box(1, "car", 4, 4)),)
    return (
        dataset(*first, labels=_PEOPLE_AND_CARS),
        dataset(*second, labels=_PEOPLE_AND_CARS),
        dataset(*first, *second, labels=_PEOPLE_AND_CARS),
    )


def append_subset_of_classes() -> tuple[ObjectDetectionDataset, ObjectDetectionDataset, ObjectDetectionDataset]:
    first, second = _first(), (image(3, "c.png", box(1, "car", 4, 4)),)
    return (
        dataset(*first, labels=_PEOPLE_AND_CARS),
        dataset(*second, labels={1: "car"}),
        dataset(*first, *second, labels=_PEOPLE_AND_CARS),
    )


def append_new_class() -> tuple[ObjectDetectionDataset, ObjectDetectionDataset, ObjectDetectionDataset]:
    first, second = _first(), (image(3, "c.png", box(2, "bus", 4, 4), box(1, "car", 30, 20)),)
    return (
        dataset(*first, labels=_PEOPLE_AND_CARS),
        dataset(*second, labels={1: "car", 2: "bus"}),
        dataset(*first, *second, labels={**_PEOPLE_AND_CARS, 2: "bus"}),
    )


def append_conflicting_classes() -> tuple[ObjectDetectionDataset, ObjectDetectionDataset, ObjectDetectionDataset]:
    """The second dataset uses ids 0 and 1 for other classes (#130's reproducer)."""
    first, second = _first(), (image(3, "c.png", box(0, "bus", 4, 4), box(1, "truck", 30, 20)),)
    both = {**_PEOPLE_AND_CARS, 2: "bus", 3: "truck"}
    return (
        dataset(*first, labels=_PEOPLE_AND_CARS),
        dataset(*second, labels={0: "bus", 1: "truck"}),
        dataset(*first, *second, labels=both),
    )


def append_file_name_collision() -> tuple[ObjectDetectionDataset, ObjectDetectionDataset, ObjectDetectionDataset]:
    """The second dataset has another image named ``a.png``: both must be kept, one under a new name."""
    first, second = _first(), (image(3, "a.png", box(1, "car", 4, 4)),)
    return (
        dataset(*first, labels=_PEOPLE_AND_CARS),
        dataset(*second, labels=_PEOPLE_AND_CARS),
        dataset(*first, *second, labels=_PEOPLE_AND_CARS),
    )


def append_nothing() -> tuple[ObjectDetectionDataset, ObjectDetectionDataset, ObjectDetectionDataset]:
    first = _first()
    return (
        dataset(*first, labels=_PEOPLE_AND_CARS),
        dataset(labels=_PEOPLE_AND_CARS),
        dataset(*first, labels=_PEOPLE_AND_CARS),
    )


APPEND_CASES: tuple[AppendCase, ...] = (
    AppendCase("same_classes", append_same_classes),
    AppendCase("subset_of_classes", append_subset_of_classes),
    AppendCase("new_class", append_new_class),
    AppendCase("conflicting_classes", append_conflicting_classes, names_only=True),
    AppendCase("file_name_collision", append_file_name_collision),
    AppendCase("nothing", append_nothing),
)
