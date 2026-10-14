"""Still-image conversion matrix: every OD/IC input -> output pair through ``convert()``.

The sibling of :mod:`tests.test_conversion_matrix` (MOT). Each cell writes one
small synthetic dataset in the input format, converts it with
:func:`datamaite.convert`, reloads the destination, and compares images, class
names and boxes against the original. Before these tests existed, conversions
into COCO wrote an empty dataset, VisDrone -> YOLO dropped every label, YOLO ->
VisDrone relabelled cars as pedestrians, and Hugging Face OD reloads had no
integer labels -- all while ``convert()`` returned normally.

Formats are derived from the registries, so a new read+write OD/IC format joins
the matrix automatically. The only exclusions are formats whose writer declares
it cannot carry the annotations at all (``flat_images`` stores images only).

Images are tiny pure-Python PNGs: ``convert()`` copies media without decoding,
so no image stack is required.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import pytest

from datamaite import convert, load_ic, load_od, write
from datamaite._types import DatasetFormat, Task
from datamaite.image_classification import ImageClassificationDataset
from datamaite.loaders import available_formats
from datamaite.object_detection import ObjectDetectionDataset
from datamaite.records import (
    ClassificationLabel,
    DatasetMetadata,
    ImageClassificationSample,
    ImageObjectDetectionSample,
    ObjectDetectionAnnotation,
)
from datamaite.taxonomy import CategoryEntry, Taxonomy
from datamaite.writers import available_output_formats, get_writer

# VisDrone ids/names, so every format -- including VisDrone's fixed class table --
# can represent the source classes losslessly.
_CLASSES = {1: "pedestrian", 4: "car"}
_WIDTH, _HEIGHT = 64, 48


def _png(width: int = _WIDTH, height: int = _HEIGHT) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    rows = b"".join(b"\x00" + (b"\x00\x00\x00" * width) for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def _convertible(task: Task, lossy_key: str | None) -> list[DatasetFormat]:
    return sorted(
        (
            fmt
            for fmt in set(available_formats(task=task)) & set(available_output_formats(task=task))
            if lossy_key is None or lossy_key not in get_writer(fmt, task=task).capabilities.lossy_without
        ),
        key=lambda fmt: fmt.value,
    )


OD_FORMATS = _convertible(Task.OD, "detections")
IC_FORMATS = _convertible(Task.IC, None)


def _taxonomy() -> Taxonomy:
    return Taxonomy(entries=tuple(CategoryEntry(source_id=cid, name=name) for cid, name in _CLASSES.items()))


# ---------------------------------------------------------------------------
# Object detection
# ---------------------------------------------------------------------------


def _od_dataset(raw: Path) -> ObjectDetectionDataset:
    raw.mkdir(parents=True)
    samples = []
    for index in range(3):
        image = raw / f"img{index}.png"
        image.write_bytes(_png())
        detections = tuple(
            ObjectDetectionAnnotation(
                bbox=(4.0 + 3 * j + index, 5.0 + 2 * j, 10.0, 12.0),
                category_id=category_id,
                category_name=_CLASSES[category_id],
                source_category_id=category_id,
            )
            for j, category_id in enumerate([1, 4][: index + 1])
        )
        samples.append(
            ImageObjectDetectionSample(
                image_id=index + 1,
                path_or_uri=str(image),
                file_name=image.name,
                width=_WIDTH,
                height=_HEIGHT,
                split="train",
                detections=detections,
            )
        )
    return ObjectDetectionDataset(samples=tuple(samples), dataset_metadata=DatasetMetadata(taxonomy=_taxonomy()))


def _od_fingerprint(ds: ObjectDetectionDataset) -> list[tuple[str, tuple[tuple[str | None, tuple[float, ...]], ...]]]:
    """Image stem -> sorted (class name, box); ids and paths legitimately change across formats."""
    return sorted(
        (
            Path(sample.file_name or sample.path_or_uri or "").stem,
            tuple(
                sorted(
                    (d.category_name, tuple(round(value, 2) for value in d.bbox))  # YOLO normalization rounding
                    for d in sample.detections
                )
            ),
        )
        for sample in ds.samples
    )


def _assert_maite_labels(ds: ObjectDetectionDataset) -> None:
    """Every box has an integer class in ``index2label`` (a MAITE target label other than -1)."""
    index2label = ds.index2label()
    for sample in ds.samples:
        for detection in sample.detections:
            assert isinstance(detection.category_id, int), detection
            assert index2label.get(detection.category_id) == detection.category_name, (detection, index2label)


def test_od_matrix_covers_the_expected_formats() -> None:
    # Guardrail: flat_images is the only OD format excluded, and only because
    # its writer declares it drops detections.
    assert [fmt.value for fmt in OD_FORMATS] == ["coco", "huggingface_vision", "visdrone", "yolo"]


@pytest.mark.parametrize("output_format", OD_FORMATS, ids=lambda fmt: fmt.value)
@pytest.mark.parametrize("input_format", OD_FORMATS, ids=lambda fmt: fmt.value)
def test_od_convert_preserves_images_classes_and_boxes(
    input_format: DatasetFormat, output_format: DatasetFormat, tmp_path: Path
) -> None:
    original = _od_dataset(tmp_path / "raw")
    write(original, tmp_path / "src", output_format=input_format)
    src = tmp_path / "src"  # loads at the write destination, VisDrone split roots included
    assert _od_fingerprint(load_od(src, dataset_format=input_format)) == _od_fingerprint(original)

    convert(src, tmp_path / "out", input_format=input_format, output_format=output_format, task="od")

    reloaded = load_od(tmp_path / "out", dataset_format=output_format)
    assert _od_fingerprint(reloaded) == _od_fingerprint(original)
    _assert_maite_labels(reloaded)


# ---------------------------------------------------------------------------
# Image classification
# ---------------------------------------------------------------------------


def _ic_dataset(raw: Path) -> ImageClassificationDataset:
    raw.mkdir(parents=True)
    samples = []
    for index, category_id in enumerate([1, 4, 1, 4]):
        image = raw / f"img{index}.png"
        image.write_bytes(_png())
        samples.append(
            ImageClassificationSample(
                image_id=f"img{index}",
                path_or_uri=str(image),
                file_name=image.name,
                width=_WIDTH,
                height=_HEIGHT,
                split="train",
                labels=(
                    ClassificationLabel(
                        category_id=category_id, category_name=_CLASSES[category_id], source_category_id=category_id
                    ),
                ),
            )
        )
    return ImageClassificationDataset(samples=tuple(samples), dataset_metadata=DatasetMetadata(taxonomy=_taxonomy()))


def _ic_fingerprint(ds: ImageClassificationDataset) -> list[tuple[str, ...]]:
    return sorted(tuple(sorted(label.category_name or "" for label in sample.labels)) for sample in ds.samples)


# VisDrone IC samples are object crops (a ``region`` of a source image); the
# whole-image YOLO / Hugging Face classification layouts cannot express a crop,
# so VisDrone is only a valid IC *input* for VisDrone output.
_IC_PAIRS = [
    (fi, fo)
    for fi in IC_FORMATS
    for fo in IC_FORMATS
    if fi is not DatasetFormat.VISDRONE or fo is DatasetFormat.VISDRONE
]


@pytest.mark.parametrize(("input_format", "output_format"), _IC_PAIRS, ids=lambda fmt: fmt.value)
def test_ic_convert_preserves_labels(input_format: DatasetFormat, output_format: DatasetFormat, tmp_path: Path) -> None:
    original = _ic_dataset(tmp_path / "raw")
    write(original, tmp_path / "src", output_format=input_format)
    src = tmp_path / "src"  # loads at the write destination, VisDrone split roots included
    assert _ic_fingerprint(load_ic(src, dataset_format=input_format)) == _ic_fingerprint(original)

    convert(src, tmp_path / "out", input_format=input_format, output_format=output_format, task="ic")

    reloaded = load_ic(tmp_path / "out", dataset_format=output_format)
    assert _ic_fingerprint(reloaded) == _ic_fingerprint(original)
