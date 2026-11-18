"""Sizes and boxes of EXIF-rotated JPEGs are in the displayed frame that decoding produces (#133)."""

from __future__ import annotations

import json
import logging
import struct
from pathlib import Path
from typing import Any

import pytest

from datamaite import load_od


def _jpeg(width: int, height: int, orientation: int | None) -> bytes:
    """A JPEG header stored ``width`` x ``height``, with an EXIF orientation tag (no pixels needed)."""
    segments = b""
    if orientation is not None:
        exif = b"Exif\0\0MM\0*" + struct.pack(">IH", 8, 1) + struct.pack(">HHII", 0x0112, 3, 1, orientation << 16)
        exif += b"\0\0\0\0"
        segments = b"\xff\xe1" + struct.pack(">H", len(exif) + 2) + exif
    sof = b"\x08" + struct.pack(">HH", height, width) + b"\x03\x01\x11\x00\x02\x11\x00\x03\x11\x00"
    return b"\xff\xd8" + segments + b"\xff\xc0" + struct.pack(">H", len(sof) + 2) + sof + b"\xff\xd9"


def _coco(root: Path, image: bytes, record: dict[str, Any], annotation: dict[str, Any]) -> Path:
    (root / "annotations").mkdir(parents=True)
    (root / record["file_name"]).write_bytes(image)
    document = {
        "images": [{"id": 1, **record}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, **annotation}],
        "categories": [{"id": 1, "name": "car"}],
    }
    (root / "annotations" / "instances.json").write_text(json.dumps(document), encoding="utf-8")
    return root


# A 64x32 stored image. The box covers x 40..64, y 0..8 and the polygon is its outline.
_RAW_BOX = [40, 0, 24, 8]
_RAW_POLYGON = [40, 0, 64, 0, 64, 8, 40, 8]


@pytest.mark.parametrize(
    ("orientation", "box", "polygon"),
    [
        (5, (0, 40, 8, 24), [0, 40, 0, 64, 8, 64, 8, 40]),  # transpose
        (6, (24, 40, 8, 24), [32, 40, 32, 64, 24, 64, 24, 40]),  # 90 degrees clockwise
        (7, (24, 0, 8, 24), [32, 24, 32, 0, 24, 0, 24, 24]),  # transverse
        (8, (0, 0, 8, 24), [0, 24, 0, 0, 8, 0, 8, 24]),  # 90 degrees anticlockwise
    ],
)
def test_coco_record_in_the_stored_frame_is_turned(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, orientation: int, box: tuple[int, ...], polygon: list[int]
) -> None:
    root = _coco(
        tmp_path,
        _jpeg(64, 32, orientation),
        {"file_name": "a.jpg", "width": 64, "height": 32},
        {"bbox": _RAW_BOX, "segmentation": [_RAW_POLYGON], "area": 192},
    )

    with caplog.at_level(logging.WARNING, logger="datamaite._formats.coco.loader"):
        sample = load_od(root, dataset_format="coco").samples[0]

    assert (sample.width, sample.height) == (32, 64)
    detection = sample.detections[0]
    assert detection.bbox == box
    assert detection.segmentation == [polygon]
    assert detection.area == 192
    assert "EXIF-rotated" in caplog.text
    assert "a.jpg" in caplog.text


@pytest.mark.parametrize(
    ("orientation", "record"),
    [
        (6, {"width": 32, "height": 64}),  # already in the displayed frame
        (3, {"width": 64, "height": 32}),  # keeps the size, so it can't be detected
        (None, {"width": 64, "height": 32}),
        (6, {}),  # no size to compare
    ],
    ids=["displayed-frame", "orientation-3", "no-exif", "no-size"],
)
def test_coco_records_that_cannot_be_turned_are_left_alone(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, orientation: int | None, record: dict[str, int]
) -> None:
    root = _coco(tmp_path, _jpeg(64, 32, orientation), {"file_name": "a.jpg", **record}, {"bbox": [1, 2, 3, 4]})

    with caplog.at_level(logging.WARNING, logger="datamaite._formats.coco.loader"):
        sample = load_od(root, dataset_format="coco").samples[0]

    assert (sample.width, sample.height) == (record.get("width"), record.get("height"))
    assert sample.detections[0].bbox == (1, 2, 3, 4)
    assert "EXIF-rotated" not in caplog.text


def test_coco_rle_masks_are_kept_with_a_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    rle = {"size": [32, 64], "counts": [0, 2048]}
    root = _coco(
        tmp_path,
        _jpeg(64, 32, 6),
        {"file_name": "a.jpg", "width": 64, "height": 32},
        {"bbox": _RAW_BOX, "segmentation": rle, "iscrowd": 1},
    )

    with caplog.at_level(logging.WARNING, logger="datamaite._formats.coco.loader"):
        detection = load_od(root, dataset_format="coco").samples[0].detections[0]

    assert detection.bbox == (24, 40, 8, 24)
    assert detection.segmentation == rle
    assert "RLE segmentation stays in the stored" in caplog.text


def test_coco_missing_image_is_left_alone(tmp_path: Path) -> None:
    root = _coco(tmp_path, b"", {"file_name": "a.jpg", "width": 64, "height": 32}, {"bbox": _RAW_BOX})
    (root / "a.jpg").unlink()

    sample = load_od(root, dataset_format="coco").samples[0]

    assert (sample.width, sample.height) == (64, 32)
    assert sample.detections[0].bbox == tuple(_RAW_BOX)


def _yolo(root: Any, image: bytes) -> None:
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (root / "images" / "train" / "a.jpg").write_bytes(image)
    (root / "labels" / "train" / "a.txt").write_text("0 0.5 0.25 1.0 0.5\n")  # top half of the displayed image
    (root / "data.yaml").write_text("train: images/train\nnames:\n  0: car\n")


@pytest.mark.parametrize("orientation", [5, 6, 7, 8])
def test_yolo_boxes_are_scaled_in_the_displayed_frame(tmp_path: Path, orientation: int) -> None:
    _yolo(tmp_path, _jpeg(64, 32, orientation))

    sample = load_od(tmp_path, dataset_format="yolo").samples[0]

    assert (sample.width, sample.height) == (32, 64)
    assert sample.detections[0].bbox == (0.0, 0.0, 32.0, 32.0)


def test_yolo_reads_the_orientation_of_a_remote_image(memory_root) -> None:  # type: ignore[no-untyped-def]
    root = memory_root / "yolo"
    _yolo(root, _jpeg(64, 32, 6))

    sample = load_od(root, dataset_format="yolo").samples[0]

    assert (sample.width, sample.height) == (32, 64)


@pytest.mark.optional
def test_issue_reproducers_decode_to_the_box_frame(tmp_path: Path) -> None:
    """Both #133 reproducers: the MAITE image and its boxes share a frame."""
    import cv2
    import numpy as np

    jpg = cv2.imencode(".jpg", np.zeros((32, 64, 3), np.uint8))[1].tobytes()
    app1 = b"Exif\0\0MM\0*" + struct.pack(">IH", 8, 1) + struct.pack(">HHII", 0x0112, 3, 1, 6 << 16) + b"\0\0\0\0"
    rotated = jpg[:2] + b"\xff\xe1" + struct.pack(">H", len(app1) + 2) + app1 + jpg[2:]

    coco = _coco(tmp_path / "coco", rotated, {"file_name": "a.jpg", "width": 64, "height": 32}, {"bbox": _RAW_BOX})
    image, target, _ = load_od(coco, dataset_format="coco")[0]
    assert image.shape == (3, 64, 32)
    assert target.boxes.tolist() == [[24.0, 40.0, 32.0, 64.0]]

    _yolo(tmp_path / "yolo", rotated)
    image, target, _ = load_od(tmp_path / "yolo", dataset_format="yolo")[0]
    assert image.shape == (3, 64, 32)
    assert target.boxes.tolist() == [[0.0, 0.0, 32.0, 32.0]]


@pytest.mark.parametrize("output_format", ["coco", "yolo"])
def test_a_writer_sizes_a_record_with_no_size_in_the_displayed_frame(tmp_path: Path, output_format: str) -> None:
    """A record with no stored size (as Hugging Face loads) is probed in the frame the loaders use."""
    from dataclasses import replace

    from datamaite import write

    _yolo(tmp_path / "src", _jpeg(64, 32, 6))
    # The bottom half of the displayed 32x64 image: outside the stored 64x32 frame.
    (tmp_path / "src" / "labels" / "train" / "a.txt").write_text("0 0.5 0.75 1.0 0.5\n")
    dataset = load_od(tmp_path / "src", dataset_format="yolo")
    unsized = replace(dataset, samples=[replace(s, width=None, height=None) for s in dataset.samples])

    write(unsized, tmp_path / "out", output_format=output_format)
    sample = load_od(tmp_path / "out", dataset_format=output_format).samples[0]

    assert (sample.width, sample.height) == (32, 64)
    assert [d.bbox for d in sample.detections] == [(0.0, 32.0, 32.0, 32.0)]


def _with_png_orientation(png: bytes, orientation: int) -> bytes:
    """``png`` with an eXIf chunk holding ``orientation``, right after IHDR."""
    import zlib

    tiff = b"MM\0*" + struct.pack(">IH", 8, 1) + struct.pack(">HHII", 0x0112, 3, 1, orientation << 16) + b"\0\0\0\0"
    chunk = struct.pack(">I", len(tiff)) + b"eXIf" + tiff + struct.pack(">I", zlib.crc32(b"eXIf" + tiff) & 0xFFFFFFFF)
    return png[:33] + chunk + png[33:]


def _one_image(image: bytes, width: int, height: int) -> Any:
    from datamaite.object_detection import ObjectDetectionDataset
    from datamaite.records import ImageObjectDetectionSample, ObjectDetectionAnnotation

    sample = ImageObjectDetectionSample(
        image_id="a",
        file_name="a.png",
        image_bytes=image,
        width=width,
        height=height,
        detections=(ObjectDetectionAnnotation(bbox=(0, 0, 4, 4), category_id=0, category_name="car"),),
    )
    return ObjectDetectionDataset(samples=(sample,))


@pytest.mark.optional
@pytest.mark.parametrize("orientation", range(1, 9))
def test_native_decodes_in_the_frame_opencv_displays(orientation: int) -> None:
    """Pixel for pixel, native turns a lossless image as OpenCV turns it for rgb."""
    import cv2
    import numpy as np

    pixels = np.random.default_rng(orientation).integers(0, 256, (30, 50, 3), dtype=np.uint8)
    png = _with_png_orientation(cv2.imencode(".png", pixels)[1].tobytes(), orientation)
    dataset = _one_image(png, 50, 30)

    rgb = dataset[0][0]
    native = dataset.with_image_options(color_mode="native")[0][0]

    np.testing.assert_array_equal(native, rgb)


@pytest.mark.optional
def test_native_image_and_metadata_share_a_frame() -> None:
    """The review's reproducer: orientation 6, stored 50x30."""
    import cv2
    import numpy as np

    jpg = cv2.imencode(".jpg", np.zeros((30, 50, 3), np.uint8))[1].tobytes()
    app1 = b"Exif\0\0MM\0*" + struct.pack(">IH", 8, 1) + struct.pack(">HHII", 0x0112, 3, 1, 6 << 16) + b"\0\0\0\0"
    rotated = jpg[:2] + b"\xff\xe1" + struct.pack(">H", len(app1) + 2) + app1 + jpg[2:]
    dataset = _one_image(rotated, 30, 50).with_image_options(color_mode="native")

    image, _, metadata = dataset[0]

    assert image.shape == (3, 50, 30)
    assert (metadata["width"], metadata["height"]) == (30, 50)
