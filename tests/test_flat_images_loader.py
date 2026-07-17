"""Tests for the flat-folder still-image loader (IR-3.2-S-1)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from datamaite import DatasetFormat, FlatImagesLoader, Task, load, load_od
from datamaite._formats.flat_images.loader import load_flat_images
from datamaite.loaders import available_formats, get_loader
from datamaite.object_detection import ObjectDetectionDataset

_JPEG_HEAD = b"\xff\xd8\xff\xe0" + b"\x00" * 20
_PNG_HEAD = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
_TIFF_LE_HEAD = b"II*\x00" + b"\x00" * 20
_TIFF_BE_HEAD = b"MM\x00*" + b"\x00" * 20


class TestFlatImagesRegistry:
    def test_registered_and_public_api(self) -> None:
        assert DatasetFormat.FLAT_IMAGES in available_formats()
        assert DatasetFormat.FLAT_IMAGES in available_formats(task=Task.OD)
        assert isinstance(get_loader(DatasetFormat.FLAT_IMAGES), FlatImagesLoader)
        assert isinstance(get_loader("flat_images"), FlatImagesLoader)
        assert callable(load_flat_images)

    def test_dispatch_via_load_and_load_od(self, tmp_path: Path) -> None:
        (tmp_path / "a.jpg").write_bytes(_JPEG_HEAD)

        via_load = load(tmp_path, dataset_format="flat_images")
        via_load_od = load_od(tmp_path, dataset_format="flat_images")

        assert isinstance(via_load, ObjectDetectionDataset)
        assert isinstance(via_load_od, ObjectDetectionDataset)
        assert len(via_load_od) == 1

    def test_never_autodetected(self, tmp_path: Path) -> None:
        # Explicit opt-in only (#40): a bare folder of images must not sniff.
        (tmp_path / "a.jpg").write_bytes(_JPEG_HEAD)
        (tmp_path / "b.png").write_bytes(_PNG_HEAD)

        assert FlatImagesLoader.sniff(tmp_path) is False
        with pytest.raises(ValueError, match="Could not autodetect"):
            load(tmp_path, dataset_format=None)


class TestFlatImagesHappyPath:
    def test_loads_all_standard_formats(self, tmp_path: Path) -> None:
        (tmp_path / "a.jpg").write_bytes(_JPEG_HEAD)
        (tmp_path / "b.png").write_bytes(_PNG_HEAD)
        (tmp_path / "c.tif").write_bytes(_TIFF_LE_HEAD)
        (tmp_path / "d.TIFF").write_bytes(_TIFF_BE_HEAD)  # case-insensitive suffix, big-endian magic

        ds = load_flat_images(tmp_path)

        assert [s.image_id for s in ds.samples] == ["a.jpg", "b.png", "c.tif", "d.TIFF"]
        assert ds.task is Task.OD
        assert ds.num_detections == 0
        assert ds.dataset_metadata.taxonomy is None
        assert ds.dataset_id == "flat_images"
        assert all(s.detections == () for s in ds.samples)
        assert all(s.path_or_uri and s.file_name for s in ds.samples)
        assert all(s.metadata["source_format"] == "flat_images" for s in ds.samples)
        # Dimensions stay unset at load; MAITE decode fills them in.
        assert all((s.width, s.height) == (None, None) for s in ds.samples)

    def test_ignores_subdirectories_and_other_suffixes(self, tmp_path: Path) -> None:
        (tmp_path / "a.jpg").write_bytes(_JPEG_HEAD)
        (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")
        # SafeTensors ingest is deferred pending the standards change (#74).
        (tmp_path / "tensor.safetensors").write_bytes(b"\x00" * 16)
        nested = tmp_path / "nested"
        nested.mkdir()
        (nested / "ignored.jpg").write_bytes(_JPEG_HEAD)

        ds = load_flat_images(tmp_path)

        assert [s.image_id for s in ds.samples] == ["a.jpg"]

    def test_image_extensions_option_narrows_the_scan(self, tmp_path: Path) -> None:
        (tmp_path / "a.jpg").write_bytes(_JPEG_HEAD)
        (tmp_path / "b.png").write_bytes(_PNG_HEAD)

        only_jpg = load_flat_images(tmp_path, image_extensions=".jpg")
        also_bare = load_flat_images(tmp_path, image_extensions=["jpg"])

        assert [s.image_id for s in only_jpg.samples] == ["a.jpg"]
        assert [s.image_id for s in also_bare.samples] == ["a.jpg"]


class TestFlatImagesMalformedInputs:
    def test_missing_or_empty_root_returns_empty_dataset(self, tmp_path: Path, caplog) -> None:  # type: ignore[no-untyped-def]
        with caplog.at_level(logging.WARNING, logger="datamaite._formats.flat_images.loader"):
            missing = load_flat_images(tmp_path / "missing")
            empty = load_flat_images(tmp_path)

        assert len(missing) == 0
        assert len(empty) == 0
        assert "not a directory" in caplog.text
        assert "No immediate image files" in caplog.text

    def test_skips_files_whose_magic_does_not_match_the_suffix(self, tmp_path: Path, caplog) -> None:  # type: ignore[no-untyped-def]
        (tmp_path / "good.jpg").write_bytes(_JPEG_HEAD)
        (tmp_path / "bad.jpg").write_bytes(b"plainly not a jpeg")
        (tmp_path / "bad.png").write_bytes(_JPEG_HEAD)  # jpeg bytes behind a .png suffix
        (tmp_path / "empty.tif").write_bytes(b"")

        with caplog.at_level(logging.WARNING, logger="datamaite._formats.flat_images.loader"):
            ds = load_flat_images(tmp_path)

        assert [s.image_id for s in ds.samples] == ["good.jpg"]
        assert "does not match its .jpg suffix" in caplog.text
        assert "does not match its .png suffix" in caplog.text
        assert "empty flat image file" in caplog.text


class TestFlatImagesMaiteDecode:
    def test_encoded_image_decodes_through_opencv(self, tmp_path: Path) -> None:
        cv2 = pytest.importorskip("cv2")
        source = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
        ok, buf = cv2.imencode(".png", source)  # PNG is lossless: exact roundtrip
        assert ok
        (tmp_path / "real.png").write_bytes(buf.tobytes())

        ds = load_flat_images(tmp_path)
        image, target, meta = ds[0]

        assert image.shape == (3, 4, 6)
        assert image.dtype == np.uint8
        # decode_image returns RGB CHW; cv2.imencode consumed BGR.
        np.testing.assert_array_equal(np.transpose(image, (1, 2, 0)), source[:, :, ::-1])
        assert target.boxes.shape == (0, 4)  # type: ignore[attr-defined]
        assert meta == {"id": "real.png", "height": 4, "width": 6}
