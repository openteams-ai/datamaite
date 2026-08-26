"""Tests for the flat-folder still-image loader (IR-3.2-S-1)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from datamaite import DatasetFormat, FlatImagesLoader, Task, load, load_od
from datamaite._formats.flat_images.loader import load_flat_images
from datamaite._io import probe_image_dimensions
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

    def test_malformed_image_filtering_matches_remote(self, tmp_path: Path, memory_root) -> None:  # type: ignore[no-untyped-def]
        local_root = tmp_path / "local"
        local_root.mkdir()
        payloads = {"a.png": _PNG_HEAD, "bogus.png": b"not a png", "empty.png": b""}
        remote_root = memory_root / "flat-image-parity"
        remote_root.mkdir(parents=True)
        for name, payload in payloads.items():
            (local_root / name).write_bytes(payload)
            (remote_root / name).write_bytes(payload)

        local = load_flat_images(local_root)
        remote = load_flat_images(remote_root)

        assert [sample.image_id for sample in local.samples] == ["a.png"]
        assert [sample.image_id for sample in remote.samples] == ["a.png"]


class TestImageHeaderDimensions:
    @pytest.mark.parametrize("extension", [".png", ".jpg", ".webp", ".tiff"])
    def test_real_encoded_dimensions(self, extension: str) -> None:
        cv2 = pytest.importorskip("cv2")
        source = np.zeros((37, 53, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(extension, source)
        if not ok:
            pytest.skip(f"OpenCV cannot encode {extension}")

        assert probe_image_dimensions(encoded.tobytes()) == (53, 37)


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
        assert meta["id"] == "real.png"
        assert meta["height"] == 4
        assert meta["width"] == 6
        # #79: the loader's source-preserving per-image passthrough is now
        # surfaced on the datum metadata as flat keys.
        assert meta["source_format"] == "flat_images"

    def test_memory_root_loads_and_decodes_through_shared_fsspec_path(self, memory_root) -> None:  # type: ignore[no-untyped-def]
        """Format discovery and lazy media decode share one UPath."""
        cv2 = pytest.importorskip("cv2")
        source = np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3)
        ok, buf = cv2.imencode(".png", source)
        assert ok

        root = memory_root / "flat-images"
        root.mkdir(parents=True)
        (root / "remote.png").write_bytes(buf.tobytes())
        storage_options = {"poc_marker": "must-not-appear-in-repr"}

        # Exercise both task-first and generic explicit dispatch. The loader's
        # supports_remote capability replaces the former HMIE-only policy.
        ds = load_od(str(root), dataset_format="flat_images", storage_options=storage_options)
        generic = load(str(root), dataset_format="flat_images", storage_options=storage_options)

        assert len(ds) == len(generic) == 1
        assert isinstance(ds.samples[0].path_or_uri, str)
        assert ds.samples[0].path_or_uri.startswith("memory://")
        assert "must-not-appear-in-repr" not in repr(ds)

        image, target, meta = ds[0]
        np.testing.assert_array_equal(np.transpose(image, (1, 2, 0)), source[:, :, ::-1])
        assert target.boxes.shape == (0, 4)  # type: ignore[attr-defined]
        assert meta["id"] == "remote.png"

    def test_remote_metadata_reads_bounded_header_without_decoding(self, memory_root, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        cv2 = pytest.importorskip("cv2")
        source = np.zeros((64, 96, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", source)
        assert ok
        root = memory_root / "metadata-header"
        image_path = root / "large.png"
        image_path.parent.mkdir(parents=True)
        image_path.write_bytes(encoded.tobytes() + b"\0" * (1024 * 1024))
        dataset = load_flat_images(root)
        calls: list[tuple[int | None, int | None]] = []
        original_cat_file = root.fs.cat_file

        def counted_cat_file(path, start=None, end=None, **kwargs):  # type: ignore[no-untyped-def]
            calls.append((start, end))
            return original_cat_file(path, start=start, end=end, **kwargs)

        def fail_decode(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("get_metadata must not decode image pixels")

        monkeypatch.setattr(root.fs, "cat_file", counted_cat_file)
        monkeypatch.setattr("datamaite.maite._od.od_input", fail_decode)

        metadata = dataset.get_metadata(0)

        assert (metadata["width"], metadata["height"]) == (96, 64)
        assert calls == [(0, 32)]
        assert dataset.samples[0].width is None
        assert dataset.samples[0].height is None

    def test_memory_discovery_reads_only_magic_prefix(self, memory_root, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        root = memory_root / "lazy-flat-images"
        root.mkdir(parents=True)
        image_path = root / "remote.png"
        image_path.write_bytes(_PNG_HEAD)
        calls: list[tuple[int | None, int | None]] = []
        original_cat_file = root.fs.cat_file

        def counted_cat_file(path, start=None, end=None, **kwargs):  # type: ignore[no-untyped-def]
            calls.append((start, end))
            return original_cat_file(path, start=start, end=end, **kwargs)

        monkeypatch.setattr(root.fs, "cat_file", counted_cat_file)
        ds = load_od(str(root), dataset_format="flat_images")

        assert [sample.image_id for sample in ds.samples] == ["remote.png"]
        assert calls == [(0, 8)]

    def test_empty_remote_image_is_skipped_at_load(self, memory_root, caplog) -> None:  # type: ignore[no-untyped-def]
        root = memory_root / "malformed-flat-images"
        root.mkdir(parents=True)
        (root / "empty.png").write_bytes(b"")

        with caplog.at_level(logging.WARNING, logger="datamaite._formats.flat_images.loader"):
            ds = load_od(str(root), dataset_format="flat_images")

        assert ds.sample_count == 0
        assert "empty flat image file" in caplog.text
