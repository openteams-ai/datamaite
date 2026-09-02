"""Tests for the flat-folder still-image loader (IR-3.2-S-1)."""

from __future__ import annotations

import json
import logging
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

from datamaite import DatasetFormat, FlatImagesLoader, Task, load, load_od
from datamaite import _safetensors as _safetensors_module
from datamaite._formats.flat_images.loader import load_flat_images
from datamaite._io import probe_image_dimensions
from datamaite._safetensors import load_image_rgb_hwc, read_entries
from datamaite.loaders import available_formats, get_loader
from datamaite.object_detection import ObjectDetectionDataset

_JPEG_HEAD = b"\xff\xd8\xff\xe0" + b"\x00" * 20
_PNG_HEAD = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
_TIFF_LE_HEAD = b"II*\x00" + b"\x00" * 20
_TIFF_BE_HEAD = b"MM\x00*" + b"\x00" * 20

_ST_DTYPE_CODES = {
    "uint8": "U8",
    "uint16": "U16",
    "int8": "I8",
    "int16": "I16",
    "uint32": "U32",
    "float16": "F16",
    "float32": "F32",
    "float64": "F64",
    "bool": "BOOL",
}

_LOADER_LOG = "datamaite._formats.flat_images.loader"


def _safetensors_bytes(tensors: dict[str, np.ndarray], metadata: dict[str, str] | None = None) -> bytes:
    """Serialize named numpy arrays as a spec-conformant safetensors file."""
    header: dict[str, object] = {}
    payload = b""
    for name, array in tensors.items():
        # The wire format is little-endian regardless of host, and tobytes() is
        # host-endian; be explicit so multi-byte dtypes are right everywhere.
        # No copy=False: a big-endian host has to copy to byteswap at all.
        data = array.astype(array.dtype.newbyteorder("<")).tobytes()
        header[name] = {
            "dtype": _ST_DTYPE_CODES[str(array.dtype)],
            "shape": list(array.shape),
            "data_offsets": [len(payload), len(payload) + len(data)],
        }
        payload += data
    if metadata is not None:
        header["__metadata__"] = metadata
    raw = json.dumps(header).encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw + payload


def _write_safetensors(path: Path, tensors: dict[str, np.ndarray], metadata: dict[str, str] | None = None) -> Path:
    path.write_bytes(_safetensors_bytes(tensors, metadata))
    return path


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
        _write_safetensors(tmp_path / "e.safetensors", {"image": np.zeros((4, 6, 3), dtype=np.uint8)})

        ds = load_flat_images(tmp_path)

        assert [s.image_id for s in ds.samples] == ["a.jpg", "b.png", "c.tif", "d.TIFF", "e.safetensors#image"]
        assert ds.task is Task.OD
        assert ds.num_detections == 0
        assert ds.dataset_metadata.taxonomy is None
        assert ds.dataset_id == "flat_images"
        assert all(s.detections == () for s in ds.samples)
        assert all(s.path_or_uri and s.file_name for s in ds.samples)
        assert all(s.metadata["source_format"] == "flat_images" for s in ds.samples)
        # Encoded dimensions stay unset at load; MAITE decode fills them in.
        # SafeTensors dimensions come from the header, which we do parse.
        encoded = [s for s in ds.samples if not s.file_name.endswith(".safetensors")]
        assert all((s.width, s.height) == (None, None) for s in encoded)
        assert (ds.samples[-1].width, ds.samples[-1].height) == (6, 4)

    def test_ignores_subdirectories_and_other_suffixes(self, tmp_path: Path) -> None:
        (tmp_path / "a.jpg").write_bytes(_JPEG_HEAD)
        (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")
        # SafeTensors is now ingested (#74), so it must survive this scan while
        # the .txt and the nested image do not.
        _write_safetensors(tmp_path / "tensor.safetensors", {"image": np.zeros((2, 2, 3), dtype=np.uint8)})
        nested = tmp_path / "nested"
        nested.mkdir()
        (nested / "ignored.jpg").write_bytes(_JPEG_HEAD)

        ds = load_flat_images(tmp_path)

        assert [s.image_id for s in ds.samples] == ["a.jpg", "tensor.safetensors#image"]

    def test_safetensors_dimensions_come_from_the_header(self, tmp_path: Path) -> None:
        _write_safetensors(tmp_path / "hwc.safetensors", {"image": np.zeros((4, 6, 3), dtype=np.uint8)})
        _write_safetensors(tmp_path / "chw.safetensors", {"image": np.zeros((3, 7, 9), dtype=np.float32)})
        _write_safetensors(tmp_path / "hw.safetensors", {"image": np.zeros((5, 8), dtype=np.uint16)})
        (tmp_path / "a.jpg").write_bytes(_JPEG_HEAD)

        by_id = {s.image_id: s for s in load_flat_images(tmp_path).samples}

        assert (by_id["hwc.safetensors#image"].width, by_id["hwc.safetensors#image"].height) == (6, 4)
        assert (by_id["chw.safetensors#image"].width, by_id["chw.safetensors#image"].height) == (9, 7)
        assert (by_id["hw.safetensors#image"].width, by_id["hw.safetensors#image"].height) == (8, 5)
        # Encoded images stay undimensioned at load; MAITE decode fills them in.
        assert (by_id["a.jpg"].width, by_id["a.jpg"].height) == (None, None)

    def test_multi_tensor_safetensors_yields_one_sample_per_image(self, tmp_path: Path) -> None:
        _write_safetensors(
            tmp_path / "batch.safetensors",
            {
                "b_second": np.zeros((2, 2, 3), dtype=np.uint8),
                "a_first": np.zeros((3, 5), dtype=np.float32),
                "not_an_image": np.zeros((7,), dtype=np.float32),  # 1-D: skipped, not an image
            },
        )

        ds = load_flat_images(tmp_path)

        assert [s.image_id for s in ds.samples] == ["batch.safetensors#a_first", "batch.safetensors#b_second"]
        assert [s.metadata["safetensors_key"] for s in ds.samples] == ["a_first", "b_second"]
        assert all(s.file_name == "batch.safetensors" for s in ds.samples)

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

    def test_skips_malformed_safetensors_files(self, tmp_path: Path, caplog) -> None:  # type: ignore[no-untyped-def]
        """Container-level damage skips the file; a valid non-image file is skipped as empty."""
        (tmp_path / "truncated.safetensors").write_bytes(b"\x01\x02")  # shorter than the length prefix
        (tmp_path / "bad-json.safetensors").write_bytes(struct.pack("<Q", 4) + b"nope")
        (tmp_path / "lying-length.safetensors").write_bytes(struct.pack("<Q", 10_000) + b"{}")
        # Readable offsets that run past the 4-byte payload: container-fatal, not a per-tensor drop.
        raw = json.dumps({"t": {"dtype": "U8", "shape": [2, 2, 3], "data_offsets": [0, 12]}}).encode()
        (tmp_path / "bad-offsets.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + b"\x00" * 4)
        # Valid cover: F32 vector + BF16 image-shaped tensor (itemsize 2) 0→32→56 over 56 bytes.
        raw = json.dumps(
            {
                "vector": {"dtype": "F32", "shape": [8], "data_offsets": [0, 32]},
                "bf16": {"dtype": "BF16", "shape": [2, 2, 3], "data_offsets": [32, 56]},
            }
        ).encode()
        (tmp_path / "no-image.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + b"\x00" * 56)

        with caplog.at_level(logging.WARNING):
            ds = load_flat_images(tmp_path)

        assert len(ds) == 0
        assert caplog.text.count("Skipping malformed safetensors file") == 4
        assert caplog.text.count("Skipping malformed safetensors tensor") == 0
        assert caplog.text.count("no image-shaped tensor") == 1  # no-image.safetensors only

    def test_byte_span_mismatch_is_container_invalid(self, tmp_path: Path, caplog) -> None:  # type: ignore[no-untyped-def]
        # Shape says 12 uint8 values but the span covers 6 bytes — official cover fails the file.
        raw = json.dumps({"t": {"dtype": "U8", "shape": [2, 2, 3], "data_offsets": [0, 6]}}).encode()
        (tmp_path / "short-span.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + b"\x00" * 6)

        with caplog.at_level(logging.WARNING):
            ds = load_flat_images(tmp_path)

        assert len(ds) == 0
        assert "Skipping malformed safetensors file" in caplog.text
        assert "byte span that does not match its shape" in caplog.text

    def test_pixel_count_over_the_cap_is_not_an_image(self, tmp_path: Path, monkeypatch, caplog) -> None:  # type: ignore[no-untyped-def]
        """H*W past the decode cap is refused by shape alone, before any allocation.

        The cap is lowered rather than writing a huge file: a header claiming
        2e12 pixels would fail the offset bounds check first (its span cannot
        fit a small file), so that route never reaches the pixel guard.
        """
        _write_safetensors(tmp_path / "wide.safetensors", {"image": np.zeros((4, 6, 3), dtype=np.uint8)})
        monkeypatch.setattr("datamaite._safetensors.MAX_DECODED_IMAGE_PIXELS", 8)  # 4*6 = 24 > 8

        with caplog.at_level(logging.WARNING, logger=_LOADER_LOG):
            ds = load_flat_images(tmp_path)

        assert len(ds) == 0
        assert "no image-shaped tensor" in caplog.text

    def test_pixel_cap_also_applies_at_decode(self, tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        """A cached header is no more trusted than a fresh one: the bound holds at decode too."""
        path = _write_safetensors(tmp_path / "wide.safetensors", {"image": np.zeros((4, 6, 3), dtype=np.uint8)})
        monkeypatch.setattr("datamaite._safetensors.MAX_DECODED_IMAGE_PIXELS", 8)

        with pytest.raises(ValueError, match="is not decodable as an image"):
            load_image_rgb_hwc(path, "image")

    def test_over_cap_image_is_reported_as_over_cap(self, tmp_path: Path, monkeypatch, caplog) -> None:  # type: ignore[no-untyped-def]
        """An image-shaped tensor too big to decode says so, instead of hiding behind
        the file-level "no image-shaped tensor" line."""
        _write_safetensors(tmp_path / "big.safetensors", {"image": np.zeros((4, 6, 3), dtype=np.uint8)})
        monkeypatch.setattr("datamaite._safetensors.MAX_ENCODED_IMAGE_BYTES", 8)  # 4*6*3 = 72 output bytes

        with caplog.at_level(logging.WARNING):
            ds = load_flat_images(tmp_path)

        assert len(ds) == 0
        assert "over the 8 byte cap" in caplog.text
        assert "Skipping image-shaped safetensors tensor 'image'" in caplog.text

    def test_source_span_over_cap_is_not_loaded_as_an_unusable_sample(
        self, tmp_path: Path, monkeypatch, caplog
    ) -> None:  # type: ignore[no-untyped-def]
        # U16 source is 24 bytes while its normalized RGB output is 12 bytes.
        _write_safetensors(tmp_path / "wide.safetensors", {"image": np.zeros((2, 2, 3), dtype=np.uint16)})
        monkeypatch.setattr("datamaite._safetensors.MAX_ENCODED_IMAGE_BYTES", 16)

        with caplog.at_level(logging.WARNING):
            ds = load_flat_images(tmp_path)

        assert len(ds) == 0
        assert "source span is 24 bytes" in caplog.text
        assert "over the 16 byte cap" in caplog.text

    def test_malformed_safetensors_filtering_matches_remote(self, tmp_path: Path, memory_root) -> None:  # type: ignore[no-untyped-def]
        local_root = tmp_path / "local"
        local_root.mkdir()
        payloads = {
            "good.safetensors": _safetensors_bytes({"image": np.zeros((2, 3, 3), dtype=np.uint8)}),
            "truncated.safetensors": b"\x01\x02",
            "no-image.safetensors": _safetensors_bytes({"vector": np.zeros((8,), dtype=np.float32)}),
        }
        remote_root = memory_root / "flat-safetensors-parity"
        remote_root.mkdir(parents=True)
        for name, payload in payloads.items():
            (local_root / name).write_bytes(payload)
            (remote_root / name).write_bytes(payload)

        local = load_flat_images(local_root)
        remote = load_flat_images(remote_root)

        assert [s.image_id for s in local.samples] == ["good.safetensors#image"]
        assert [s.image_id for s in remote.samples] == ["good.safetensors#image"]
        assert [(s.width, s.height) for s in remote.samples] == [(3, 2)]


class TestSafeTensorsReaderGuards:
    """Header-parse and decode guards not reachable through a plain load."""

    def test_unstattable_resource_is_malformed(self, tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        # Offsets are bounds-checked against the real size, so a resource we
        # cannot stat is one we cannot validate.
        path = _write_safetensors(tmp_path / "img.safetensors", {"image": np.zeros((2, 2, 3), dtype=np.uint8)})

        def unknown_size(*_args, **_kwargs):  # type: ignore[no-untyped-def]
            return None

        monkeypatch.setattr("datamaite._safetensors.resource_size", unknown_size)

        with pytest.raises(ValueError, match="could not determine the size"):
            read_entries(path)

    def _write_raw_header(self, path: Path, header: str, payload: bytes = b"\x00" * 4) -> Path:
        raw = header.encode()
        path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
        return path

    @pytest.mark.parametrize("header", ["[]", '"a string"', "12"])
    def test_container_damage_raises(self, tmp_path: Path, header: str) -> None:
        """A header that is not an object makes the whole file untrustworthy."""
        path = self._write_raw_header(tmp_path / "bad.safetensors", header)

        with pytest.raises(ValueError, match="is not a JSON object"):
            read_entries(path)

    @pytest.mark.parametrize(
        ("spec", "message"),
        [
            ("5", "is not a JSON object"),
            ('{"shape": [2, 2], "data_offsets": [0, 4]}', "has no dtype string"),
            ('{"dtype": "U8", "shape": "2x2", "data_offsets": [0, 4]}', "has an invalid shape"),
            ('{"dtype": "U8", "shape": [2, -2], "data_offsets": [0, 4]}', "has an invalid shape"),
            ('{"dtype": "U8", "shape": [2, true], "data_offsets": [0, 4]}', "has an invalid shape"),
            ('{"dtype": "U8", "shape": [2, 2]}', "data_offsets outside the file"),
            ('{"dtype": "U8", "shape": [2, 2], "data_offsets": [4, 0]}', "data_offsets outside the file"),
            ('{"dtype": "U8", "shape": [2, 2], "data_offsets": [0, true]}', "data_offsets outside the file"),
        ],
    )
    def test_invalid_tensor_spec_fails_the_container(
        self,
        tmp_path: Path,
        spec: str,
        message: str,
    ) -> None:
        """An unreadable sibling is container-fatal; valid tensors in that file do not load."""
        header = f'{{"bad": {spec}, "good": {{"dtype": "U8", "shape": [2, 2], "data_offsets": [0, 4]}}}}'
        path = self._write_raw_header(tmp_path / "mixed.safetensors", header)

        with pytest.raises(ValueError, match=message):
            read_entries(path)

    def test_broken_sibling_skips_the_whole_file(self, tmp_path: Path, caplog) -> None:  # type: ignore[no-untyped-def]
        """An unreadable tensor spec fails the container; the good sibling does not load."""
        good = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        header = json.dumps(
            {
                "image": {"dtype": "U8", "shape": [2, 2, 3], "data_offsets": [0, 12]},
                "broken": 5,
            }
        ).encode()
        (tmp_path / "mixed.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + good.tobytes())

        with caplog.at_level(logging.WARNING):
            ds = load_flat_images(tmp_path)

        assert len(ds) == 0
        assert "Skipping malformed safetensors file" in caplog.text
        assert "is not a JSON object" in caplog.text

    def test_decode_rejects_missing_and_non_image_tensors(self, tmp_path: Path) -> None:
        path = _write_safetensors(
            tmp_path / "img.safetensors",
            {"image": np.zeros((2, 2, 3), dtype=np.uint8), "vector": np.zeros((8,), dtype=np.float32)},
        )

        with pytest.raises(ValueError, match="has no tensor named 'absent'"):
            load_image_rgb_hwc(path, "absent")
        with pytest.raises(ValueError, match="is not decodable as an image"):
            load_image_rgb_hwc(path, "vector")

    def test_remote_tensor_over_the_read_cap_is_refused(self, memory_root, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        root = memory_root / "safetensors-cap"
        root.mkdir(parents=True)
        path = root / "img.safetensors"
        path.write_bytes(_safetensors_bytes({"image": np.zeros((2, 2, 3), dtype=np.uint8)}))
        monkeypatch.setattr("datamaite._safetensors.MAX_ENCODED_IMAGE_BYTES", 4)
        monkeypatch.setattr("datamaite._safetensors.MAX_DECODED_IMAGE_PIXELS", 1_000_000)

        with pytest.raises(ValueError, match="over the 4 byte cap"):
            load_image_rgb_hwc(path, "image")

    def test_local_tensor_is_subject_to_the_byte_cap(self, tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        path = _write_safetensors(tmp_path / "img.safetensors", {"image": np.zeros((2, 2, 3), dtype=np.uint8)})
        monkeypatch.setattr("datamaite._safetensors.MAX_ENCODED_IMAGE_BYTES", 4)
        monkeypatch.setattr("datamaite._safetensors.MAX_DECODED_IMAGE_PIXELS", 1_000_000)

        with pytest.raises(ValueError, match="over the 4 byte cap"):
            load_image_rgb_hwc(path, "image")

    def test_output_bytes_cap_fires_when_span_and_pixels_pass(self, tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        """Grayscale U8: source and H*W pass, H*W*3 fails. Patch both production caps."""
        path = _write_safetensors(tmp_path / "gray.safetensors", {"image": np.zeros((10, 10), dtype=np.uint8)})
        monkeypatch.setattr("datamaite._safetensors.MAX_ENCODED_IMAGE_BYTES", 200)  # 100 source, 300 output
        monkeypatch.setattr("datamaite._safetensors.MAX_DECODED_IMAGE_PIXELS", 10_000)

        with pytest.raises(ValueError, match="over the 200 byte cap"):
            load_image_rgb_hwc(path, "image")

    def test_short_tensor_read_is_truncated_not_reshaped(self, tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        # A writer racing the decode can shorten the span after the header
        # validates; a short buffer must raise, never reach reshape().
        path = _write_safetensors(tmp_path / "img.safetensors", {"image": np.zeros((2, 2, 3), dtype=np.uint8)})
        original = _safetensors_module.read_resource_range

        def short_tensor_read(target, start, end, storage_options=None):  # type: ignore[no-untyped-def]
            payload = original(target, start, end, storage_options)
            return payload[:-1] if start > 8 else payload

        monkeypatch.setattr("datamaite._safetensors.read_resource_range", short_tensor_read)

        with pytest.raises(ValueError, match="is truncated"):
            load_image_rgb_hwc(path, "image")


class TestSafeTensorsLayoutConvention:
    """The convention is a shape heuristic; ``__metadata__`` is opaque (#74)."""

    def test_ambiguous_shape_reads_as_hwc_and_warns(self, tmp_path: Path, caplog) -> None:  # type: ignore[no-untyped-def]
        _write_safetensors(tmp_path / "tile.safetensors", {"image": np.zeros((3, 3, 3), dtype=np.uint8)})

        with caplog.at_level(logging.WARNING, logger=_LOADER_LOG):
            ds = load_flat_images(tmp_path)

        assert (ds.samples[0].width, ds.samples[0].height) == (3, 3)
        assert "ambiguous shape" in caplog.text
        assert "assuming HWC" in caplog.text

    def test_unambiguous_chw_does_not_warn(self, tmp_path: Path, caplog) -> None:  # type: ignore[no-untyped-def]
        # (3, 2, 2): the last dim is not channel-like, so CHW is the only read.
        _write_safetensors(tmp_path / "chw.safetensors", {"image": np.zeros((3, 2, 2), dtype=np.uint8)})

        with caplog.at_level(logging.WARNING, logger=_LOADER_LOG):
            ds = load_flat_images(tmp_path)

        assert (ds.samples[0].width, ds.samples[0].height) == (2, 2)
        assert "ambiguous" not in caplog.text

    def test_metadata_map_cannot_override_the_layout(self, tmp_path: Path, caplog) -> None:  # type: ignore[no-untyped-def]
        """A producer's private ``__metadata__`` keys are ignored, by decision.

        safetensors' metadata is one file-level string map: it cannot describe
        the individual tensors of a multi-image file, and the ecosystem has no
        image vocabulary for it. Honouring our own keys would make files that
        work for us scramble for every other IR-3.2-S-1 consumer.
        """
        _write_safetensors(
            tmp_path / "hinted.safetensors",
            {"image": np.arange(27, dtype=np.uint8).reshape(3, 3, 3)},
            metadata={"layout": "chw", "channels": "bgr", "format": "pt"},
        )

        with caplog.at_level(logging.WARNING, logger=_LOADER_LOG):
            ds = load_flat_images(tmp_path)

        assert [s.image_id for s in ds.samples] == ["hinted.safetensors#image"]
        assert "ambiguous shape" in caplog.text  # still HWC, still a visible guess
        image, _, _ = ds[0]
        # HWC read of the source: channel c of pixel (h, w) is source[h, w, c].
        np.testing.assert_array_equal(np.transpose(image, (1, 2, 0)), np.arange(27, dtype=np.uint8).reshape(3, 3, 3))


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


class TestSafeTensorsMaiteDecode:
    """SafeTensors samples decode through numpy alone -- no OpenCV needed."""

    def _single_item(self, tmp_path: Path, array: np.ndarray) -> tuple[np.ndarray, object, dict]:
        _write_safetensors(tmp_path / "img.safetensors", {"image": array})
        ds = load_flat_images(tmp_path)
        assert len(ds) == 1
        return ds[0]

    def test_uint8_hwc_roundtrips_exactly(self, tmp_path: Path) -> None:
        source = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)

        image, target, meta = self._single_item(tmp_path, source)

        assert image.shape == (3, 4, 6)  # CHW
        assert image.dtype == np.uint8
        np.testing.assert_array_equal(np.transpose(image, (1, 2, 0)), source)
        assert target.boxes.shape == (0, 4)  # type: ignore[attr-defined]
        assert meta["id"] == "img.safetensors#image"
        assert (meta["height"], meta["width"]) == (4, 6)
        assert meta["safetensors_key"] == "image"

    def test_decodes_without_opencv(self, tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        """The cv2 import must stay branched behind the source kind.

        A safetensors tensor is numpy-decodable, so indexing one must not
        require the ``datamaite[od]`` extra the encoded path needs.
        """
        source = np.arange(2 * 2 * 3, dtype=np.uint8).reshape(2, 2, 3)
        _write_safetensors(tmp_path / "img.safetensors", {"image": source})
        (tmp_path / "encoded.png").write_bytes(_PNG_HEAD)
        ds = load_flat_images(tmp_path)
        by_id = {sample.image_id: index for index, sample in enumerate(ds.samples)}
        monkeypatch.setitem(sys.modules, "cv2", None)

        image, _, _ = ds[by_id["img.safetensors#image"]]

        np.testing.assert_array_equal(np.transpose(image, (1, 2, 0)), source)
        with pytest.raises(ImportError, match="needs OpenCV"):
            ds[by_id["encoded.png"]]

    def test_unit_range_float_chw_scales_to_255(self, tmp_path: Path) -> None:
        source = np.full((3, 2, 2), 0.5, dtype=np.float32)

        image, _, _ = self._single_item(tmp_path, source)

        assert image.dtype == np.uint8
        assert image.min() == image.max() == 128  # rint(0.5 * 255)

    def test_wide_range_float_clips_instead_of_scaling(self, tmp_path: Path) -> None:
        source = np.array([[[-40.0, 0.0], [128.0, 300.0]]], dtype=np.float64).reshape(2, 2, 1)

        image, _, _ = self._single_item(tmp_path, source)

        assert sorted(np.unique(image).tolist()) == [0, 128, 255]

    def test_uint16_scales_instead_of_saturating(self, tmp_path: Path) -> None:
        # A genuine high-bit-depth pixel (4096) must scale down proportionally,
        # not saturate to 255 (which would turn drone/satellite imagery to
        # near-white garbage while still advertising the dtype as supported).
        source = np.array([[0, 4096], [255, 65535]], dtype=np.uint16).reshape(2, 2, 1)

        image, _, _ = self._single_item(tmp_path, source)

        assert image.dtype == np.uint8
        # value 4096 -> round(4096 * 255 / 65535) == 16, NOT 255.
        assert image[0, 0, 1] == 16
        assert image[0, 0, 0] == 0  # 0 stays 0
        assert image[0, 1, 1] == 255  # the true max maps to 255

    def test_int16_clamps_so_zero_stays_black(self, tmp_path: Path) -> None:
        source = np.array([[-32768, 0], [4096, 32767]], dtype=np.int16).reshape(2, 2, 1)

        image, _, _ = self._single_item(tmp_path, source)

        assert image.dtype == np.uint8
        assert image[0, 0, 0] == 0  # int16 min clamps to black
        assert image[0, 0, 1] == 0  # genuine 0 stays black, not mid-grey
        assert image[0, 1, 0] == 255  # 4096 clips to white
        assert image[0, 1, 1] == 255

    def test_int8_clamps_rather_than_rescaling(self, tmp_path: Path) -> None:
        # int8 already fits 8-bit precision, so rescaling its range would move a
        # genuine 0 to mid-grey. Clamp instead: negatives go black, 0..127 pass.
        source = np.array([[0, 10], [-1, 127]], dtype=np.int8).reshape(2, 2, 1)

        image, _, _ = self._single_item(tmp_path, source)

        assert image.dtype == np.uint8
        assert [int(v) for v in image[0].ravel()] == [0, 10, 0, 127]

    def test_grayscale_replicates_and_rgba_drops_alpha(self, tmp_path: Path) -> None:
        gray = np.arange(4, dtype=np.uint8).reshape(2, 2)
        rgba = np.zeros((2, 2, 4), dtype=np.uint8)
        rgba[..., 3] = 255  # alpha channel must not leak into RGB

        gray_img, _, _ = self._single_item(tmp_path, gray)
        (tmp_path / "img.safetensors").unlink()
        rgba_img, _, _ = self._single_item(tmp_path, rgba)

        assert gray_img.shape == (3, 2, 2)
        np.testing.assert_array_equal(gray_img[0], gray_img[1])
        np.testing.assert_array_equal(gray_img[0], gray_img[2])
        assert rgba_img.shape == (3, 2, 2)
        assert rgba_img.max() == 0

    def test_bool_maps_to_0_and_255(self, tmp_path: Path) -> None:
        source = np.array([[True, False], [False, True]])

        image, _, _ = self._single_item(tmp_path, source)

        assert sorted(np.unique(image).tolist()) == [0, 255]

    def test_multi_tensor_samples_decode_their_own_tensor(self, tmp_path: Path) -> None:
        first = np.zeros((2, 2, 3), dtype=np.uint8)
        second = np.full((2, 2, 3), 7, dtype=np.uint8)
        _write_safetensors(tmp_path / "batch.safetensors", {"a_first": first, "b_second": second})

        ds = load_flat_images(tmp_path)

        np.testing.assert_array_equal(np.transpose(ds[0][0], (1, 2, 0)), first)
        np.testing.assert_array_equal(np.transpose(ds[1][0], (1, 2, 0)), second)

    def test_file_rewritten_between_load_and_decode_fails_loudly(self, tmp_path: Path) -> None:
        _write_safetensors(tmp_path / "img.safetensors", {"image": np.zeros((2, 2, 3), dtype=np.uint8)})
        ds = load_flat_images(tmp_path)
        (tmp_path / "img.safetensors").write_bytes(b"overwritten with junk")

        with pytest.raises(OSError, match="could not decode image for sample"):
            ds[0]

    def test_remote_load_reads_header_only_and_decode_adds_the_slice(self, memory_root, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        """Load transfers no tensor bytes; decode range-reads the tensor.

        Header bytes repeat on the first decode (fingerprint miss); a second
        index of the same file must not re-fetch the header.
        """
        payload = _safetensors_bytes({"image": np.arange(12, dtype=np.uint8).reshape(2, 2, 3)})
        root = memory_root / "safetensors-ranges"
        root.mkdir(parents=True)
        (root / "img.safetensors").write_bytes(payload)
        header_len = struct.unpack("<Q", payload[:8])[0]
        tensor_range = (8 + header_len, len(payload))
        calls: list[tuple[int | None, int | None]] = []
        original_cat_file = root.fs.cat_file

        def counted_cat_file(path, start=None, end=None, **kwargs):  # type: ignore[no-untyped-def]
            calls.append((start, end))
            return original_cat_file(path, start=start, end=end, **kwargs)

        monkeypatch.setattr(root.fs, "cat_file", counted_cat_file)
        ds = load_flat_images(root)

        assert calls == [(0, 8), (8, 8 + header_len)]  # header only, no tensor bytes
        assert (ds.samples[0].width, ds.samples[0].height) == (2, 2)

        calls.clear()
        image, _, _ = ds[0]

        assert tensor_range in calls  # the slice, and nothing wider
        assert all(rng[0] is not None and rng[1] is not None for rng in calls)
        np.testing.assert_array_equal(np.transpose(image, (1, 2, 0)), np.arange(12, dtype=np.uint8).reshape(2, 2, 3))

        header_calls_after_first_decode = len(calls)
        ds[0]
        extra = calls[header_calls_after_first_decode:]
        assert tensor_range in extra
        assert (0, 8) not in extra  # cache hit: no second header prefix read

    def test_decode_cache_reuses_name_index_without_copying_entries(self, tmp_path: Path) -> None:
        path = _write_safetensors(
            tmp_path / "batch.safetensors",
            {
                "first": np.zeros((2, 2, 3), dtype=np.uint8),
                "second": np.ones((2, 2, 3), dtype=np.uint8),
            },
        )

        first_lookup = _safetensors_module._entries_for_decode(path, None)
        second_lookup = _safetensors_module._entries_for_decode(path, None)

        assert first_lookup is second_lookup
        assert list(first_lookup) == ["first", "second"]
        assert first_lookup["second"].name == "second"

    def test_get_metadata_never_probes_or_decodes_safetensors(self, memory_root, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        """Header-derived dimensions keep get_metadata off the decode path.

        ``probe_image_dimensions`` only understands encoded headers, so an unset
        width/height would fall through to a full decode.
        """
        root = memory_root / "safetensors-metadata"
        root.mkdir(parents=True)
        (root / "img.safetensors").write_bytes(_safetensors_bytes({"image": np.zeros((7, 9, 3), dtype=np.uint8)}))
        ds = load_flat_images(root)

        def fail(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("get_metadata must not decode or probe a safetensors sample")

        monkeypatch.setattr("datamaite.maite._od.od_input", fail)
        monkeypatch.setattr("datamaite.object_detection.probe_image_dimensions", fail)

        metadata = ds.get_metadata(0)

        assert (metadata["width"], metadata["height"]) == (9, 7)


class TestSafeTensorsOracleAndGuards:
    def test_header_cap_rejects_before_the_body(self, tmp_path: Path) -> None:
        (tmp_path / "huge.safetensors").write_bytes(struct.pack("<Q", 100_000_001))

        with pytest.raises(ValueError, match="exceeds the 100000000 byte cap"):
            read_entries(tmp_path / "huge.safetensors")

    def test_four_d_tensor_is_not_an_image(self, tmp_path: Path, caplog) -> None:  # type: ignore[no-untyped-def]
        _write_safetensors(tmp_path / "batch.safetensors", {"image": np.zeros((2, 4, 6, 3), dtype=np.uint8)})

        with caplog.at_level(logging.WARNING, logger=_LOADER_LOG):
            ds = load_flat_images(tmp_path)

        assert len(ds) == 0
        assert "no image-shaped tensor" in caplog.text

    def test_uint32_is_not_an_image(self, tmp_path: Path, caplog) -> None:  # type: ignore[no-untyped-def]
        _write_safetensors(tmp_path / "wide.safetensors", {"image": np.zeros((2, 2, 3), dtype=np.uint32)})

        with caplog.at_level(logging.WARNING, logger=_LOADER_LOG):
            ds = load_flat_images(tmp_path)

        assert len(ds) == 0
        assert "no image-shaped tensor" in caplog.text

    def test_official_writer_roundtrip(self, tmp_path: Path) -> None:
        st_numpy = pytest.importorskip("safetensors.numpy")
        source = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        path = tmp_path / "official.safetensors"
        st_numpy.save_file({"image": source}, str(path))

        entries = read_entries(path)
        assert [e.name for e in entries] == ["image"]
        ds = load_flat_images(tmp_path)
        image, _, meta = ds[0]
        np.testing.assert_array_equal(np.transpose(image, (1, 2, 0)), source)
        assert meta["id"] == "official.safetensors#image"

    def test_official_multi_tensor_layouts_and_dtypes(self, tmp_path: Path) -> None:
        """One officially-written container across every layout and image dtype.

        ``safe_open`` is the oracle for names, shapes and dtypes; ``ds[i]`` then
        has to agree on the pixels, so the custom parser is checked against the
        reference implementation rather than against our own serializer.
        """
        st_numpy = pytest.importorskip("safetensors.numpy")
        from safetensors import safe_open

        tensors = {
            "chw_f32": np.full((3, 2, 2), 0.5, dtype=np.float32),
            "gray_hw": np.arange(6, dtype=np.uint8).reshape(2, 3),
            "hwc_u16": np.array([[0, 4096], [255, 65535]], dtype=np.uint16).reshape(2, 2, 1),
            "hwc_u8": np.arange(12, dtype=np.uint8).reshape(2, 2, 3),
            "mask_bool": np.array([[True, False], [False, True]]),
        }
        path = tmp_path / "official.safetensors"
        st_numpy.save_file(tensors, str(path))

        entries = {entry.name: entry for entry in read_entries(path)}
        with safe_open(str(path), framework="np") as official:
            official_names = list(official.keys())  # safe_open is not a Mapping
            assert sorted(entries) == sorted(official_names)
            for name in official_names:
                tensor = official.get_tensor(name)
                assert entries[name].shape == tensor.shape
                assert np.dtype(_safetensors_module._NUMPY_DTYPES[entries[name].dtype]) == tensor.dtype

        ds = load_flat_images(tmp_path)
        images = {sample.image_id: index for index, sample in enumerate(ds.samples)}
        assert sorted(images) == [f"official.safetensors#{name}" for name in sorted(tensors)]
        decoded = {name: ds[images[f"official.safetensors#{name}"]][0] for name in tensors}

        assert all(image.dtype == np.uint8 and image.shape[0] == 3 for image in decoded.values())
        np.testing.assert_array_equal(np.transpose(decoded["hwc_u8"], (1, 2, 0)), tensors["hwc_u8"])
        np.testing.assert_array_equal(decoded["gray_hw"][0], tensors["gray_hw"])  # replicated to 3 channels
        assert decoded["chw_f32"].min() == decoded["chw_f32"].max() == 128  # rint(0.5 * 255)
        assert [int(v) for v in decoded["hwc_u16"][0].ravel()] == [0, 16, 1, 255]  # scaled by the type max
        assert sorted(np.unique(decoded["mask_bool"]).tolist()) == [0, 255]

    def test_safe_open_rejects_overlap_and_gap(self, tmp_path: Path) -> None:
        pytest.importorskip("safetensors")
        from safetensors import SafetensorError, safe_open

        overlap = json.dumps(
            {
                "a": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]},
                "b": {"dtype": "U8", "shape": [4], "data_offsets": [2, 6]},
            }
        ).encode()
        gap = json.dumps(
            {
                "a": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]},
                "b": {"dtype": "U8", "shape": [2], "data_offsets": [4, 6]},
            }
        ).encode()
        unknown = json.dumps({"t": {"dtype": "NOTATYPE", "shape": [2], "data_offsets": [0, 2]}}).encode()
        unreadable = json.dumps({"t": {"dtype": "U8", "shape": [2], "data_offsets": "nope"}}).encode()
        cases = [
            (overlap, b"\x00" * 6),
            (gap, b"\x00" * 6),
            (unknown, b"\x00" * 2),
            (unreadable, b"\x00" * 2),
        ]
        for header, payload in cases:
            path = tmp_path / "hostile.safetensors"
            path.write_bytes(struct.pack("<Q", len(header)) + header + payload)
            with pytest.raises((ValueError, OSError)):
                read_entries(path)
            with pytest.raises(SafetensorError), safe_open(str(path), framework="np"):
                pass

    def test_recursion_error_fails_the_container(self, tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        path = _write_safetensors(tmp_path / "img.safetensors", {"image": np.zeros((2, 2, 3), dtype=np.uint8)})

        def boom(_raw: bytes) -> object:
            raise RecursionError("too deep")

        monkeypatch.setattr("datamaite._safetensors.json.loads", boom)

        with pytest.raises(ValueError, match="exceeds the parser recursion limit"):
            read_entries(path)

    def test_uncovered_trailing_bytes_fail_the_container(self, tmp_path: Path) -> None:
        header = json.dumps({"t": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]}}).encode()
        (tmp_path / "tail.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\x00" * 8)

        with pytest.raises(ValueError, match="not fully covered"):
            read_entries(tmp_path / "tail.safetensors")

    def test_distinct_storage_options_produce_distinct_fingerprints(self, memory_root) -> None:  # type: ignore[no-untyped-def]
        """Same display path, different storage_options must not share a cache key.

        Two bare ``memory://`` filesystems share a class-level store and
        ``_fs_token``, so they *should* share a manifest. Extra constructor
        kwargs change the token.
        """
        from datamaite._safetensors import _resource_fingerprint
        from datamaite._upath import to_dataset_path

        payload = _safetensors_bytes({"image": np.zeros((2, 2, 3), dtype=np.uint8)})
        (memory_root / "img.safetensors").write_bytes(payload)
        display = str(memory_root / "img.safetensors")
        first = to_dataset_path(display, {"auto_mkdir": True})
        second = to_dataset_path(display, {"auto_mkdir": False})
        assert _resource_fingerprint(first) != _resource_fingerprint(second)

    # Same name, same dtype, same payload -- only the shape list is permuted, so
    # the header JSON and the whole file keep their byte length. Rewriting with
    # different *pixels* would prove nothing: the offsets would be unchanged, so
    # a stale manifest still range-reads the new bytes correctly. Changing the
    # layout makes a stale manifest visibly misread them.
    _REWRITE_BEFORE = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)  # HWC
    _REWRITE_AFTER = np.arange(12, dtype=np.uint8).reshape(3, 2, 2)  # CHW

    def _assert_rewrite_seen(self, ds: ObjectDetectionDataset) -> None:
        """``ds[0]`` after the rewrite must be the CHW reading, not the cached HWC one."""
        stale = np.transpose(self._REWRITE_BEFORE, (2, 0, 1))
        after, _, _ = ds[0]
        np.testing.assert_array_equal(after, self._REWRITE_AFTER)
        assert not np.array_equal(after, stale)

    def test_same_size_rewrite_busts_the_decode_cache(self, tmp_path: Path) -> None:
        path = tmp_path / "img.safetensors"
        _write_safetensors(path, {"image": self._REWRITE_BEFORE})
        ds = load_flat_images(tmp_path)
        before, _, _ = ds[0]
        np.testing.assert_array_equal(before, np.transpose(self._REWRITE_BEFORE, (2, 0, 1)))

        _write_safetensors(path, {"image": self._REWRITE_AFTER})
        assert path.stat().st_size  # same size; st_mtime_ns is the only discriminator
        self._assert_rewrite_seen(ds)

    def test_same_size_rewrite_busts_the_decode_cache_remotely(self, memory_root) -> None:  # type: ignore[no-untyped-def]
        """The remote twin, on a backend that publishes no ETag.

        ``memory://`` info() carries only ``created``, so a fingerprint built
        from the older etag/version/mtime keys alone would collide on a
        same-size rewrite and serve the stale manifest.
        """
        root = memory_root / "st-rewrite"
        root.mkdir(parents=True)
        path = root / "img.safetensors"
        first_payload = _safetensors_bytes({"image": self._REWRITE_BEFORE})
        second_payload = _safetensors_bytes({"image": self._REWRITE_AFTER})
        assert len(first_payload) == len(second_payload)
        path.write_bytes(first_payload)
        ds = load_flat_images(root)
        ds[0]

        path.write_bytes(second_payload)
        self._assert_rewrite_seen(ds)


class TestSafeTensorsWriteReject:
    def test_write_rejects_safetensors_only(self, tmp_path: Path) -> None:
        _write_safetensors(tmp_path / "a.safetensors", {"image": np.zeros((2, 2, 3), dtype=np.uint8)})
        ds = load_flat_images(tmp_path)
        dest = tmp_path / "out"
        dest.mkdir()
        marker = dest / "keep.txt"
        marker.write_text("stay", encoding="utf-8")

        from datamaite import write

        with pytest.raises(ValueError, match=r"First offending sample: 'a\.safetensors#image'"):
            write(ds, dest, output_format="flat_images", mode="replace")

        assert marker.read_text(encoding="utf-8") == "stay"

    def test_convert_rejects_safetensors(self, tmp_path: Path) -> None:
        _write_safetensors(tmp_path / "a.safetensors", {"image": np.zeros((2, 2, 3), dtype=np.uint8)})
        dest = tmp_path / "out"
        dest.mkdir()
        (dest / "keep.txt").write_text("stay", encoding="utf-8")

        from datamaite import convert

        with pytest.raises(ValueError, match="not supported"):
            convert(tmp_path, dest, input_format="flat_images", output_format="flat_images", mode="replace")

        assert (dest / "keep.txt").read_text(encoding="utf-8") == "stay"

    def test_mixed_folder_fails_whole(self, tmp_path: Path) -> None:
        (tmp_path / "a.jpg").write_bytes(_JPEG_HEAD)
        _write_safetensors(tmp_path / "b.safetensors", {"image": np.zeros((2, 2, 3), dtype=np.uint8)})
        ds = load_flat_images(tmp_path)
        dest = tmp_path / "out"

        from datamaite import write

        with pytest.raises(ValueError, match="First offending sample"):
            write(ds, dest, output_format="flat_images")
        assert not dest.exists() or not any(dest.iterdir())

    def test_memory_root_write_rejects(self, memory_root) -> None:  # type: ignore[no-untyped-def]
        root = memory_root / "st-write"
        root.mkdir(parents=True)
        (root / "a.safetensors").write_bytes(_safetensors_bytes({"image": np.zeros((2, 2, 3), dtype=np.uint8)}))
        ds = load_flat_images(root)
        dest = memory_root / "st-write-out"

        from datamaite import write

        with pytest.raises(ValueError, match="not supported"):
            write(ds, dest, output_format="flat_images")

    def test_encoded_flat_images_still_writes(self, tmp_path: Path) -> None:
        cv2 = pytest.importorskip("cv2")
        ok, buf = cv2.imencode(".png", np.zeros((4, 6, 3), dtype=np.uint8))
        assert ok
        (tmp_path / "a.png").write_bytes(buf.tobytes())
        ds = load_flat_images(tmp_path)
        dest = tmp_path / "out"

        from datamaite import write

        write(ds, dest, output_format="flat_images")
        assert (dest / "a.png").is_file()
