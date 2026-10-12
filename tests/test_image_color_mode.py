"""MAITE image color mode for the IC/OD view: rgb (default) / gray / native (#84)."""

from __future__ import annotations

import io
import pickle
from pathlib import Path

import numpy as np
import pytest

from datamaite.image_classification import ImageClassificationDataset
from datamaite.object_detection import ObjectDetectionDataset
from datamaite.records import ImageClassificationSample, ImageObjectDetectionSample, ImageRecord

cv2 = pytest.importorskip("cv2")

W, H = 8, 6


def _gray_png(path: Path) -> Path:
    """Single-channel PNG whose value encodes the column."""
    cv2.imwrite(str(path), np.tile(np.arange(W, dtype=np.uint8)[None, :] * 10, (H, 1)))
    return path


def _bgr_png(path: Path) -> Path:
    """3-channel PNG with constant B/G/R planes 10/20/30."""
    bgr = np.zeros((H, W, 3), dtype=np.uint8)
    bgr[:, :] = (10, 20, 30)
    cv2.imwrite(str(path), bgr)
    return path


def _bgra_png(path: Path) -> Path:
    """4-channel PNG: B/G/R/A = 10/20/30/40."""
    bgra = np.zeros((H, W, 4), dtype=np.uint8)
    bgra[:, :] = (10, 20, 30, 40)
    cv2.imwrite(str(path), bgra)
    return path


def _decode(record: ImageRecord, **kwargs: object) -> np.ndarray:
    from datamaite.maite._image import decode_image

    return decode_image(record, **kwargs)  # type: ignore[arg-type]


def _record(path: Path, **kwargs: object) -> ImageRecord:
    return ImageRecord(image_id="x", path_or_uri=str(path), **kwargs)  # type: ignore[arg-type]


class TestDecodeEncoded:
    def test_default_rgb_replicates_grayscale(self, tmp_path: Path) -> None:
        arr = _decode(_record(_gray_png(tmp_path / "g.png")))
        assert arr.shape == (3, H, W)
        assert arr.dtype == np.uint8
        assert np.array_equal(arr[0], arr[1])
        assert np.array_equal(arr[1], arr[2])

    def test_default_rgb_drops_alpha_and_orders_rgb(self, tmp_path: Path) -> None:
        arr = _decode(_record(_bgra_png(tmp_path / "a.png")))
        assert arr.shape == (3, H, W)
        assert [int(arr[c, 0, 0]) for c in range(3)] == [30, 20, 10]

    def test_gray_keeps_single_channel_values(self, tmp_path: Path) -> None:
        arr = _decode(_record(_gray_png(tmp_path / "g.png")), color_mode="gray")
        assert arr.shape == (1, H, W)
        assert arr[0].tolist() == [[x * 10 for x in range(W)]] * H

    def test_gray_from_color_matches_opencv_luma(self, tmp_path: Path) -> None:
        path = _bgr_png(tmp_path / "c.png")
        arr = _decode(_record(path), color_mode="gray")
        assert arr.shape == (1, H, W)
        assert np.array_equal(arr[0], cv2.imread(str(path), cv2.IMREAD_GRAYSCALE))

    def test_native_grayscale_stays_one_channel(self, tmp_path: Path) -> None:
        arr = _decode(_record(_gray_png(tmp_path / "g.png")), color_mode="native")
        assert arr.shape == (1, H, W)

    def test_native_color_is_rgb_ordered(self, tmp_path: Path) -> None:
        arr = _decode(_record(_bgr_png(tmp_path / "c.png")), color_mode="native")
        assert arr.shape == (3, H, W)
        assert [int(arr[c, 0, 0]) for c in range(3)] == [30, 20, 10]

    def test_native_keeps_alpha_as_rgba(self, tmp_path: Path) -> None:
        arr = _decode(_record(_bgra_png(tmp_path / "a.png")), color_mode="native")
        assert arr.shape == (4, H, W)
        assert [int(arr[c, 0, 0]) for c in range(4)] == [30, 20, 10, 40]

    def test_native_keeps_16_bit_depth(self, tmp_path: Path) -> None:
        path = tmp_path / "deep.png"
        cv2.imwrite(str(path), np.full((H, W), 60000, dtype=np.uint16))
        arr = _decode(_record(path), color_mode="native")
        assert arr.shape == (1, H, W)
        assert arr.dtype == np.uint16
        assert int(arr.max()) == 60000
        # rgb/gray stay 8-bit views of the same file.
        assert _decode(_record(path)).dtype == np.uint8
        assert _decode(_record(path), color_mode="gray").dtype == np.uint8

    @pytest.mark.parametrize(("mode", "channels"), [("rgb", 3), ("gray", 1), ("native", 4)])
    def test_byte_backed_sample(self, tmp_path: Path, mode: str, channels: int) -> None:
        payload = _bgra_png(tmp_path / "a.png").read_bytes()
        arr = _decode(ImageRecord(image_id="b", image_bytes=payload), color_mode=mode)
        assert arr.shape == (channels, H, W)

    @pytest.mark.parametrize(("mode", "channels"), [("gray", 1), ("native", 4)])
    def test_region_crop_applies_in_every_mode(self, tmp_path: Path, mode: str, channels: int) -> None:
        sample = ImageClassificationSample(
            image_id="crop", path_or_uri=str(_bgra_png(tmp_path / "a.png")), region=(2.0, 1.0, 3.0, 2.0)
        )
        assert _decode(sample, color_mode=mode).shape == (channels, 2, 3)

    def test_unknown_mode_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="color_mode"):
            _decode(_record(_gray_png(tmp_path / "g.png")), color_mode="unchanged")

    def test_shared_cache_does_not_mix_modes(self, tmp_path: Path) -> None:
        record = _record(_bgra_png(tmp_path / "a.png"))
        cache: dict[tuple[str, str], np.ndarray] = {}
        assert _decode(record, base_cache=cache).shape == (3, H, W)
        assert _decode(record, base_cache=cache, color_mode="gray").shape == (1, H, W)
        assert _decode(record, base_cache=cache, color_mode="native").shape == (4, H, W)
        assert _decode(record, base_cache=cache).shape == (3, H, W)


class TestDecodeSafeTensors:
    @pytest.fixture
    def st_numpy(self):  # type: ignore[no-untyped-def]
        return pytest.importorskip("safetensors.numpy")

    def _record(self, st_numpy, tmp_path: Path, tensor: np.ndarray) -> ImageRecord:  # type: ignore[no-untyped-def]
        path = tmp_path / "t.safetensors"
        st_numpy.save_file({"image": tensor}, str(path))
        return ImageRecord(image_id="st", path_or_uri=str(path), metadata={"safetensors_key": "image"})

    def test_default_rgb_replicates_grayscale(self, st_numpy, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        record = self._record(st_numpy, tmp_path, np.full((H, W), 7, dtype=np.uint8))
        arr = _decode(record)
        assert arr.shape == (3, H, W)
        assert int(arr.min()) == int(arr.max()) == 7

    def test_native_keeps_rgba(self, st_numpy, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        rgba[:, :] = (30, 20, 10, 40)
        arr = _decode(self._record(st_numpy, tmp_path, rgba), color_mode="native")
        assert arr.shape == (4, H, W)
        assert [int(arr[c, 0, 0]) for c in range(4)] == [30, 20, 10, 40]

    def test_native_grayscale_stays_one_channel(self, st_numpy, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        record = self._record(st_numpy, tmp_path, np.full((H, W), 7, dtype=np.uint8))
        assert _decode(record, color_mode="native").shape == (1, H, W)

    def test_gray_from_rgb_uses_opencv_luma(self, st_numpy, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        rgb[:, :] = (30, 20, 10)
        arr = _decode(self._record(st_numpy, tmp_path, rgb), color_mode="gray")
        expected = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        assert arr.shape == (1, H, W)
        assert np.array_equal(arr[0], expected)


class TestDatasetSurface:
    def _ic(self, path: Path) -> ImageClassificationDataset:
        return ImageClassificationDataset(samples=(ImageClassificationSample(image_id=0, path_or_uri=str(path)),))

    def _od(self, path: Path) -> ObjectDetectionDataset:
        return ObjectDetectionDataset(samples=(ImageObjectDetectionSample(image_id=0, path_or_uri=str(path)),))

    @pytest.mark.parametrize("kind", ["ic", "od"])
    def test_default_is_rgb(self, tmp_path: Path, kind: str) -> None:
        path = _gray_png(tmp_path / "g.png")
        ds = self._ic(path) if kind == "ic" else self._od(path)
        assert ds.image_color_mode == "rgb"
        assert ds[0][0].shape == (3, H, W)

    @pytest.mark.parametrize("kind", ["ic", "od"])
    def test_with_image_options_applies_to_every_accessor(self, tmp_path: Path, kind: str) -> None:
        path = _bgra_png(tmp_path / "a.png")
        base = self._ic(path) if kind == "ic" else self._od(path)
        ds = base.with_image_options(color_mode="native")
        assert ds.image_color_mode == "native"
        assert base.image_color_mode == "rgb"  # a copy, not a mutation
        assert ds[0][0].shape == (4, H, W)
        assert ds.get_input(0).shape == (4, H, W)
        meta = ds.get_metadata(0)
        assert (meta["height"], meta["width"]) == (H, W)

    @pytest.mark.parametrize("kind", ["ic", "od"])
    def test_omitted_option_keeps_current_value(self, tmp_path: Path, kind: str) -> None:
        path = _gray_png(tmp_path / "g.png")
        ds = (self._ic(path) if kind == "ic" else self._od(path)).with_image_options(color_mode="gray")
        assert ds.with_image_options().image_color_mode == "gray"

    @pytest.mark.parametrize("kind", ["ic", "od"])
    def test_with_image_options_keeps_storage_options(self, tmp_path: Path, kind: str) -> None:
        path = _gray_png(tmp_path / "g.png")
        base = self._ic(path) if kind == "ic" else self._od(path)
        ds = base.with_storage_options({"anon": True}).with_image_options(color_mode="gray")
        assert ds._runtime_storage_options == {"anon": True}

    @pytest.mark.parametrize("kind", ["ic", "od"])
    def test_invalid_mode_rejected(self, tmp_path: Path, kind: str) -> None:
        path = _gray_png(tmp_path / "g.png")
        base = self._ic(path) if kind == "ic" else self._od(path)
        with pytest.raises(ValueError, match="color_mode"):
            base.with_image_options(color_mode="bgr")
        cls = type(base)
        with pytest.raises(ValueError, match="image_color_mode"):
            cls(samples=base.samples, image_color_mode="bgr")

    def test_ic_region_crop_metadata_follows_decoded_crop(self, tmp_path: Path) -> None:
        sample = ImageClassificationSample(
            image_id=0, path_or_uri=str(_bgra_png(tmp_path / "a.png")), region=(6.0, 0.0, 4.0, 2.0)
        )
        ds = ImageClassificationDataset(samples=(sample,)).with_image_options(color_mode="gray")
        image, _, meta = ds[0]
        assert image.shape == (1, 2, 2)  # clamped at the right edge
        assert (meta["height"], meta["width"]) == (2, 2)

    @pytest.mark.parametrize("kind", ["ic", "od"])
    def test_mode_survives_pickle(self, tmp_path: Path, kind: str) -> None:
        path = _gray_png(tmp_path / "g.png")
        ds = (self._ic(path) if kind == "ic" else self._od(path)).with_image_options(color_mode="gray")
        restored = pickle.loads(pickle.dumps(ds))  # noqa: S301 - round-tripping our own object
        assert restored.image_color_mode == "gray"
        assert restored[0][0].shape == (1, H, W)


MODES = ("rgb", "gray", "native")


class TestReturnedArraysAreOwned:
    """Every mode must hand back a fresh, writable array (review of !74).

    A 1-channel CHW transpose is already C-contiguous, so a bare
    ``ascontiguousarray`` returned a view: of the IC decode cache (an in-place
    edit corrupted later reads) or of a read-only SafeTensors buffer.
    """

    @staticmethod
    def _assert_owned(arr: np.ndarray) -> None:
        assert arr.flags.writeable
        assert arr.flags.c_contiguous
        assert arr.base is None

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("writer", [_gray_png, _bgr_png, _bgra_png], ids=["gray", "bgr", "bgra"])
    def test_ic_edit_does_not_reach_the_cache(self, tmp_path: Path, mode: str, writer) -> None:  # type: ignore[no-untyped-def]
        path = writer(tmp_path / "img.png")
        ds = ImageClassificationDataset(
            samples=(ImageClassificationSample(image_id=0, path_or_uri=str(path)),)
        ).with_image_options(color_mode=mode)
        before = ds.get_input(0).copy()
        first = ds.get_input(0)
        self._assert_owned(first)
        first[...] = 255 - first
        assert np.array_equal(ds.get_input(0), before)
        assert np.array_equal(ds[0][0], before)

    @pytest.mark.parametrize("mode", MODES)
    def test_full_frame_region_crop_is_owned(self, tmp_path: Path, mode: str) -> None:
        sample = ImageClassificationSample(
            image_id=0, path_or_uri=str(_gray_png(tmp_path / "g.png")), region=(0.0, 0.0, float(W), float(H))
        )
        ds = ImageClassificationDataset(samples=(sample,)).with_image_options(color_mode=mode)
        before = ds.get_input(0).copy()
        first = ds.get_input(0)
        self._assert_owned(first)
        first += 1
        assert np.array_equal(ds.get_input(0), before)

    @pytest.mark.parametrize("mode", MODES)
    def test_od_and_byte_backed_are_owned(self, tmp_path: Path, mode: str) -> None:
        path = _gray_png(tmp_path / "g.png")
        od = ObjectDetectionDataset(
            samples=(ImageObjectDetectionSample(image_id=0, path_or_uri=str(path)),)
        ).with_image_options(color_mode=mode)
        self._assert_owned(od.get_input(0))
        self._assert_owned(_decode(ImageRecord(image_id="b", image_bytes=path.read_bytes()), color_mode=mode))

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize(
        "tensor",
        [
            np.full((H, W), 7, dtype=np.uint8),
            np.full((H, W, 1), 7, dtype=np.uint8),
            np.full((1, H, W), 7, dtype=np.uint8),
            np.full((H, W, 3), 7, dtype=np.uint8),
            np.full((H, W, 4), 7, dtype=np.uint8),
        ],
        ids=["hw", "hwc1", "chw1", "hwc3", "hwc4"],
    )
    def test_safetensors_outputs_are_writable(self, tmp_path: Path, mode: str, tensor: np.ndarray) -> None:
        st_numpy = pytest.importorskip("safetensors.numpy")
        path = tmp_path / "t.safetensors"
        st_numpy.save_file({"image": tensor}, str(path))
        arr = _decode(
            ImageRecord(image_id="st", path_or_uri=str(path), metadata={"safetensors_key": "image"}), color_mode=mode
        )
        self._assert_owned(arr)
        arr[...] = 0  # must not raise "assignment destination is read-only"


class TestNativeTiff:
    """Multispectral and SAR TIFFs (#84) decode through tifffile in ``native``."""

    @pytest.fixture(autouse=True)
    def _tifffile(self) -> None:
        pytest.importorskip("tifffile")

    @pytest.mark.parametrize(
        ("name", "bands", "dtype"),
        [("synthetic_s2_13band_uint16.tif", 13, np.uint16), ("synthetic_s1_2band_float32.tif", 2, np.float32)],
        ids=["sentinel2", "sentinel1-sar"],
    )
    def test_rasters_keep_every_band_and_dtype(
        self, synthetic_rasters: Path, name: str, bands: int, dtype: type
    ) -> None:
        import tifffile

        path = synthetic_rasters / name
        arr = _decode(_record(path), color_mode="native")
        expected = np.transpose(tifffile.imread(path), (2, 0, 1))
        assert arr.shape == (bands, 256, 256)
        assert arr.dtype == dtype
        np.testing.assert_array_equal(arr, expected)
        assert arr.flags.writeable
        assert arr.base is None

    @pytest.mark.parametrize("mode", ["rgb", "gray"])
    def test_rgb_and_gray_point_to_native(self, synthetic_rasters: Path, mode: str) -> None:
        path = synthetic_rasters / "synthetic_s2_13band_uint16.tif"
        with pytest.raises(OSError, match="color_mode='native'"):
            _decode(_record(path), color_mode=mode)

    def test_byte_backed_region_crop(self, synthetic_rasters: Path) -> None:
        import tifffile

        payload = (synthetic_rasters / "synthetic_s1_2band_float32.tif").read_bytes()
        sample = ImageClassificationSample(image_id="sar", image_bytes=payload, region=(250.0, 10.0, 20.0, 5.0))
        arr = _decode(sample, color_mode="native")
        expected = np.transpose(tifffile.imread(io.BytesIO(payload))[10:15, 250:256], (2, 0, 1))
        assert arr.shape == (2, 5, 6)  # clamped at the right edge
        np.testing.assert_array_equal(arr, expected)

    def test_memory_backed_raster(self, synthetic_rasters: Path, memory_root) -> None:  # type: ignore[no-untyped-def]
        source = synthetic_rasters / "synthetic_s2_13band_uint16.tif"
        remote = memory_root / "s2.tif"
        remote.write_bytes(source.read_bytes())
        local = _decode(_record(source), color_mode="native")
        assert np.array_equal(_decode(ImageRecord(image_id="r", path_or_uri=str(remote)), color_mode="native"), local)

    @pytest.mark.parametrize(
        ("shape", "dtype", "options"),
        [
            ((H, W), np.uint16, {"photometric": "minisblack"}),
            ((H, W), np.float32, {"photometric": "minisblack"}),
            ((H, W), np.bool_, {"photometric": "minisblack"}),
            ((H, W), np.int8, {"photometric": "minisblack"}),
            ((H, W), np.int32, {"photometric": "minisblack"}),
            ((H, W), np.uint32, {"photometric": "minisblack"}),
            ((H, W), np.int64, {"photometric": "minisblack"}),
            ((H, W), np.uint64, {"photometric": "minisblack"}),
            ((H, W), np.float16, {"photometric": "minisblack"}),
            ((H, W, 3), np.uint8, {"photometric": "rgb"}),
            ((H, W, 3), np.uint16, {"photometric": "rgb"}),
            ((H, W, 2), np.uint8, {"photometric": "minisblack", "planarconfig": "contig"}),
            ((H, W, 2), np.float32, {"photometric": "minisblack", "planarconfig": "contig"}),
            ((H, W, 3), np.int16, {"photometric": "minisblack", "planarconfig": "contig"}),
            ((H, W, 3), np.float32, {"photometric": "minisblack", "planarconfig": "contig"}),
            ((H, W, 4), np.uint8, {"photometric": "rgb", "extrasamples": ["unassalpha"]}),
            ((H, W, 6), np.int16, {"photometric": "minisblack", "planarconfig": "contig"}),
            ((3, H, W), np.uint8, {"photometric": "rgb", "planarconfig": "separate"}),
            ((5, H, W), np.uint16, {"photometric": "minisblack", "planarconfig": "separate"}),
            ((13, H, W), np.uint16, {"photometric": "minisblack", "planarconfig": "separate", "compression": "zlib"}),
        ],
        ids=[
            "1band-uint16",
            "1band-float32",
            "1band-bool",
            "1band-int8",
            "1band-int32",
            "1band-uint32",
            "1band-int64",
            "1band-uint64",
            "1band-float16",
            "rgb-uint8",
            "rgb-uint16",
            "2band-uint8",
            "2band-float32",
            "3band-int16",
            "3band-float32",
            "rgba-unassociated-alpha",
            "6band-int16",
            "rgb-planar",
            "5band-planar",
            "13band-planar-deflate",
        ],
    )
    def test_tiff_keeps_stored_bands(self, tmp_path: Path, shape, dtype, options) -> None:  # type: ignore[no-untyped-def]
        """Every layout decodes as stored, including those OpenCV silently truncates or alters."""
        import tifffile

        array = (np.arange(int(np.prod(shape))) % 200).astype(dtype).reshape(shape)
        path = tmp_path / "raster.tiff"
        tifffile.imwrite(path, array, **options)
        arr = _decode(_record(path), color_mode="native")
        if array.ndim == 2:
            expected = array[None]
        else:
            expected = array if options.get("planarconfig") == "separate" else np.transpose(array, (2, 0, 1))
        assert arr.dtype == array.dtype
        np.testing.assert_array_equal(arr, expected)

    @pytest.mark.parametrize(
        ("options", "axes"),
        [
            ({"metadata": {"axes": "CYX"}}, "CYX"),
            ({"metadata": {"axes": "CYX"}, "ome": True}, "CYX"),
            ({"metadata": {"axes": "CYX"}, "imagej": True}, "CYX"),
            ({}, "QYX"),
            ({"metadata": None}, "IYX"),
        ],
        ids=["channels", "ome-channels", "imagej-channels", "shaped-unlabelled", "plain-pages"],
    )
    def test_multi_page_stacks_keep_channels_or_raise(self, tmp_path: Path, options, axes: str) -> None:  # type: ignore[no-untyped-def]
        """OpenCV reads only page 1, so a page stack is read whole as channels or rejected, never truncated."""
        import tifffile

        array = np.arange(13 * H * W, dtype=np.uint16).reshape(13, H, W)
        path = tmp_path / "stack.tif"
        tifffile.imwrite(path, array, photometric="minisblack", **options)
        assert tifffile.TiffFile(path).series[0].axes == axes
        if axes == "CYX":
            np.testing.assert_array_equal(_decode(_record(path), color_mode="native"), array)
        else:
            with pytest.raises(OSError, match=f"unsupported TIFF layout '{axes}'"):
                _decode(_record(path), color_mode="native")

    def test_byte_backed_tiffs_sharing_a_path_are_not_cached(self) -> None:
        import tifffile

        def payload(value: int) -> bytes:
            buffer = io.BytesIO()
            tifffile.imwrite(
                buffer, np.full((H, W, 2), value, dtype=np.uint8), photometric="minisblack", planarconfig="contig"
            )
            return buffer.getvalue()

        ds = ImageClassificationDataset(
            samples=tuple(
                ImageClassificationSample(image_id=i, path_or_uri="shared.tif", image_bytes=payload(value))
                for i, value in enumerate((10, 20))
            )
        ).with_image_options(color_mode="native")
        assert [int(ds.get_input(i)[0, 0, 0]) for i in range(2)] == [10, 20]

    def test_decode_error_suggests_native_only_outside_native(self) -> None:
        from datamaite.maite._image import _decode_error

        sample = _record(Path("raster.tif"))
        assert "color_mode='native'" in str(_decode_error(sample, "rgb"))
        assert "color_mode='native'" not in str(_decode_error(sample, "native"))

    def test_lzw_rgb_tiff_decodes_without_imagecodecs(self, tmp_path: Path) -> None:
        """OpenCV writes LZW TIFFs; plain RGB ones decode through OpenCV, not tifffile."""
        path = tmp_path / "lzw.tif"
        bgr = np.zeros((H, W, 3), dtype=np.uint8)
        bgr[:, :] = (10, 20, 30)
        cv2.imwrite(str(path), bgr)
        np.testing.assert_array_equal(_decode(_record(path), color_mode="native")[:, 0, 0], [30, 20, 10])

    def test_palette_tiff_decodes_to_colors(self, tmp_path: Path) -> None:
        import tifffile

        colormap = np.zeros((3, 256), dtype=np.uint16)
        colormap[0] = np.arange(256) * 257
        path = tmp_path / "palette.tif"
        tifffile.imwrite(path, np.full((H, W), 200, dtype=np.uint8), photometric="palette", colormap=colormap)
        np.testing.assert_array_equal(_decode(_record(path), color_mode="native")[:, 0, 0], [200, 0, 0])

    def test_decoded_size_is_capped(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A small compressed TIFF cannot expand past the decoded byte cap."""
        import tifffile

        path = tmp_path / "big.tif"
        tifffile.imwrite(path, np.zeros((2, H, W), dtype=np.float64), planarconfig="separate", compression="zlib")
        monkeypatch.setattr("datamaite.maite._image.MAX_ENCODED_IMAGE_BYTES", 2 * H * W * 8 - 1)
        with pytest.raises(OSError, match="byte cap"):
            _decode(_record(path), color_mode="native")

    def test_ic_cache_keeps_native_tiff_isolated(self, synthetic_rasters: Path) -> None:
        path = synthetic_rasters / "synthetic_s2_13band_uint16.tif"
        ds = ImageClassificationDataset(
            samples=(ImageClassificationSample(image_id=0, path_or_uri=str(path)),)
        ).with_image_options(color_mode="native")
        before = ds.get_input(0).copy()
        first = ds.get_input(0)
        first[...] = 0
        assert np.array_equal(ds.get_input(0), before)
        assert ds.get_metadata(0)["height"] == 256
