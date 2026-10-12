"""Shared MAITE image decoding helpers for still-image tasks."""

from __future__ import annotations

import io
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import PurePath
from typing import IO, Any

import numpy as np

from datamaite._io import MAX_DECODED_IMAGE_PIXELS, decode_image_bytes, decode_image_path
from datamaite._safetensors import MAX_ENCODED_IMAGE_BYTES, _to_uint8, load_image_hwc
from datamaite._upath import sanitized_uri, to_dataset_path
from datamaite.records import ImageRecord

# Channel layouts the MAITE IC/OD view can emit (#84). MAITE fixes only the
# ``(C, H, W)`` axis order, not the channel count, so ``(1, H, W)`` grayscale
# and ``(4, H, W)`` RGBA are as protocol-conformant as ``(3, H, W)`` RGB.
#
# * ``"rgb"`` (default) -- 3-channel RGB: grayscale is replicated, alpha dropped.
# * ``"gray"`` -- one luma channel, ``(1, H, W)``.
# * ``"native"`` -- the stored bands and dtype, in stored (RGB(A)) order. TIFFs
#   with more than one band, other than plain RGB, decode with ``tifffile``, so
#   any band count works (multispectral, SAR); everything else uses OpenCV's
#   1/3/4 channels.
#   Not rescaled to ``uint8``: 16-bit stays ``uint16``, float stays float.
COLOR_MODES = ("rgb", "gray", "native")

_TIFF_SUFFIXES = frozenset({".tif", ".tiff"})
_TIFF_MAGIC = (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")  # classic and BigTIFF

# ITU-R BT.601 luma weights, the ones OpenCV's IMREAD_GRAYSCALE uses, so a
# SafeTensors sample reads the same gray as an encoded one.
_LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def validate_color_mode(color_mode: str) -> str:
    """Return ``color_mode`` if it names a supported layout, else raise ``ValueError``."""
    if color_mode not in COLOR_MODES:
        raise ValueError(f"color_mode must be one of {COLOR_MODES}, got {color_mode!r}")
    return color_mode


def decode_image(
    sample: ImageRecord,
    *,
    task_name: str = "image",
    extra: str = "all",
    storage_options: Mapping[str, Any] | None = None,
    base_cache: dict[tuple[str, str], np.ndarray] | None = None,
    color_mode: str = "rgb",
) -> np.ndarray:
    """Decode an image sample to a ``(C, H, W)`` array.

    ``color_mode`` selects the channel layout (see :data:`COLOR_MODES`). The
    default ``"rgb"`` and ``"gray"`` are ``uint8``; ``"native"`` keeps the
    stored dtype.

    If ``sample.region`` is set (a ``(left, top, width, height)`` crop rectangle,
    used by loaders that derive classification crops from detection boxes, e.g.
    VisDrone IC), the decoded image is cropped to that region after clamping it to
    the image bounds. A region that is empty after clamping (fully outside the
    image, or sub-pixel) raises ``ValueError``.

    Encoded images decode through OpenCV, which is imported lazily so
    importing/loading datasets does not require the optional task extras; only
    MAITE-style indexing does. SafeTensors sources (flat-images loader) decode
    with numpy alone via :mod:`datamaite._safetensors`, so the OpenCV check is
    branched behind the source kind rather than run unconditionally.
    """
    validate_color_mode(color_mode)
    if isinstance(sample.metadata.get("safetensors_key"), str):
        hwc = _decode_safetensors(sample, storage_options, color_mode)
    else:
        hwc = _decode_encoded(
            sample,
            task_name=task_name,
            extra=extra,
            storage_options=storage_options,
            base_cache=base_cache,
            color_mode=color_mode,
        )
    region = getattr(sample, "region", None)
    if region is not None:
        img_h, img_w = hwc.shape[:2]
        left, top, box_w, box_h = region
        x1 = max(0, round(left))
        y1 = max(0, round(top))
        x2 = min(img_w, round(left + box_w))
        y2 = min(img_h, round(top + box_h))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"crop region {region} for sample {sample.image_id!r} is empty after clamping to image bounds"
            )
        hwc = hwc[y1:y2, x1:x2]
    # HWC -> CHW as a fresh, writable array the caller owns. ascontiguousarray
    # would return a view whenever the transpose is already C-contiguous (any
    # 1-channel image), aliasing the IC decode cache or a read-only
    # SafeTensors buffer, so an in-place edit would corrupt later reads.
    return np.array(np.transpose(hwc, (2, 0, 1)), order="C", copy=True)


def _decode_safetensors(sample: ImageRecord, storage_options: Mapping[str, Any] | None, color_mode: str) -> np.ndarray:
    """Decode a safetensors-backed sample to an ``(H, W, C)`` array in ``color_mode``."""
    path = str(sample.path_or_uri)
    key = sample.metadata["safetensors_key"]
    try:
        hwc = load_image_hwc(path, key, storage_options, native_channels=color_mode != "rgb")
    except (OSError, ValueError) as exc:
        raise OSError(f"could not decode image for sample {sample.image_id!r} ({sanitized_uri(path)}): {exc}") from exc
    if color_mode == "gray":
        hwc = _to_uint8(hwc)
        if hwc.shape[2] != 1:
            luma = hwc[:, :, :3].astype(np.float32) @ _LUMA_WEIGHTS
            hwc = np.clip(np.rint(luma), 0, 255).astype(np.uint8)[:, :, None]
    return hwc


def _decode_encoded(
    sample: ImageRecord,
    *,
    task_name: str,
    extra: str,
    storage_options: Mapping[str, Any] | None,
    base_cache: dict[tuple[str, str], np.ndarray] | None,
    color_mode: str,
) -> np.ndarray:
    """Decode an encoded (JPEG/PNG/TIFF/...) sample to an ``(H, W, C)`` array in ``color_mode``."""
    _require_cv2(task_name, extra)
    if color_mode == "native" and _is_tiff(sample):
        multiband = _decode_multiband_tiff(sample, storage_options=storage_options, base_cache=base_cache)
        if multiband is not None:
            return multiband

    # "native" reads IMREAD_UNCHANGED (neither flag); OpenCV skips EXIF
    # orientation in that mode, so native keeps the stored pixel grid (#76).
    flags = {"color": color_mode == "rgb", "gray": color_mode == "gray"}
    if sample.image_bytes is not None:
        decoded = decode_image_bytes(sample.image_bytes, **flags)
    elif sample.path_or_uri is not None:
        # Keyed by mode too: the cache holds the raw decode, which differs per mode.
        cache_key = (sample.path_or_uri, color_mode)
        if base_cache is not None and cache_key in base_cache:
            decoded = base_cache[cache_key]
        else:
            decoded = decode_image_path(sample.path_or_uri, storage_options, **flags)
            if decoded is not None and base_cache is not None:
                base_cache.clear()
                base_cache[cache_key] = decoded
    else:
        raise ValueError(f"image sample {sample.image_id!r} has neither path_or_uri nor image_bytes")
    if decoded is None:
        raise _decode_error(sample, color_mode)
    if color_mode == "gray":
        return decoded[:, :, None]
    if color_mode == "native":
        return _native_to_rgb_order(decoded)
    # The cache holds BGR (OpenCV's order); flip to RGB per caller, above the
    # shared region crop. Both are views, so this allocates nothing.
    return decoded[:, :, ::-1]


def _require_cv2(task_name: str, extra: str) -> None:
    try:
        __import__("cv2")
    except ImportError as exc:
        raise ImportError(
            f"Indexing a {task_name} dataset as a MAITE dataset decodes images and needs OpenCV. "
            f"Install it with: pip install datamaite[{extra}]"
        ) from exc


def _native_to_rgb_order(decoded: np.ndarray) -> np.ndarray:
    """An IMREAD_UNCHANGED array as ``(H, W, C)`` in RGB(A) channel order, stored dtype kept."""
    hwc = decoded[:, :, None] if decoded.ndim == 2 else decoded
    if hwc.shape[2] == 3:
        return hwc[:, :, ::-1]  # BGR -> RGB
    if hwc.shape[2] == 4:
        return hwc[:, :, [2, 1, 0, 3]]  # BGRA -> RGBA
    return hwc


def _is_tiff(sample: ImageRecord) -> bool:
    if sample.image_bytes is not None:
        return sample.image_bytes[:4] in _TIFF_MAGIC
    return sample.path_or_uri is not None and PurePath(str(sample.path_or_uri)).suffix.lower() in _TIFF_SUFFIXES


def _decode_error(sample: ImageRecord, color_mode: str) -> OSError:
    """The OpenCV decode failure for ``sample``, pointing ``rgb``/``gray`` TIFFs at ``native``."""
    location = sanitized_uri(sample.path_or_uri) if sample.path_or_uri is not None else "embedded bytes"
    hint = ""
    if color_mode != "native" and _is_tiff(sample):
        hint = (
            "; OpenCV reads only 1/3/4-channel integer images, so for multispectral, SAR, or "
            "float TIFFs use with_image_options(color_mode='native')"
        )
    return OSError(f"could not decode image for sample {sample.image_id!r} ({location}){hint}")


# Base-cache tag for tifffile decodes. They are stored in final band order,
# unlike the raw OpenCV decode cached under ``(path, "native")``.
_TIFFFILE_CACHE_TAG = "native-tifffile"

# Single-band TIFF dtypes OpenCV decodes as stored. It rejects uint32, 64-bit
# integers, and float16, and widens 1-bit samples to uint8 {0, 255}.
_OPENCV_SINGLE_BAND_DTYPES = frozenset({"int8", "uint8", "int16", "uint16", "int32", "float32", "float64"})


def _decode_multiband_tiff(
    sample: ImageRecord,
    *,
    storage_options: Mapping[str, Any] | None,
    base_cache: dict[tuple[str, str], np.ndarray] | None,
) -> np.ndarray | None:
    """Decode a TIFF that OpenCV cannot read as stored with ``tifffile``.

    OpenCV is used only where it reads the stored values exactly (see
    :func:`_opencv_reads_as_stored`), so LZW/JPEG and palette TIFFs still decode
    without ``imagecodecs``. Elsewhere it rejects or silently alters the data:
    it keeps only the first band of 2-band, non-RGB, and band-separate
    (planar) rasters, premultiplies unassociated alpha, widens 1-bit samples,
    and rejects multi-band float, uint32, 64-bit, and float16 samples. So
    multispectral (e.g. 13-band Sentinel-2), SAR (e.g. 2-band float32
    Sentinel-1), RGBA, and unusual dtypes decode with ``tifffile``. Returns
    ``(H, W, bands)`` in the stored dtype, or ``None`` when OpenCV should
    decode the file.
    """
    # Like the OpenCV path, byte-backed samples bypass the cache: their bytes,
    # not the path they share with other samples, are the pixel source.
    path = sample.path_or_uri if sample.image_bytes is None else None
    if base_cache is not None and path is not None:
        if (path, _TIFFFILE_CACHE_TAG) in base_cache:
            return base_cache[(path, _TIFFFILE_CACHE_TAG)]
        if (path, "native") in base_cache:
            return None
    try:
        import tifffile
    except ImportError as exc:
        raise ImportError(
            "Decoding TIFFs with color_mode='native' needs tifffile. "
            "Install it with: pip install datamaite[od] (or datamaite[ic])"
        ) from exc

    location = sanitized_uri(sample.path_or_uri) if sample.path_or_uri is not None else "embedded bytes"
    try:
        with _open_encoded(sample, storage_options) as stream, tifffile.TiffFile(stream) as tif:
            # Route on the whole series: OpenCV reads only the first page, so a
            # multi-page channel stack must not reach it.
            series = tif.series[0]
            if series.axes in ("YX", "YXS") and _opencv_reads_as_stored(tif.pages[0], tifffile):
                return None
            hwc = _tiff_series_to_hwc(series)
    except (OSError, ValueError, IndexError, tifffile.TiffFileError) as exc:
        raise OSError(f"could not decode image for sample {sample.image_id!r} ({location}): {exc}") from exc
    if base_cache is not None and path is not None:
        base_cache.clear()
        base_cache[(path, _TIFFFILE_CACHE_TAG)] = hwc
    return hwc


def _opencv_reads_as_stored(page: Any, tifffile: Any) -> bool:
    """True when OpenCV decodes this TIFF page exactly as stored (checked on OpenCV 4.13).

    That is palette images (expanded to their colors), single-band images in
    :data:`_OPENCV_SINGLE_BAND_DTYPES`, and interleaved 8/16-bit RGB or YCbCr.
    """
    if page.photometric == tifffile.PHOTOMETRIC.PALETTE:
        return True
    if page.dtype is None:  # tifffile cannot decode it either; let OpenCV try
        return True
    dtype = np.dtype(page.dtype)
    if page.bitspersample != dtype.itemsize * 8:  # 1-bit, 12-bit, ...
        return False
    if page.samplesperpixel == 1:
        return dtype.name in _OPENCV_SINGLE_BAND_DTYPES
    return (
        page.samplesperpixel == 3
        and page.photometric in (tifffile.PHOTOMETRIC.RGB, tifffile.PHOTOMETRIC.YCBCR)
        and page.planarconfig == tifffile.PLANARCONFIG.CONTIG
        and dtype.name in ("uint8", "uint16")
    )


@contextmanager
def _open_encoded(sample: ImageRecord, storage_options: Mapping[str, Any] | None) -> Iterator[IO[bytes]]:
    """A seekable binary stream over the sample's encoded bytes, local or remote."""
    if sample.image_bytes is not None:
        yield io.BytesIO(sample.image_bytes)
    elif sample.path_or_uri is not None:
        with to_dataset_path(sample.path_or_uri, storage_options).open("rb") as stream:
            yield stream
    else:
        raise ValueError(f"image sample {sample.image_id!r} has neither path_or_uri nor image_bytes")


def _tiff_series_to_hwc(series: Any) -> np.ndarray:
    """A TIFF image series as ``(H, W, bands)``; ``ValueError`` if unsupported.

    Single-band (``YX``), band-last (``YXS``), and band-first (``SYX``, or
    ``CYX`` for a multi-page channel stack such as OME-TIFF) layouts are
    supported. Other page stacks (``IYX``, ``QYX``, ``ZYX``, ``TYX``, ...) are
    rejected rather than read as bands: their pages may be time steps, depth
    slices, or separate images. The decoded size is checked before reading pixels, against the
    same byte cap as SafeTensors tensors, so a small compressed file cannot
    expand into an unbounded array.
    """
    layouts = {"YX": None, "YXS": None, "YXC": None, "SYX": (1, 2, 0), "CYX": (1, 2, 0)}
    axes = series.axes
    if axes not in layouts:
        raise ValueError(
            f"unsupported TIFF layout {axes!r}: native mode reads single images (YX, YXS) and channel "
            "stacks (SYX, CYX); label a multi-page channel stack's page axis as channels (C)"
        )
    shape = tuple(int(n) for n in series.shape)
    if shape[axes.index("Y")] * shape[axes.index("X")] > MAX_DECODED_IMAGE_PIXELS:
        raise ValueError("image exceeds the decoded-pixel cap")
    if int(np.prod(shape)) * np.dtype(series.dtype).itemsize > MAX_ENCODED_IMAGE_BYTES:
        raise ValueError(f"decoded raster {shape} {series.dtype} exceeds the {MAX_ENCODED_IMAGE_BYTES} byte cap")
    array = series.asarray()
    if axes == "YX":
        return array[:, :, None]
    order = layouts[axes]
    return array if order is None else np.transpose(array, order)
