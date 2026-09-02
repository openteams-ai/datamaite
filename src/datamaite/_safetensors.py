"""Minimal SafeTensors image reading (IR-3.2-S-1 flat-image support).

Deliberately *not* a general safetensors reader: it recognises the subset of
tensors that plausibly encode one still image and normalises them to RGB
``uint8`` HWC for the MAITE decode surface. Two callers, two I/O budgets:

* :func:`read_entries` / :func:`image_entries` parse only the JSON header, so
  the flat-images *loader* can validate the *container* (official offset cover,
  not per-tensor salvage) and read image dimensions without transferring any
  tensor bytes;
* :func:`load_image_rgb_hwc` materialises one tensor with ``numpy`` (a core
  dependency) at MAITE decode time, range-reading exactly that tensor's byte
  span. No OpenCV: a tensor needs no image codec.

Every read goes through :mod:`datamaite._io`, so a local directory, an
``s3://`` prefix and a ``memory://`` root behave identically and a remote file
is never downloaded whole to reach a header or a single tensor.

Wire format (https://github.com/huggingface/safetensors#format): an 8-byte
little-endian unsigned header length ``N``, then ``N`` bytes of JSON mapping
tensor names to ``{"dtype", "shape", "data_offsets"}`` (offsets relative to
the data section that follows the header), then the raw tensor buffer.

Conventions this module imposes on "a tensor that is an image"
=============================================================

safetensors is a tensor container: it has no image-layout convention, so ours
has to be written down. It is a pure function of ``shape`` and ``dtype``:

* shape ``(H, W)`` (grayscale), ``(H, W, C)`` or ``(C, H, W)`` with ``C`` in
  ``{1, 3, 4}``. HWC is tried first, so CHW only wins when the last dimension
  is not channel-like. An ambiguous shape whose first *and* last dimension are
  both channel-like (``(3, 3, 3)``, ``(3, 480, 4)``, ``(1, H, 1)``) reads as
  HWC and the loader warns that HWC was assumed, so the guess is visible;
* channel order RGB (RGBA's alpha is dropped, grayscale is replicated). Encoded
  images already reach MAITE as RGB, and ML tensors are RGB;
* dtype ``uint8`` passes through; bool maps to ``{0, 255}``; floats in
  ``[0, 1]`` scale by 255, wider floats clip into ``[0, 255]``; ``uint16``
  scales by the type max so a full-range 16-bit value is not clipped to white;
  signed integers (``int8`` / ``int16``) are clamped so a genuine ``0`` stays
  black. ``uint32`` / ``uint64`` / ``int32`` / ``int64`` are not images.
  ``BF16`` / ``F8_*`` are parsed for span validation (official cover) but are
  never decoded as images;
* the source span (``end - start``) *and* the ``H * W * 3`` uint8 RGB output
  (after replicate / drop-alpha) are each refused above
  :data:`MAX_ENCODED_IMAGE_BYTES` before the tensor is read, local and remote:
  a 100 MP grayscale ``U8`` tensor is a 100 MB source that becomes 300 MB of
  RGB. :data:`~datamaite._io.MAX_DECODED_IMAGE_PIXELS` is a cheaper ``H * W``
  early-out (~89 MP RGB is the binding 256 MiB limit; 200 MP would already
  have failed the byte cap). Do not treat the pixel cap as load-bearing.

``__metadata__`` is **ignored**. It is one file-level flat string-to-string map,
so it cannot describe the individual tensors of a multi-image file, and the
ecosystem has no image vocabulary for it: honouring private keys would narrow
interoperability while looking liberal. If an escape hatch is ever needed it
belongs on the caller (a loader option applied to the whole load), not on the
producer.
"""

from __future__ import annotations

import json
import logging
import math
import struct
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from datamaite._io import (
    MAX_DECODED_IMAGE_PIXELS,
    read_resource_prefix,
    read_resource_range,
    resource_size,
)
from datamaite._upath import is_remote_path, to_dataset_path

logger = logging.getLogger(__name__)

SAFETENSORS_SUFFIX = ".safetensors"
_HEADER_PREFIX_BYTES = 8
# The reference implementation caps the JSON header at 100 MB; anything larger
# is malformed (or hostile) input, not a real dataset.
_MAX_HEADER_BYTES = 100_000_000

# Source span and H*W*3 RGB output, local and remote, checked before the tensor
# read. 256 MiB of RGB is ~89 MP, so MAX_DECODED_IMAGE_PIXELS (200 MP) is only a
# cheap pre-read early-out — this constant is the load-bearing bound.
MAX_ENCODED_IMAGE_BYTES = 256 * 1024 * 1024

#: Official dtype code -> itemsize. Includes types we never decode as images
#: (BF16, F8, wide integers) so their spans still participate in the container
#: cover. An unrecognized code is container-fatal (safe_open rejects it).
_ITEMSIZES: dict[str, int] = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "U16": 2,
    "I16": 2,
    "U32": 4,
    "I32": 4,
    "U64": 8,
    "I64": 8,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "BF16": 2,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
}
#: numpy dtype spec for tensors we can materialise. Explicitly little-endian.
_NUMPY_DTYPES: dict[str, str] = {
    "BOOL": "|b1",
    "U8": "|u1",
    "I8": "|i1",
    "U16": "<u2",
    "I16": "<i2",
    "F16": "<f2",
    "F32": "<f4",
    "F64": "<f8",
}
_IMAGE_DTYPES = frozenset(_NUMPY_DTYPES)
_IMAGE_CHANNEL_COUNTS = frozenset({1, 3, 4})

#: ``fs.info()`` keys that identify *which bytes*, best first. Backends disagree
#: on the name and some offer only a creation stamp; without any of them a
#: same-size rewrite would keep its cached manifest.
_REMOTE_STAMP_KEYS = ("etag", "ETag", "version", "mtime", "LastModified", "last_modified", "created")

_DECODE_READ: ContextVar[tuple[str | Path, Mapping[str, Any] | None] | None] = ContextVar(
    "datamaite_safetensors_decode_read", default=None
)


@dataclass(frozen=True)
class SafeTensorEntry:
    """One tensor recorded in a safetensors header.

    ``data_offsets`` are absolute file offsets (header-relative offsets from
    the wire format are resolved at parse time) so a decode can range-read
    directly, with the end offset exclusive.
    """

    name: str
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]


def read_entries(path: str | Path, storage_options: Mapping[str, Any] | None = None) -> list[SafeTensorEntry]:
    """Parse the header of ``path`` and return its tensors, sorted by name.

    Structural validation follows the official cover: readable offsets, no
    overlap or gap, last end equals the data-section size, known dtype, and
    dtype x shape matches the span. Unreadable or out-of-bounds offsets are
    container-fatal — a valid sibling does not load. Whether a tensor is
    image-like is :func:`image_entries`' job. ``__metadata__`` is skipped.
    ``OSError`` propagates from unreadable resources.
    """
    header, data_start, data_size = _read_header(path, storage_options)
    _validate_container(header, data_size=data_size, path=path)
    return [
        _parse_entry(name, spec, data_start=data_start)
        for name, spec in sorted(header.items())
        if name != "__metadata__"
    ]


def _read_header(path: str | Path, storage_options: Mapping[str, Any] | None) -> tuple[dict[str, Any], int, int]:
    """Read and validate the container prefix: ``(header, data_start, data_size)``.

    Two ranged reads and one stat, never a tensor byte. Prefix, header JSON,
    and offset-cover failures are all fatal to the whole file.
    """
    prefix = read_resource_prefix(path, _HEADER_PREFIX_BYTES, storage_options)
    if len(prefix) < _HEADER_PREFIX_BYTES:
        raise ValueError(f"truncated safetensors header prefix in {path}")
    (header_len,) = struct.unpack("<Q", prefix)
    if header_len > _MAX_HEADER_BYTES:
        raise ValueError(f"safetensors header length {header_len} exceeds the {_MAX_HEADER_BYTES} byte cap")
    file_size = resource_size(path, storage_options)
    if file_size is None:
        raise ValueError(f"could not determine the size of safetensors file {path}")
    data_start = _HEADER_PREFIX_BYTES + header_len
    if data_start > file_size:
        raise ValueError(f"safetensors header length {header_len} exceeds the file size of {path}")
    raw_header = read_resource_range(path, _HEADER_PREFIX_BYTES, data_start, storage_options)
    if len(raw_header) < header_len:
        raise ValueError(f"truncated safetensors header in {path}")
    try:
        header = json.loads(raw_header)
    except RecursionError as exc:
        raise ValueError(f"safetensors header JSON in {path} exceeds the parser recursion limit") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid safetensors header JSON in {path}: {exc}") from exc
    if not isinstance(header, dict):
        raise ValueError(f"safetensors header in {path} is not a JSON object")
    return header, data_start, file_size - data_start


def _is_int(value: Any) -> bool:
    """A real integer, not a JSON ``true``/``false`` (which ``int`` accepts)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_container(header: dict[str, Any], *, data_size: int, path: str | Path) -> None:
    """Official relative-offset cover: unreadable or out-of-bounds is fatal."""
    spans: list[tuple[int, int, str]] = []
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        start, end = _relative_span(name, spec, data_size=data_size, path=path)
        spans.append((start, end, name))
    spans.sort(key=lambda item: item[0])
    cursor = 0
    for start, end, name in spans:
        if start != cursor:
            raise ValueError(f"safetensors tensor {name!r} in {path} has data_offsets that leave a gap or overlap")
        cursor = end
    if cursor != data_size:
        raise ValueError(f"safetensors data section in {path} is not fully covered by tensor offsets")


def _relative_span(name: str, spec: Any, *, data_size: int, path: str | Path) -> tuple[int, int]:
    """Return relative ``(start, end)`` or raise: unreadable offsets fail the file."""
    if not isinstance(spec, dict):
        raise ValueError(f"safetensors tensor {name!r} in {path} is not a JSON object")
    dtype = spec.get("dtype")
    shape = spec.get("shape")
    offsets = spec.get("data_offsets")
    if not isinstance(dtype, str):
        raise ValueError(f"safetensors tensor {name!r} in {path} has no dtype string")
    itemsize = _ITEMSIZES.get(dtype)
    if itemsize is None:
        raise ValueError(f"safetensors tensor {name!r} in {path} has unrecognized dtype {dtype!r}")
    if not isinstance(shape, list) or not all(_is_int(dim) and dim >= 0 for dim in shape):
        raise ValueError(f"safetensors tensor {name!r} in {path} has an invalid shape")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or not all(_is_int(off) for off in offsets)
        or not 0 <= offsets[0] <= offsets[1] <= data_size
    ):
        raise ValueError(f"safetensors tensor {name!r} in {path} has data_offsets outside the file")
    start, end = offsets[0], offsets[1]
    if end - start != math.prod(shape) * itemsize:
        raise ValueError(f"safetensors tensor {name!r} in {path} has a byte span that does not match its shape")
    return start, end


def _parse_entry(name: str, spec: Any, *, data_start: int) -> SafeTensorEntry:
    offsets = spec["data_offsets"]
    return SafeTensorEntry(
        name=name,
        dtype=spec["dtype"],
        shape=tuple(spec["shape"]),
        data_offsets=(data_start + offsets[0], data_start + offsets[1]),
    )


def image_layout(shape: tuple[int, ...]) -> tuple[int, int, str] | None:
    """Return ``(height, width, layout)`` when ``shape`` plausibly encodes one image.

    ``layout`` is ``"hw"``, ``"hwc"``, or ``"chw"``. HWC is tried first, so an
    ambiguous 3-D shape whose first *and* last dims are channel-like (e.g.
    ``(3, 3, 3)``) reads as HWC, the dominant convention for decoded images;
    :func:`layout_is_ambiguous` reports that case so callers can warn.

    ``H * W`` above ``MAX_DECODED_IMAGE_PIXELS`` is a cheap early-out. The
    load-bearing bound is ``H * W * 3`` against :data:`MAX_ENCODED_IMAGE_BYTES`.
    """
    if len(shape) == 2 and shape[0] >= 1 and shape[1] >= 1:
        return _bounded(shape[0], shape[1], "hw")
    if len(shape) == 3:
        if shape[2] in _IMAGE_CHANNEL_COUNTS and shape[0] >= 1 and shape[1] >= 1:
            return _bounded(shape[0], shape[1], "hwc")
        if shape[0] in _IMAGE_CHANNEL_COUNTS and shape[1] >= 1 and shape[2] >= 1:
            return _bounded(shape[1], shape[2], "chw")
    return None


def _bounded(height: int, width: int, layout: str) -> tuple[int, int, str] | None:
    if height * width > MAX_DECODED_IMAGE_PIXELS:
        return None
    return height, width, layout


def layout_is_ambiguous(shape: tuple[int, ...]) -> bool:
    """Whether ``shape`` reads as both HWC and CHW, so HWC was merely assumed.

    Only collides when a spatial extent is 1, 3, or 4 -- tiny tiles and
    few-pixel-wide strips, not photographs.
    """
    return len(shape) == 3 and shape[0] in _IMAGE_CHANNEL_COUNTS and shape[2] in _IMAGE_CHANNEL_COUNTS


def image_entries(entries: list[SafeTensorEntry]) -> list[tuple[SafeTensorEntry, tuple[int, int, str]]]:
    """Pair each tensor this module can decode as an image with its layout."""
    pairs: list[tuple[SafeTensorEntry, tuple[int, int, str]]] = []
    for entry in entries:
        layout = image_layout(entry.shape) if entry.dtype in _IMAGE_DTYPES else None
        if layout is None:
            continue
        source_bytes = entry.data_offsets[1] - entry.data_offsets[0]
        output_bytes = layout[0] * layout[1] * 3
        if source_bytes > MAX_ENCODED_IMAGE_BYTES or output_bytes > MAX_ENCODED_IMAGE_BYTES:
            # An image-shaped tensor that could never decode: say so, rather
            # than returning a sample whose every MAITE access must fail.
            logger.warning(
                "Skipping image-shaped safetensors tensor %r: its source span is %d bytes and its "
                "%d x %d RGB output is %d bytes, over the %d byte cap",
                entry.name,
                source_bytes,
                layout[0],
                layout[1],
                output_bytes,
                MAX_ENCODED_IMAGE_BYTES,
            )
            continue
        pairs.append((entry, layout))
    return pairs


def _resource_fingerprint(resolved: Path) -> tuple[Any, ...]:
    """Hashable identity plus size and a version stamp from one stat/info. No credentials."""
    if is_remote_path(resolved):
        fs = resolved.fs  # type: ignore[attr-defined]
        info = fs.info(resolved.path)  # type: ignore[attr-defined]
        size = int(info["size"])
        # Whatever this backend calls "the version of these bytes". Without one,
        # a same-size rewrite would keep its cache entry: memory:// publishes
        # only ``created``, and S3/GCS/Azure publish an ETag.
        stamp = next((str(info[key]) for key in _REMOTE_STAMP_KEYS if info.get(key)), "")
        token = getattr(fs, "_fs_token", b"")
        token_key = token.hex() if isinstance(token, bytes) else str(token)
        protocol = getattr(resolved, "protocol", "")
        if isinstance(protocol, (tuple, list)):
            protocol = protocol[0] if protocol else ""
        return ("remote", str(protocol), token_key, str(getattr(resolved, "path", resolved)), size, stamp)
    stat = resolved.stat()
    return ("local", str(resolved.resolve(strict=False)), int(stat.st_size), int(stat.st_mtime_ns))


@lru_cache(maxsize=32)
def _cached_decode_entries(_fingerprint: tuple[Any, ...]) -> dict[str, SafeTensorEntry]:
    """Decode-path name index. Load uses :func:`read_entries` uncached.

    ``fingerprint`` is computed outside this function. The path and storage
    options used to read live in a contextvar so they are not cache keys
    (mappings are unhashable; credentials must not be). ``_fs_token`` stays
    in this in-process cache and never reaches sample metadata.
    """
    pending = _DECODE_READ.get()
    if pending is None:
        raise RuntimeError("safetensors decode cache missed its read context")
    path, storage_options = pending
    return {entry.name: entry for entry in read_entries(path, storage_options)}


def _entries_for_decode(path: str | Path, storage_options: Mapping[str, Any] | None) -> dict[str, SafeTensorEntry]:
    resolved = to_dataset_path(path, storage_options)
    fingerprint = _resource_fingerprint(resolved)
    token = _DECODE_READ.set((path, storage_options))
    try:
        return _cached_decode_entries(fingerprint)
    finally:
        _DECODE_READ.reset(token)


def load_image_rgb_hwc(path: str | Path, name: str, storage_options: Mapping[str, Any] | None = None) -> np.ndarray:
    """Load tensor ``name`` from ``path`` as an ``(H, W, 3)`` ``uint8`` RGB array.

    Decode reuses a fingerprint-keyed header cache (path identity, size, and
    ``st_mtime_ns`` / ETag from one stat) so an N-tensor file does not re-parse
    N times. A rewrite that changes the fingerprint misses the cache and fails
    loudly instead of reading stale offsets. Source-span and ``H * W * 3``
    caps apply locally and remotely, before the tensor read. Raises
    ``ValueError`` for a missing/non-image tensor and propagates
    ``ValueError``/``OSError`` from header parsing.
    """
    entry = _entries_for_decode(path, storage_options).get(name)
    if entry is None:
        raise ValueError(f"safetensors file {path} has no tensor named {name!r}")
    layout = image_layout(entry.shape)
    dtype = _NUMPY_DTYPES.get(entry.dtype)
    if layout is None or dtype is None or entry.dtype not in _IMAGE_DTYPES:
        raise ValueError(f"safetensors tensor {name!r} in {path} is not decodable as an image")

    start, end = entry.data_offsets
    span = end - start
    height, width, kind = layout
    output_bytes = height * width * 3
    if span > MAX_ENCODED_IMAGE_BYTES or output_bytes > MAX_ENCODED_IMAGE_BYTES:
        raise ValueError(
            f"safetensors tensor {name!r} in {path} spans {span} bytes "
            f"(output {output_bytes} bytes), over the {MAX_ENCODED_IMAGE_BYTES} byte cap"
        )
    buffer = read_resource_range(path, start, end, storage_options)
    if len(buffer) < span:
        raise ValueError(f"safetensors tensor {name!r} in {path} is truncated")
    array = np.frombuffer(buffer, dtype=np.dtype(dtype)).reshape(entry.shape)

    # Annotated as the plain ndarray type: pyright otherwise over-narrows the
    # shape tuple from reshape() and rejects the shape[2] checks below.
    hwc: np.ndarray
    if kind == "hw":
        hwc = array[:, :, None]
    elif kind == "chw":
        hwc = np.transpose(array, (1, 2, 0))
    else:
        hwc = array
    hwc = _to_uint8(hwc)
    if hwc.shape[2] == 1:
        hwc = np.repeat(hwc, 3, axis=2)
    elif hwc.shape[2] == 4:
        hwc = hwc[:, :, :3]
    return np.ascontiguousarray(hwc)


def _to_uint8(array: np.ndarray) -> np.ndarray:
    """Normalise an image array to ``uint8``: U16 scales by type max, signed clamps."""
    if array.dtype == np.uint8:
        return array
    if array.dtype == np.bool_:
        return array.astype(np.uint8) * 255
    if np.issubdtype(array.dtype, np.floating):
        finite = np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0, copy=False)
        if finite.size and finite.min() >= 0.0 and finite.max() <= 1.0:
            finite = finite * 255.0
        return np.clip(np.rint(finite), 0, 255).astype(np.uint8)
    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        if info.min >= 0 and info.max > 255:
            scaled = array.astype(np.float32) * (255.0 / float(info.max))
            return np.clip(np.rint(scaled), 0, 255).astype(np.uint8)
    return np.clip(array, 0, 255).astype(np.uint8)
