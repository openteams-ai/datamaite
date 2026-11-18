"""Read an image's size and EXIF orientation from its header, without an image library.

Core datamaite has no Pillow, so everything that needs an image's size without
decoding it reads the header here: loaders (YOLO de-normalises boxes against
it; COCO checks it against EXIF-rotated photos), and, through
``datamaite._io.probe_image_dimensions``, writers and the MAITE datasets.
Supported: PNG, GIF, BMP, JPEG, WebP (``VP8``, ``VP8L`` and ``VP8X``) and TIFF
in either byte order.

:class:`ImageHeader` keeps the stored (raw) size and the EXIF orientation of a
JPEG (its first Exif segment), a PNG (its ``eXIf`` chunk) or a WebP (its
``EXIF`` chunk). OpenCV, and so Ultralytics and datamaite's MAITE decoding,
apply that orientation when decoding, so :attr:`ImageHeader.display_size` is
the size of the decoded image. Orientations 5-8 swap width and height. As
OpenCV does, an orientation is read only when stored as a SHORT. A JPEG's
segments and a PNG's chunks are looked for only in the first
:data:`MAX_HEADER_SCAN` bytes, so a PNG ``eXIf`` chunk after that is missed.
"""

from __future__ import annotations

import io
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from datamaite._upath import is_remote_path, to_dataset_path

#: JPEG marker segments (and so the EXIF block and the frame header) and PNG
#: chunks are only looked for in this many leading bytes.
MAX_HEADER_SCAN = 256 * 1024
_REMOTE_BLOCK_SIZE = 64 * 1024
_SOF_MARKERS = frozenset({0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF})
_EXIF_ORIENTATION_TAG = 0x0112
_TIFF_WIDTH_TAG, _TIFF_HEIGHT_TAG = 256, 257
_MAX_IFD_ENTRIES = 4096
_MAX_EXIF_BYTES = 64 * 1024
_MAX_WEBP_CHUNKS = 64
_VP8X_EXIF_FLAG = 0x08


@dataclass(frozen=True)
class ImageHeader:
    """An image's stored size and EXIF orientation (1, the default, means upright)."""

    width: int
    height: int
    orientation: int = 1

    @property
    def transposed(self) -> bool:
        """Whether the orientation swaps width and height (EXIF orientations 5-8)."""
        return self.orientation in (5, 6, 7, 8)

    @property
    def display_size(self) -> tuple[int, int]:
        """``(width, height)`` once the orientation is applied, as OpenCV decodes the image."""
        return (self.height, self.width) if self.transposed else (self.width, self.height)


def to_display_frame(hwc: np.ndarray, orientation: int) -> np.ndarray:
    """``hwc``, an ``(H, W, C)`` array in the stored frame, turned and flipped as EXIF ``orientation`` says.

    The same as OpenCV applies when decoding, for any dtype and band count. A view, not a copy.
    """
    if orientation == 2:
        return hwc[:, ::-1]
    if orientation == 3:
        return hwc[::-1, ::-1]
    if orientation == 4:
        return hwc[::-1]
    transposed = hwc.transpose(1, 0, *range(2, hwc.ndim))
    if orientation == 5:
        return transposed
    if orientation == 6:
        return transposed[:, ::-1]
    if orientation == 7:
        return transposed[::-1, ::-1]
    if orientation == 8:
        return transposed[::-1]
    return hwc


def read_image_header(
    source: bytes | str | Path, storage_options: Mapping[str, Any] | None = None
) -> ImageHeader | None:
    """Read the header of encoded image bytes or of a local or remote image file.

    Returns ``None`` when the format is not recognised or its header is
    malformed. Raises ``OSError`` when the file cannot be read.
    """
    if isinstance(source, bytes):
        return parse_image_header(io.BytesIO(source))
    path = to_dataset_path(source, storage_options) if isinstance(source, str) else source
    # A small block size stops fsspec reading ahead far past the header.
    open_kwargs = {"block_size": _REMOTE_BLOCK_SIZE} if is_remote_path(path) else {}
    with path.open("rb", **open_kwargs) as fh:  # type: ignore[call-overload]
        return parse_image_header(fh)


def parse_image_header(fh: BinaryIO) -> ImageHeader | None:
    """Read the header from a seekable binary stream positioned at the start of the image."""
    try:
        head = fh.read(32)
        if head.startswith(b"\x89PNG\r\n\x1a\n"):
            return _png(head, fh)
        if head.startswith((b"GIF87a", b"GIF89a")):
            return _positive(*struct.unpack("<HH", head[6:10]))
        if head.startswith(b"BM"):
            return _bmp(head)
        if head.startswith(b"\xff\xd8"):
            return _jpeg(fh)
        if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
            return _webp(head, fh)
        if head[:4] in (b"II*\x00", b"MM\x00*"):
            return _tiff(fh)
    except (struct.error, IndexError, ValueError):
        return None
    return None


def _positive(width: int, height: int, orientation: int = 1) -> ImageHeader | None:
    if width > 0 and height > 0:
        return ImageHeader(width, height, orientation if 1 <= orientation <= 8 else 1)
    return None


def _png(head: bytes, fh: BinaryIO) -> ImageHeader | None:
    if head[12:16] != b"IHDR":
        return None
    width, height = struct.unpack(">II", head[16:24])
    return _positive(width, height, _png_orientation(fh))


def _png_orientation(fh: BinaryIO) -> int:
    """The orientation in a PNG's ``eXIf`` chunk, looked for among the chunks in the scan budget."""
    position = 8  # after the signature
    while position + 8 <= MAX_HEADER_SCAN:
        fh.seek(position)
        header = fh.read(8)
        if len(header) < 8:
            return 1
        length, kind = struct.unpack(">I", header[:4])[0], header[4:]
        if kind == b"eXIf":
            return _exif_orientation(fh.read(min(length, _MAX_EXIF_BYTES)))
        if kind == b"IEND":
            return 1
        position += 12 + length  # length, type, data, CRC
    return 1


def _bmp(head: bytes) -> ImageHeader | None:
    dib_size = struct.unpack("<I", head[14:18])[0]
    if dib_size == 12:  # BITMAPCOREHEADER: unsigned 16-bit sizes
        return _positive(*struct.unpack("<HH", head[18:22]))
    width, height = struct.unpack("<ii", head[18:26])
    return _positive(width, abs(height))  # a negative height means top-down rows


def _webp(head: bytes, fh: BinaryIO) -> ImageHeader | None:
    chunk = head[12:16]
    if chunk == b"VP8X":
        width, height = 1 + int.from_bytes(head[24:27], "little"), 1 + int.from_bytes(head[27:30], "little")
        orientation = _webp_orientation(fh) if head[20] & _VP8X_EXIF_FLAG else 1
        return _positive(width, height, orientation)
    if chunk == b"VP8L":
        if head[20] != 0x2F:
            return None
        bits = int.from_bytes(head[21:25], "little")
        return _positive((bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1)
    if chunk == b"VP8 ":
        # 3-byte frame tag, the start code 9d 01 2a, then 14-bit width and height.
        data = head[20:30] if len(head) >= 30 else head[20:] + fh.read(30 - len(head))
        if data[3:6] != b"\x9d\x01\x2a":
            return None
        width, height = struct.unpack("<HH", data[6:10])
        return _positive(width & 0x3FFF, height & 0x3FFF)
    return None


def _webp_orientation(fh: BinaryIO) -> int:
    """The orientation in an extended WebP's ``EXIF`` chunk, which usually follows the image data.

    Only the chunk headers are read on the way, so this is a few small reads
    however large the image.
    """
    position = 12  # after "RIFF", the file size and "WEBP"
    for _ in range(_MAX_WEBP_CHUNKS):
        fh.seek(position)
        header = fh.read(8)
        if len(header) < 8:
            return 1
        kind, length = header[:4], struct.unpack("<I", header[4:])[0]
        if kind == b"EXIF":
            data = fh.read(min(length, _MAX_EXIF_BYTES))
            # Some writers keep JPEG's "Exif\0\0" prefix.
            return _exif_orientation(data.removeprefix(b"Exif\x00\x00"))
        position += 8 + length + (length & 1)  # chunks are padded to an even size
    return 1


def _jpeg(fh: BinaryIO) -> ImageHeader | None:  # noqa: C901 - JPEG marker scan is branchy
    orientation = 1
    exif_seen = False  # OpenCV uses the first Exif segment
    fh.seek(2)
    while fh.tell() < MAX_HEADER_SCAN:
        prefix = fh.read(1)
        if not prefix:
            return None
        if prefix != b"\xff":
            continue
        marker = fh.read(1)
        while marker == b"\xff" and fh.tell() < MAX_HEADER_SCAN:  # fill bytes
            marker = fh.read(1)
        if marker == b"\xff":  # fill bytes up to the scan limit
            return None
        if not marker:
            return None
        value = marker[0]
        if value in (0x01, 0xD8, 0xD9) or 0xD0 <= value <= 0xD7:  # markers with no length
            continue
        length_bytes = fh.read(2)
        if len(length_bytes) != 2:
            return None
        length = struct.unpack(">H", length_bytes)[0]
        if length < 2 or fh.tell() + length - 2 > MAX_HEADER_SCAN:
            return None
        if value in _SOF_MARKERS:
            data = fh.read(length - 2)
            if len(data) < 5:
                return None
            height, width = struct.unpack(">HH", data[1:5])
            return _positive(width, height, orientation)
        if value == 0xE1 and not exif_seen:
            data = fh.read(length - 2)
            if data.startswith(b"Exif\x00\x00"):
                orientation = _exif_orientation(data[6:])
                exif_seen = True
            continue
        fh.seek(length - 2, 1)
    return None


def _exif_orientation(tiff: bytes) -> int:
    """The orientation tag of an EXIF block's first IFD, or 1 when it has none."""
    try:
        endian = _tiff_endian(tiff[:4])
        if endian is None:
            return 1
        offset = struct.unpack(f"{endian}I", tiff[4:8])[0]
        count = struct.unpack(f"{endian}H", tiff[offset : offset + 2])[0]
        for index in range(min(count, _MAX_IFD_ENTRIES)):
            start = offset + 2 + index * 12
            tag, kind = struct.unpack(f"{endian}HH", tiff[start : start + 4])
            if tag == _EXIF_ORIENTATION_TAG and kind == 3:
                value = struct.unpack(f"{endian}H", tiff[start + 8 : start + 10])[0]
                return value if 1 <= value <= 8 else 1
    except struct.error:
        return 1
    return 1


def _tiff_endian(signature: bytes) -> str | None:
    if signature == b"II*\x00":
        return "<"
    if signature == b"MM\x00*":
        return ">"
    return None


def _tiff(fh: BinaryIO) -> ImageHeader | None:
    fh.seek(0)
    endian = _tiff_endian(fh.read(4))
    if endian is None:
        return None
    (offset,) = struct.unpack(f"{endian}I", fh.read(4))
    fh.seek(offset)
    (count,) = struct.unpack(f"{endian}H", fh.read(2))
    entries = fh.read(12 * min(count, _MAX_IFD_ENTRIES))
    values: dict[int, int] = {}
    for start in range(0, len(entries) - 11, 12):
        tag, kind, number = struct.unpack(f"{endian}HHI", entries[start : start + 8])
        if tag not in (_TIFF_WIDTH_TAG, _TIFF_HEIGHT_TAG) or number != 1:
            continue
        if kind == 3:  # SHORT
            values[tag] = struct.unpack(f"{endian}H", entries[start + 8 : start + 10])[0]
        elif kind == 4:  # LONG
            values[tag] = struct.unpack(f"{endian}I", entries[start + 8 : start + 12])[0]
    if _TIFF_WIDTH_TAG not in values or _TIFF_HEIGHT_TAG not in values:
        return None
    return _positive(values[_TIFF_WIDTH_TAG], values[_TIFF_HEIGHT_TAG])
