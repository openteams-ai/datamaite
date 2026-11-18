"""The pure-Python image header reader: stored size and JPEG EXIF orientation."""

from __future__ import annotations

import io
import struct
import zlib
from pathlib import Path

import pytest

from datamaite._image_header import MAX_HEADER_SCAN, ImageHeader, parse_image_header, read_image_header


def _png(width: int, height: int) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    rows = b"".join(b"\x00" + b"\x00\x00\x00" * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def _gif(width: int, height: int) -> bytes:
    return b"GIF89a" + struct.pack("<HH", width, height) + b"\x00" * 24


def _bmp(width: int, height: int) -> bytes:
    dib = struct.pack("<IiiHH", 40, width, height, 1, 24) + b"\x00" * 24
    return b"BM" + struct.pack("<IHHI", 14 + len(dib), 0, 0, 14 + len(dib)) + dib


def _bmp_core(width: int, height: int) -> bytes:
    """An OS/2 BITMAPCOREHEADER, with 16-bit sizes."""
    dib = struct.pack("<IHHHH", 12, width, height, 1, 24) + b"\x00" * 16
    return b"BM" + struct.pack("<IHHI", 14 + len(dib), 0, 0, 14 + len(dib)) + dib


def _exif(orientation: int, endian: str) -> bytes:
    order = b"II*\x00" if endian == "<" else b"MM\x00*"
    other = struct.pack(f"{endian}HHI", 0x010F, 2, 4) + b"cam\x00"  # Make, before the orientation
    tag = struct.pack(f"{endian}HHIH", 0x0112, 3, 1, orientation) + b"\x00\x00"
    return b"Exif\x00\x00" + order + struct.pack(f"{endian}IH", 8, 2) + other + tag + b"\x00\x00\x00\x00"


def _jpeg(width: int, height: int, *, orientation: int | None = None, endian: str = ">", padding: int = 0) -> bytes:
    segments = b""
    while padding > 0:  # APP2 segments before the frame
        size = min(padding, 60_000)
        segments += b"\xff\xe2" + struct.pack(">H", size + 2) + b"\x00" * size
        padding -= size
    if orientation is not None:
        exif = _exif(orientation, endian)
        segments += b"\xff\xe1" + struct.pack(">H", len(exif) + 2) + exif
    sof = b"\x08" + struct.pack(">HH", height, width) + b"\x03\x01\x11\x00\x02\x11\x00\x03\x11\x00"
    return b"\xff\xd8" + segments + b"\xff\xc0" + struct.pack(">H", len(sof) + 2) + sof + b"\xff\xd9"


def _webp_vp8(width: int, height: int) -> bytes:
    frame = b"\x10\x02\x00" + b"\x9d\x01\x2a" + struct.pack("<HH", width, height) + b"\x00" * 8
    return b"RIFF" + struct.pack("<I", 4 + 8 + len(frame)) + b"WEBP" + b"VP8 " + struct.pack("<I", len(frame)) + frame


def _webp_vp8l(width: int, height: int) -> bytes:
    bits = (width - 1) | ((height - 1) << 14)
    data = b"\x2f" + struct.pack("<I", bits) + b"\x00" * 8
    return b"RIFF" + struct.pack("<I", 4 + 8 + len(data)) + b"WEBP" + b"VP8L" + struct.pack("<I", len(data)) + data


def _webp_vp8x(width: int, height: int) -> bytes:
    data = b"\x00\x00\x00\x00" + (width - 1).to_bytes(3, "little") + (height - 1).to_bytes(3, "little")
    return b"RIFF" + struct.pack("<I", 4 + 8 + len(data)) + b"WEBP" + b"VP8X" + struct.pack("<I", len(data)) + data


def _tiff(width: int, height: int, endian: str, *, long_width: bool = False) -> bytes:
    """A TIFF whose IFD follows its pixel data, as libtiff writes it."""
    order = b"II*\x00" if endian == "<" else b"MM\x00*"
    pixels = b"\x00" * 64
    entries = [
        struct.pack(f"{endian}HHII", 256, 4, 1, width)
        if long_width
        else struct.pack(f"{endian}HHIH", 256, 3, 1, width) + b"\x00\x00",
        struct.pack(f"{endian}HHIH", 257, 3, 1, height) + b"\x00\x00",
        struct.pack(f"{endian}HHIH", 259, 3, 1, 1) + b"\x00\x00",
    ]
    ifd = struct.pack(f"{endian}H", len(entries)) + b"".join(entries) + b"\x00\x00\x00\x00"
    return order + struct.pack(f"{endian}I", 8 + len(pixels)) + pixels + ifd


@pytest.mark.parametrize(
    ("data", "size"),
    [
        (_png(64, 48), (64, 48)),
        (_gif(32, 16), (32, 16)),
        (_bmp(24, 12), (24, 12)),
        (_bmp(24, -12), (24, 12)),  # top-down rows
        (_bmp_core(20, 10), (20, 10)),
        (_jpeg(80, 40), (80, 40)),
        (_jpeg(80, 40, padding=1000), (80, 40)),
        (_webp_vp8(65, 33), (65, 33)),
        (_webp_vp8l(300, 7), (300, 7)),
        (_webp_vp8x(5000, 4000), (5000, 4000)),
        (_tiff(64, 48, "<"), (64, 48)),
        (_tiff(64, 48, ">"), (64, 48)),
        (_tiff(70000, 3, ">", long_width=True), (70000, 3)),
    ],
    ids=[
        "png",
        "gif",
        "bmp",
        "bmp-top-down",
        "bmp-core",
        "jpeg",
        "jpeg-app2",
        "webp-vp8",
        "webp-vp8l",
        "webp-vp8x",
        "tiff-le",
        "tiff-be",
        "tiff-long",
    ],
)
def test_reads_the_stored_size(data: bytes, size: tuple[int, int], tmp_path: Path) -> None:
    path = tmp_path / "image"
    path.write_bytes(data)
    assert read_image_header(data) == ImageHeader(*size)
    assert read_image_header(path) == ImageHeader(*size)
    assert read_image_header(str(path)) == ImageHeader(*size)


@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize(
    ("orientation", "display_size"),
    [
        (1, (64, 32)),
        (2, (64, 32)),
        (3, (64, 32)),
        (4, (64, 32)),
        (5, (32, 64)),
        (6, (32, 64)),
        (7, (32, 64)),
        (8, (32, 64)),
    ],
)
def test_reads_the_jpeg_exif_orientation(orientation: int, display_size: tuple[int, int], endian: str) -> None:
    header = read_image_header(_jpeg(64, 32, orientation=orientation, endian=endian))
    assert header == ImageHeader(64, 32, orientation)
    assert header is not None
    assert header.display_size == display_size
    assert header.transposed is (orientation >= 5)


def test_an_out_of_range_orientation_means_upright() -> None:
    assert read_image_header(_jpeg(64, 32, orientation=9)) == ImageHeader(64, 32, 1)


@pytest.mark.optional
@pytest.mark.parametrize("ext", [".png", ".jpg", ".bmp", ".webp", ".tif"])
def test_matches_opencv_encodings(ext: str) -> None:
    import cv2
    import numpy as np

    ok, data = cv2.imencode(ext, np.zeros((48, 64, 3), np.uint8))
    assert ok
    assert read_image_header(data.tobytes()) == ImageHeader(64, 48)


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"not an image at all",
        _png(64, 48)[:20],  # cut inside IHDR
        _png(0, 48),
        _jpeg(80, 40)[:6],
        b"\xff\xd8" + b"\xff\xe2" + struct.pack(">H", 65535) + b"\x00" * 10,  # segment runs past the end
        _jpeg(80, 40, padding=MAX_HEADER_SCAN),  # frame header beyond the scan limit
        _tiff(64, 48, "<")[:80],  # IFD cut off
        b"RIFF\x00\x00\x00\x00WEBPVP8L" + b"\x00" * 20,  # bad VP8L signature
    ],
    ids=["empty", "unknown", "png-cut", "png-zero", "jpeg-cut", "jpeg-overrun", "jpeg-too-far", "tiff-cut", "vp8l-bad"],
)
def test_unreadable_headers_give_none(data: bytes) -> None:
    assert read_image_header(data) is None


def test_jpeg_fill_bytes_stop_at_the_scan_limit() -> None:
    """A run of 0xFF fill bytes longer than the scan limit is not read to its end."""
    data = b"\xff\xd8" + b"\xff" * (4 * MAX_HEADER_SCAN) + _jpeg(80, 40)[2:]

    class Counting(io.BytesIO):
        reads = 0

        def read(self, size: int | None = -1) -> bytes:
            Counting.reads += 1
            return super().read(size)

    stream = Counting(data)

    assert parse_image_header(stream) is None
    assert stream.tell() <= MAX_HEADER_SCAN + 1
    assert Counting.reads <= MAX_HEADER_SCAN + 8


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


@pytest.mark.parametrize("after_pixels", [False, True], ids=["before-idat", "after-idat"])
def test_reads_the_png_exif_orientation(after_pixels: bool) -> None:
    png = _png(64, 32)
    exif = _png_chunk(b"eXIf", _exif(6, ">").removeprefix(b"Exif\x00\x00"))
    at = png.rfind(b"IEND") - 4 if after_pixels else 8 + 25  # before IEND, or right after IHDR
    header = read_image_header(png[:at] + exif + png[at:])
    assert header is not None
    assert header.display_size == (32, 64)


@pytest.mark.parametrize("jpeg_prefix", [False, True], ids=["tiff", "exif-prefixed"])
def test_reads_the_webp_exif_orientation(jpeg_prefix: bool) -> None:
    """The EXIF chunk follows the image data, and the VP8X header flags it."""
    exif = _exif(6, "<")
    exif = exif if jpeg_prefix else exif.removeprefix(b"Exif\x00\x00")
    vp8x = b"\x08\x00\x00\x00" + (64 - 1).to_bytes(3, "little") + (32 - 1).to_bytes(3, "little")
    image = b"\x00" * 1001  # an odd size, so the chunk is padded
    # Each chunk: its type, its size, then its data, padded to an even length.
    chunks = [(b"VP8X", vp8x), (b"VP8L", image), (b"EXIF", exif)]
    body = b"WEBP" + b"".join(
        kind + struct.pack("<I", len(data)) + data + b"\x00" * (len(data) & 1) for kind, data in chunks
    )
    header = read_image_header(b"RIFF" + struct.pack("<I", len(body)) + body)
    assert header is not None
    assert header.display_size == (32, 64)


def test_a_jpeg_takes_its_first_exif_segment() -> None:
    """As OpenCV does: an upright first segment wins over a rotated second one."""
    first, second = _exif(1, ">"), _exif(6, ">")
    segments = b"".join(b"\xff\xe1" + struct.pack(">H", len(e) + 2) + e for e in (first, second))
    jpeg = _jpeg(64, 32)
    assert read_image_header(jpeg[:2] + segments + jpeg[2:]) == ImageHeader(64, 32, 1)


def test_an_orientation_stored_as_long_is_ignored() -> None:
    """OpenCV only reads a SHORT orientation, so a LONG one decodes upright."""
    tiff = b"MM\x00*" + struct.pack(">IH", 8, 1) + struct.pack(">HHII", 0x0112, 4, 1, 6) + b"\x00" * 4
    exif = b"Exif\x00\x00" + tiff
    jpeg = _jpeg(64, 32)
    data = jpeg[:2] + b"\xff\xe1" + struct.pack(">H", len(exif) + 2) + exif + jpeg[2:]
    assert read_image_header(data) == ImageHeader(64, 32, 1)


def test_a_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(OSError, match="missing"):
        read_image_header(tmp_path / "missing.png")


def test_reads_a_remote_file(memory_root) -> None:  # type: ignore[no-untyped-def]
    path = memory_root / "photos" / "a.jpg"
    path.write_bytes(_jpeg(64, 32, orientation=6))
    assert read_image_header(path) == ImageHeader(64, 32, 6)
    assert read_image_header("memory:///" + path.path.lstrip("/")) == ImageHeader(64, 32, 6)
