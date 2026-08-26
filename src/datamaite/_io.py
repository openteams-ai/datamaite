"""Shared storage-neutral I/O adapters built on UPath/fsspec.

Formats describe layouts; this module handles resource copying, probing, and
filesystem operations that native libraries cannot perform portably themselves.
"""

from __future__ import annotations

import contextlib
import contextvars
import shutil
import struct
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

import numpy as np

from datamaite._upath import is_remote_path, local_open_target, storage_options_for, to_dataset_path

_COPY_BUFFER_SIZE = 1024 * 1024
_MAX_REMOTE_ENCODED_IMAGE_BYTES = 256 * 1024 * 1024
_MAX_ENCODED_IMAGE_HEADER_BYTES = 256 * 1024
_INITIAL_ENCODED_IMAGE_HEADER_BYTES = 32
_MAX_DECODED_IMAGE_PIXELS = 200_000_000
_SOURCE_STORAGE_OPTIONS: contextvars.ContextVar[Mapping[str, Any] | None] = contextvars.ContextVar(
    "datamaite_source_storage_options", default=None
)
_WRITE_MODE: contextvars.ContextVar[str | None] = contextvars.ContextVar("datamaite_write_mode", default=None)


def runtime_storage_options(dataset: object) -> Mapping[str, Any]:
    """Private resolver options retained by a loaded dataset."""
    return getattr(dataset, "_runtime_storage_options", {})


@contextlib.contextmanager
def source_storage_context(dataset: object, override: Mapping[str, Any] | None = None) -> Iterator[None]:
    """Make a dataset's resolver available to deeply nested writer helpers."""
    token = _SOURCE_STORAGE_OPTIONS.set(override if override is not None else runtime_storage_options(dataset))
    try:
        yield
    finally:
        _SOURCE_STORAGE_OPTIONS.reset(token)


@contextlib.contextmanager
def write_mode_context(mode: str) -> Iterator[None]:
    token = _WRITE_MODE.set(mode)
    try:
        yield
    finally:
        _WRITE_MODE.reset(token)


def current_write_mode() -> str | None:
    """Active module-level write mode, or None for direct Writer.write calls."""
    return _WRITE_MODE.get()


def source_path(value: str | Path) -> Path:
    """Resolve a source record URI using the active writer dataset session."""
    return to_dataset_path(value, _SOURCE_STORAGE_OPTIONS.get() or {})


def ensure_directory(path: Path) -> None:
    """Create a local directory; object-store prefixes need no materialization."""
    if not is_remote_path(path):
        path.mkdir(parents=True, exist_ok=True)


def resolve_path(base: Path, value: str | Path | None) -> Path:
    """Resolve a format-declared path without losing its configured backend."""
    if value is None:
        return base
    if not isinstance(value, str):
        if getattr(value, "protocol", None):
            return value
        text = value.as_posix()
    else:
        text = value.strip()
    if "://" in text:
        return to_dataset_path(text, storage_options_for(base))
    posix = PurePosixPath(text.replace("\\", "/"))
    if posix.is_absolute():
        if is_remote_path(base):
            return base.with_segments(*posix.parts)  # type: ignore[attr-defined]
        return Path(text)
    return base.joinpath(*posix.parts)


def resource_key(path: str | Path, storage_options: Mapping[str, Any] | None = None) -> str:
    """Stable comparison key for a local or remote resource."""
    resolved = to_dataset_path(path, storage_options)
    if is_remote_path(resolved):
        return (
            f"{getattr(resolved, 'protocol', '')}://{getattr(resolved, 'path', str(resolved))}"
            f"@fs:{id(getattr(resolved, 'fs', None))}"
        )
    try:
        return str(resolved.resolve(strict=False))
    except OSError:
        return str(resolved)


def is_within(path: str | Path, root: str | Path) -> bool:
    """Backend-aware containment check; remote stores have no symlinks."""
    child = to_dataset_path(path)
    base = to_dataset_path(root)
    try:
        child_remote = is_remote_path(child)
        base_remote = is_remote_path(base)
        if child_remote != base_remote:
            return False
        if base_remote:
            if getattr(child, "protocol", "") != getattr(base, "protocol", ""):
                return False
            child.relative_to(base)
            return True
        return child.resolve(strict=False).is_relative_to(base.resolve(strict=False))
    except (OSError, RuntimeError, ValueError):
        return False


def same_resource(left: str | Path, right: str | Path, storage_options: Mapping[str, Any] | None = None) -> bool:
    """Best-effort resource identity check without requiring both paths locally."""
    lhs = to_dataset_path(left, storage_options)
    rhs = to_dataset_path(right, storage_options)
    if is_remote_path(lhs) or is_remote_path(rhs):
        return (
            is_remote_path(lhs)
            and is_remote_path(rhs)
            and getattr(lhs, "fs", None) is getattr(rhs, "fs", None)
            and getattr(lhs, "path", str(lhs)) == getattr(rhs, "path", str(rhs))
        )
    return resource_key(lhs) == resource_key(rhs)


def copy_resource(
    source: str | Path,
    destination: str | Path,
    *,
    source_storage_options: Mapping[str, Any] | None = None,
    destination_storage_options: Mapping[str, Any] | None = None,
) -> Path:
    """Copy one resource locally, remotely, or across filesystems."""
    source_options = source_storage_options
    if source_options is None and isinstance(source, str):
        source_options = _SOURCE_STORAGE_OPTIONS.get() or {}
    src = to_dataset_path(source, source_options)
    dst = to_dataset_path(destination, destination_storage_options)
    ensure_directory(dst.parent)
    if same_resource(src, dst):
        return dst
    if not is_remote_path(src) and not is_remote_path(dst):
        shutil.copy2(local_open_target(src), local_open_target(dst))
        return dst

    src_fs = getattr(src, "fs", None)
    dst_fs = getattr(dst, "fs", None)
    if src_fs is not None and src_fs is dst_fs:
        try:
            copy_file = getattr(src_fs, "cp_file", None)
            if copy_file is None:
                copy_file = src_fs.copy  # type: ignore[attr-defined]
            copy_file(getattr(src, "path", str(src)), getattr(dst, "path", str(dst)))
            return dst
        except (AttributeError, NotImplementedError, OSError):
            pass
    try:
        with src.open("rb") as source_stream, dst.open("wb") as destination_stream:
            shutil.copyfileobj(source_stream, destination_stream, length=_COPY_BUFFER_SIZE)
    except Exception:
        with contextlib.suppress(Exception):
            remove_resource(dst, missing_ok=True)
        raise
    return dst


def remove_resource(path: Path, *, missing_ok: bool = False) -> None:
    """Delete one local file or remote object without bulk-delete assumptions."""
    try:
        if is_remote_path(path):
            path.fs.rm_file(path.path)  # type: ignore[attr-defined]
        else:
            path.unlink()
    except FileNotFoundError:
        if not missing_ok:
            raise


def remove_tree(path: Path) -> None:
    """Remove a local directory tree or remote object-store prefix."""
    if not is_remote_path(path):
        shutil.rmtree(path)
        return
    filesystem = path.fs  # type: ignore[attr-defined]
    names = filesystem.find(path.path)  # type: ignore[attr-defined]
    try:
        # Delete exactly the inventoried keys; S3-like backends can batch them.
        filesystem.rm(names)
    except OSError:
        # The pinned Apache-licensed MinIO rejects modern DeleteObjects
        # checksum negotiation, so fall back to bounded individual deletes.
        def delete_one(name: str) -> None:
            with contextlib.suppress(FileNotFoundError):
                filesystem.rm_file(name)

        with ThreadPoolExecutor(max_workers=min(16, len(names) or 1)) as executor:
            for start in range(0, len(names), 256):
                list(executor.map(delete_one, names[start : start + 256]))

    # Object stores have virtual directories; avoid one metadata request per
    # ancestor. memory:// alone retains explicit directory nodes in our
    # supported remote backends.
    if getattr(path, "protocol", "") != "memory":
        return
    root = PurePosixPath(path.path)  # type: ignore[attr-defined]
    directories: set[PurePosixPath] = {root}
    for name in names:
        parent = PurePosixPath(name).parent
        while parent != root.parent and parent.is_relative_to(root):
            directories.add(parent)
            parent = parent.parent
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        with contextlib.suppress(FileNotFoundError, OSError):
            filesystem.rmdir(str(directory))


def list_files(directory: Path, *, recursive: bool = False) -> list[Path]:
    """List files while reusing detailed object-store listings."""
    if not is_remote_path(directory):
        candidates = directory.rglob("*") if recursive else directory.iterdir()
        return sorted(path for path in candidates if path.is_file())
    try:
        details = directory.fs.find(  # type: ignore[attr-defined]
            directory.path,  # type: ignore[attr-defined]
            maxdepth=None if recursive else 1,
            withdirs=False,
            detail=True,
        )
        names = details.keys() if isinstance(details, dict) else details
        return sorted(directory.with_segments(name) for name in names)  # type: ignore[attr-defined]
    except (AttributeError, NotImplementedError, OSError):
        candidates = directory.rglob("*") if recursive else directory.iterdir()
        return sorted(path for path in candidates if path.is_file())


def resource_size(path: str | Path, storage_options: Mapping[str, Any] | None = None) -> int | None:
    """Return a resource's size through pathlib/fsspec metadata."""
    resolved = to_dataset_path(path, storage_options)
    try:
        return int(resolved.stat().st_size)
    except (OSError, TypeError, ValueError):
        return None


def read_resource_prefix(path: str | Path, size: int, storage_options: Mapping[str, Any] | None = None) -> bytes:
    """Read at most ``size`` leading bytes without remote read-ahead."""
    if size < 0:
        raise ValueError(f"prefix size must be non-negative, got {size}")
    resolved = to_dataset_path(path, storage_options)
    if not is_remote_path(resolved):
        with resolved.open("rb") as stream:
            return stream.read(size)
    return bytes(resolved.fs.cat_file(resolved.path, start=0, end=size))  # type: ignore[attr-defined]


def write_cv_image(path: Path, frame: np.ndarray, extension: str) -> None:
    """Write an OpenCV image locally or as encoded bytes remotely."""
    import cv2  # type: ignore[import-untyped]

    ensure_directory(path.parent)
    if not is_remote_path(path):
        if not cv2.imwrite(local_open_target(path), frame):
            raise OSError(f"OpenCV failed to write image: {path}")
        return
    ok, encoded = cv2.imencode(extension, frame)
    if not ok:
        raise OSError(f"OpenCV failed to encode image: {path}")
    path.write_bytes(encoded.tobytes())


def probe_video(path: str | Path) -> dict[str, Any]:
    """Collect portable video properties with local OpenCV and remote PyAV."""
    resolved = to_dataset_path(path)
    if not is_remote_path(resolved):
        import cv2  # type: ignore[import-untyped]

        cap = cv2.VideoCapture(local_open_target(resolved))
        try:
            if not cap.isOpened():
                return {"opened": False}
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            fourcc_raw = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
            fourcc = "".join(chr((fourcc_raw >> (8 * index)) & 0xFF) for index in range(4)).strip("\x00 ")
            ok, frame = cap.read()
            return {
                "opened": True,
                # OpenCV exposes a container fourcc, not an FFmpeg codec name.
                "codec_name": None,
                "codec_fourcc": fourcc.lower() or None,
                "fps": fps,
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "first_frame_decodable": bool(ok and frame is not None),
            }
        finally:
            cap.release()
    import av  # type: ignore[import-untyped]

    with open_video_source(resolved) as source, av.open(source) as container:
        stream = container.streams.video[0]
        codec_name = (stream.codec_context.name or "").lower()
        codec_fourcc = _codec_tag_to_string(stream.codec_context.codec_tag)
        fps = float(stream.average_rate or stream.base_rate or 0)
        frame_count = int(stream.frames or 0)
        if frame_count <= 0 and container.duration and fps > 0:
            frame_count = max(1, round(float(container.duration / av.time_base) * fps))
        return {
            "opened": True,
            "codec_name": codec_name or None,
            "codec_fourcc": codec_fourcc,
            "fps": fps,
            "frame_count": frame_count,
            "width": int(stream.codec_context.width or 0),
            "height": int(stream.codec_context.height or 0),
            "first_frame_decodable": next(container.decode(video=0), None) is not None,
        }


def _codec_tag_to_string(value: object) -> str | None:
    """Normalize a PyAV codec tag without confusing it with a codec name."""
    if isinstance(value, str):
        return value.strip("\x00 ").lower() or None
    if isinstance(value, int) and value > 0:
        token = "".join(chr((value >> (8 * index)) & 0xFF) for index in range(4))
        return token.strip("\x00 ").lower() or None
    return None


def iter_video_frames(path: str | Path) -> Iterator[np.ndarray]:
    """Yield BGR frames using OpenCV locally and PyAV for remote streams."""
    resolved = source_path(path)
    if not is_remote_path(resolved):
        import cv2  # type: ignore[import-untyped]

        cap = cv2.VideoCapture(local_open_target(resolved))
        if not cap.isOpened():
            return
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
        finally:
            cap.release()
        return
    import av  # type: ignore[import-untyped]

    with open_video_source(resolved) as source, av.open(source) as container:
        for frame in container.decode(video=0):
            yield frame.to_ndarray(format="bgr24")


def _tiff_dimensions(payload: bytes) -> tuple[int, int] | None:
    if len(payload) < 8 or payload[:2] not in {b"II", b"MM"}:
        return None
    endian = "<" if payload[:2] == b"II" else ">"
    if struct.unpack(f"{endian}H", payload[2:4])[0] != 42:
        return None
    offset = struct.unpack(f"{endian}I", payload[4:8])[0]
    if offset + 2 > len(payload):
        return None
    count = struct.unpack(f"{endian}H", payload[offset : offset + 2])[0]
    values: dict[int, int] = {}
    for index in range(min(count, 4096)):
        start = offset + 2 + index * 12
        if start + 12 > len(payload):
            break
        tag, value_type, value_count = struct.unpack(f"{endian}HHI", payload[start : start + 8])
        if tag not in {256, 257} or value_count != 1:
            continue
        if value_type == 3:
            values[tag] = struct.unpack(f"{endian}H", payload[start + 8 : start + 10])[0]
        elif value_type == 4:
            values[tag] = struct.unpack(f"{endian}I", payload[start + 8 : start + 12])[0]
    return (values[256], values[257]) if 256 in values and 257 in values else None


def _webp_dimensions(payload: bytes) -> tuple[int, int] | None:
    if len(payload) < 30 or not (payload.startswith(b"RIFF") and payload[8:12] == b"WEBP"):
        return None
    chunk = payload[12:16]
    if chunk == b"VP8X":
        return 1 + int.from_bytes(payload[24:27], "little"), 1 + int.from_bytes(payload[27:30], "little")
    if chunk == b"VP8L" and payload[20] == 0x2F:
        bits = int.from_bytes(payload[21:25], "little")
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    if chunk == b"VP8 ":
        start = payload.find(b"\x9d\x01\x2a", 20, min(len(payload), 256))
        if start >= 0 and start + 7 <= len(payload):
            width, height = struct.unpack("<HH", payload[start + 3 : start + 7])
            return width & 0x3FFF, height & 0x3FFF
    return None


def _encoded_image_dimensions(payload: bytes) -> tuple[int, int] | None:  # noqa: C901 - bounded format parser
    """Read common encoded dimensions without allocating a pixel buffer."""
    try:
        if payload.startswith(b"\x89PNG\r\n\x1a\n") and payload[12:16] == b"IHDR":
            return struct.unpack(">II", payload[16:24])
        if payload.startswith((b"GIF87a", b"GIF89a")):
            return struct.unpack("<HH", payload[6:10])
        if payload.startswith(b"BM") and len(payload) >= 26:
            width = struct.unpack("<i", payload[18:22])[0]
            return width, abs(struct.unpack("<i", payload[22:26])[0])
        tiff = _tiff_dimensions(payload)
        if tiff is not None:
            return tiff
        webp = _webp_dimensions(payload)
        if webp is not None:
            return webp
        if payload.startswith(b"\xff\xd8"):
            offset = 2
            limit = min(len(payload), 256 * 1024)
            while offset + 4 <= limit:
                if payload[offset] != 0xFF:
                    offset += 1
                    continue
                marker = payload[offset + 1]
                offset += 2
                if marker in {0x01, 0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
                    continue
                length = struct.unpack(">H", payload[offset : offset + 2])[0]
                if length < 2 or offset + length > limit:
                    return None
                if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
                    height, width = struct.unpack(">HH", payload[offset + 3 : offset + 7])
                    return width, height
                offset += length
    except (IndexError, struct.error):
        return None
    return None


def probe_image_dimensions(
    source: bytes | str | Path, storage_options: Mapping[str, Any] | None = None
) -> tuple[int, int] | None:
    """Read encoded dimensions without decoding pixels or unbounded media."""
    if isinstance(source, bytes):
        payload = source[:_INITIAL_ENCODED_IMAGE_HEADER_BYTES]
    else:
        payload = read_resource_prefix(source, _INITIAL_ENCODED_IMAGE_HEADER_BYTES, storage_options)
    dimensions = _positive_dimensions(_encoded_image_dimensions(payload))
    if dimensions is not None:
        return dimensions
    # JPEG marker segments and TIFF IFDs can place dimensions beyond the fixed
    # header. Retry with a bounded prefix; unusual larger headers fall back to
    # the caller's existing decode path.
    if not payload.startswith((b"\xff\xd8", b"II", b"MM")):
        return None
    if isinstance(source, bytes):
        payload = source[:_MAX_ENCODED_IMAGE_HEADER_BYTES]
    else:
        payload = read_resource_prefix(source, _MAX_ENCODED_IMAGE_HEADER_BYTES, storage_options)
    return _positive_dimensions(_encoded_image_dimensions(payload))


def _positive_dimensions(dimensions: tuple[int, int] | None) -> tuple[int, int] | None:
    if dimensions is None or dimensions[0] <= 0 or dimensions[1] <= 0:
        return None
    return dimensions


def decode_image_bytes(payload: bytes, *, color: bool = False) -> np.ndarray | None:
    """Decode bounded encoded bytes after rejecting oversized declared pixels."""
    import cv2  # type: ignore[import-untyped]

    dimensions = _encoded_image_dimensions(payload)
    recognized = payload.startswith(
        (b"\x89PNG\r\n\x1a\n", b"GIF87a", b"GIF89a", b"BM", b"\xff\xd8", b"II", b"MM", b"RIFF")
    )
    if recognized and dimensions is None:
        return None
    if dimensions is not None and dimensions[0] * dimensions[1] > _MAX_DECODED_IMAGE_PIXELS:
        return None
    encoded = np.frombuffer(payload, dtype=np.uint8)
    flags = cv2.IMREAD_COLOR if color else cv2.IMREAD_UNCHANGED
    try:
        decoded = cv2.imdecode(encoded, flags) if encoded.size else None
    except cv2.error:
        return None
    if decoded is not None and decoded.shape[0] * decoded.shape[1] > _MAX_DECODED_IMAGE_PIXELS:
        return None
    return decoded


def decode_image_path(
    path: str | Path,
    storage_options: Mapping[str, Any] | None = None,
    *,
    color: bool = False,
) -> np.ndarray | None:
    """Decode an image as OpenCV BGR, retaining the native local fast path."""
    import cv2  # type: ignore[import-untyped]

    resolved = to_dataset_path(path, storage_options)
    flags = cv2.IMREAD_COLOR if color else cv2.IMREAD_UNCHANGED
    if not is_remote_path(resolved):
        return cv2.imread(local_open_target(resolved), flags)
    try:
        size = resource_size(resolved)
        if size is not None and size > _MAX_REMOTE_ENCODED_IMAGE_BYTES:
            return None
        with resolved.open("rb") as stream:
            payload = stream.read(_MAX_REMOTE_ENCODED_IMAGE_BYTES + 1)
        if len(payload) > _MAX_REMOTE_ENCODED_IMAGE_BYTES:
            return None
        return decode_image_bytes(payload, color=color)
    except (OSError, cv2.error):
        return None


def open_av_container(
    av_module: Any,
    path: str | Path,
    storage_options: Mapping[str, Any] | None = None,
    *,
    block_size: int | None = None,
) -> tuple[Any, Any]:
    """Open a PyAV container and return it with the owned input source."""
    resolved = to_dataset_path(path, storage_options)
    source: Any
    if is_remote_path(resolved):
        open_options: dict[str, Any] = {}
        if block_size is not None:
            open_options["block_size"] = block_size
        source = resolved.open("rb", **open_options)
    else:
        source = local_open_target(resolved)
    try:
        return av_module.open(source), source
    except Exception:
        if hasattr(source, "close"):
            source.close()
        raise


@contextlib.contextmanager
def open_video_source(path: str | Path, storage_options: Mapping[str, Any] | None = None) -> Iterator[str | BinaryIO]:
    """Yield a local filename or remote seekable fsspec stream for PyAV."""
    resolver_options = dict(storage_options or {})
    if "block_size" in resolver_options:
        block_size = resolver_options.pop("block_size")
        resolver_options.pop("default_block_size", None)
    else:
        block_size = resolver_options.pop("default_block_size", 8 * 1024 * 1024)
    resolved = to_dataset_path(path, resolver_options)
    if not is_remote_path(resolved):
        yield local_open_target(resolved)
        return
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
        raise ValueError(f"block_size must be a positive integer, got {block_size!r}")
    with resolved.open("rb", block_size=block_size) as stream:  # type: ignore[call-overload]
        yield stream
