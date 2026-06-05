"""Flat-folder MP4 video loader (IR-3.3-S-1).

IR-3.3-S-1 requires JATIC products that consume video to accept a flat folder
of ``.mp4`` videos encoded with H.264 or MPEG-2. This loader intentionally
models that narrow contract: it reads only the immediate ``*.mp4`` children of
``root`` (no recursive discovery, no annotations), probes each video via the
optional OpenCV video stack, filters to supported codecs, and returns one
video-backed :class:`databridge.model.VideoSequence` per accepted file.

Like the other loaders, this is best-effort: malformed/unreadable videos and
unsupported codecs are skipped with warnings rather than aborting the whole
load. Install the video extra to enable probing: ``pip install databridge[video]``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from databridge._types import DatasetFormat
from databridge.loaders import Loader, register_loader
from databridge.model import BoxTrackDataset, VideoSequence

logger = logging.getLogger(__name__)

# OpenCV/FFmpeg fourcc aliases observed for the two IR-3.3-S-1 codecs. MP4
# H.264 commonly appears as avc1; MPEG-2 aliases vary by backend/container.
_H264_FOURCCS = frozenset({"avc1", "avc3", "h264", "x264", "davc"})
_MPEG2_FOURCCS = frozenset({"mpg2", "mpeg", "mp2v", "pim2", "em2v", "m2v1"})
_SUPPORTED_CODEC_LABELS = {"h264": "H.264", "mpeg2": "MPEG-2"}


@dataclass(frozen=True)
class _VideoProbe:
    """Video metadata collected from a single OpenCV probe."""

    opened: bool
    codec: str | None = None
    codec_fourcc: str | None = None
    fps: float = 0.0
    frame_count: int = 0
    width: int = 0
    height: int = 0
    first_frame_decodable: bool = False


@register_loader
class FlatMp4Loader(Loader):
    """Loader for a flat directory of H.264/MPEG-2 ``.mp4`` videos."""

    format = DatasetFormat.FLAT_MP4

    def load(self, root: str | Path, **_: Any) -> BoxTrackDataset:
        """Read immediate ``*.mp4`` children under ``root`` into ``BoxTrackDataset``.

        Parameters
        ----------
        root
            Directory whose immediate children are ``.mp4`` videos. The loader
            does **not** recurse into subdirectories; nested videos are ignored
            by design because IR-3.3-S-1 is the flat-folder standard.

        Returns
        -------
        BoxTrackDataset
            One video-backed sequence per readable H.264/MPEG-2 MP4. The
            dataset has no boxes or categories because this format carries no
            annotations.
        """
        root = Path(root)
        if not root.is_dir():
            logger.warning("Flat MP4 root is not a directory: %s", root)
            return BoxTrackDataset(sequences=(), categories={})

        videos = _flat_mp4_files(root)
        if not videos:
            logger.warning("No immediate .mp4 files found in flat MP4 root: %s", root)
            return BoxTrackDataset(sequences=(), categories={})

        sequences: list[VideoSequence] = []
        for video_path in videos:
            seq = _load_video(video_path, next_video_id=len(sequences))
            if seq is not None:
                sequences.append(seq)

        logger.info("Loaded %d flat MP4 video(s) from %s", len(sequences), root)
        return BoxTrackDataset(sequences=tuple(sequences), categories={})


def load_flat_mp4(root: str | Path) -> BoxTrackDataset:
    """Load a flat folder of H.264/MPEG-2 ``.mp4`` videos.

    Equivalent to ``databridge.load(root, dataset_format="flat_mp4")``. See
    :meth:`FlatMp4Loader.load` for semantics.
    """
    return FlatMp4Loader().load(root)


def _flat_mp4_files(root: Path) -> list[Path]:
    """Return immediate MP4 files in deterministic order; never recurse."""
    try:
        return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() == ".mp4")
    except OSError as exc:
        logger.warning("Could not list flat MP4 root %s: %s", root, exc)
        return []


def _load_video(video_path: Path, *, next_video_id: int) -> VideoSequence | None:
    """Probe one video and build a sequence, or skip it with a warning."""
    probe = _probe_mp4_video(video_path)
    if not _probe_is_usable(video_path, probe):
        return None

    # _probe_is_usable narrows these values by validation; keep fallbacks for
    # the type checker and for future probe implementations.
    codec = probe.codec or "unknown"
    frame_count = probe.frame_count
    fps = probe.fps
    duration = frame_count / fps if fps > 0 else None

    return VideoSequence(
        video_id=next_video_id,
        video_path=str(video_path),
        fps=fps,
        num_frames=frame_count,
        duration=duration,
        annotation_path=str(video_path),
        video_meta={
            "format": "flat_mp4",
            "container": "mp4",
            "filename": video_path.name,
            "source_path": str(video_path),
            "codec": codec,
            "codec_label": _SUPPORTED_CODEC_LABELS.get(codec, codec),
            "codec_fourcc": probe.codec_fourcc,
        },
        boxes=[],
        width=probe.width,
        height=probe.height,
        size_bytes=_file_size(video_path),
        num_frames_exact=True,
    )


def _probe_mp4_video(video_path: Path) -> _VideoProbe:
    """Open ``video_path`` with OpenCV and collect codec/metadata."""
    try:
        import cv2  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("OpenCV not installed; cannot load flat MP4 videos (install databridge[video])")
        return _VideoProbe(opened=False)

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            logger.warning("Skipping flat MP4 video that cannot be opened: %s", video_path)
            return _VideoProbe(opened=False)

        try:
            fps = _finite_float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
            frame_count = int(_finite_float(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0)
            width = int(_finite_float(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0)
            height = int(_finite_float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0)
            codec_fourcc = _fourcc_to_string(cap.get(cv2.CAP_PROP_FOURCC))
            codec = _canonical_codec(codec_fourcc)
            ret, first_frame = cap.read()
            first_frame_decodable = bool(ret and first_frame is not None)
        except Exception as exc:
            logger.warning("Skipping flat MP4 video after probe error %s: %s", video_path, exc)
            return _VideoProbe(opened=False)
    finally:
        cap.release()

    return _VideoProbe(
        opened=True,
        codec=codec,
        codec_fourcc=codec_fourcc,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        first_frame_decodable=first_frame_decodable,
    )


def _probe_is_usable(video_path: Path, probe: _VideoProbe) -> bool:
    """Return True when the probed file satisfies IR-3.3-S-1."""
    if not probe.opened:
        return False
    if probe.codec not in _SUPPORTED_CODEC_LABELS:
        known = ", ".join(_SUPPORTED_CODEC_LABELS.values())
        fourcc = probe.codec_fourcc or "unknown"
        logger.warning(
            "Skipping flat MP4 video with unsupported codec %r (supported: %s): %s",
            fourcc,
            known,
            video_path,
        )
        return False
    if probe.frame_count <= 0:
        logger.warning("Skipping flat MP4 video that reports no frames: %s", video_path)
        return False
    if probe.width <= 0 or probe.height <= 0:
        logger.warning(
            "Skipping flat MP4 video with invalid resolution %sx%s: %s",
            probe.width,
            probe.height,
            video_path,
        )
        return False
    if not probe.first_frame_decodable:
        logger.warning("Skipping flat MP4 video whose first frame cannot be decoded: %s", video_path)
        return False
    return True


def _canonical_codec(fourcc: str | None) -> str | None:
    """Map an OpenCV/FFmpeg fourcc string to the supported codec key."""
    if not fourcc:
        return None
    token = fourcc.strip().lower()
    if token in _H264_FOURCCS:
        return "h264"
    if token in _MPEG2_FOURCCS:
        return "mpeg2"
    return None


def _fourcc_to_string(value: Any) -> str | None:
    """Decode OpenCV's numeric CAP_PROP_FOURCC value into a four-character token."""
    number = _finite_float(value)
    if number is None:
        return None
    raw = int(number)
    if raw <= 0:
        return None
    chars = "".join(chr((raw >> (8 * i)) & 0xFF) for i in range(4))
    # Keep printable non-NUL characters only; OpenCV may return padding.
    token = "".join(ch for ch in chars if ch.isprintable() and ch != "\x00").strip()
    return token.lower() or None


def _finite_float(value: Any) -> float | None:
    """Coerce ``value`` to a finite float, else None."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _file_size(path: Path) -> int | None:
    """Return file size in bytes, or None if unavailable."""
    try:
        return path.stat().st_size
    except OSError:
        return None
