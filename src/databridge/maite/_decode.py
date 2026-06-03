"""Video decoding for the MAITE adapters.

MAITE's multi-object-tracking input is a ``VideoStream`` (an iterable of
``VideoFrame`` objects carrying decoded ``pixels`` plus ``time_s`` / ``pts`` /
``frame_index``); the object-detection adapter needs single decoded frames.

The default backend is :class:`PyAVDecoder` (libav via PyAV), which yields true
container ``pts`` / ``time_base`` / ``time_s`` rather than fps-derived
estimates. Callers can substitute any object satisfying the :class:`Decoder`
protocol.

``frame_index`` follows MAITE's definition: the zero-based position of a frame
within the *yielded* stream (decode/sampling output order), which is not
necessarily its absolute index in the source video.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DecodedFrame:
    """One decoded frame; satisfies MAITE's ``VideoFrame`` protocol.

    ``pixels`` is a contiguous ``(C, H, W)`` uint8 RGB array. ``time_s`` and
    ``pts`` are the source frame's presentation time (seconds) and timestamp
    (in stream ``time_base`` units); ``frame_index`` is the position within
    the emitted stream.
    """

    pixels: np.ndarray
    time_s: float
    pts: int
    frame_index: int


@dataclass(frozen=True)
class VideoInfo:
    """Video-level facts needed for MAITE datum metadata."""

    width: int
    height: int
    time_base: Fraction
    size_bytes: int


@runtime_checkable
class Decoder(Protocol):
    """Pluggable video-decode backend used by the MAITE adapters."""

    def info(self, video_path: str) -> VideoInfo:
        """Return video-level dimensions, time base, and byte size."""
        ...

    def stream(self, video_path: str, source_indices: Sequence[int] | None) -> Iterable[DecodedFrame]:
        """Lazily yield frames.

        ``source_indices`` selects which source frames to emit, by their
        zero-based decode order in the source; ``None`` emits every frame.
        The returned iterable is re-iterable (each iteration re-opens the
        file), and emitted ``frame_index`` values run ``0..k-1`` in the order
        frames are yielded.
        """
        ...

    def decode_one(self, video_path: str, source_index: int) -> DecodedFrame:
        """Decode and return a single source frame by its decode-order index."""
        ...


class _PyAVStream:
    """Re-iterable lazy stream over selected source frames of one video."""

    def __init__(self, video_path: str, source_indices: Sequence[int] | None) -> None:
        self._path = video_path
        self._selection: set[int] | None = None if source_indices is None else set(source_indices)

    def __iter__(self) -> Iterator[DecodedFrame]:
        import av  # lazy: only needed when frames are actually consumed

        wanted = self._selection
        target = None if wanted is None else len(wanted)
        with av.open(self._path) as container:
            emitted = 0
            for source_index, frame in enumerate(container.decode(video=0)):
                if wanted is not None and source_index not in wanted:
                    continue
                # to_ndarray gives (H, W, C); MAITE wants (C, H, W).
                pixels = np.ascontiguousarray(frame.to_ndarray(format="rgb24").transpose(2, 0, 1))
                yield DecodedFrame(
                    pixels=pixels,
                    time_s=float(frame.time) if frame.time is not None else 0.0,
                    pts=int(frame.pts) if frame.pts is not None else 0,
                    frame_index=emitted,
                )
                emitted += 1
                if target is not None and emitted >= target:
                    break  # decoded every selected frame; no need to walk the tail


class PyAVDecoder:
    """Default :class:`Decoder` backed by PyAV (libav)."""

    def info(self, video_path: str) -> VideoInfo:
        import av

        with av.open(video_path) as container:
            stream = container.streams.video[0]
            codec = stream.codec_context
            time_base = stream.time_base or Fraction(1, 1000)
            return VideoInfo(
                width=int(codec.width),
                height=int(codec.height),
                time_base=Fraction(time_base),
                size_bytes=os.path.getsize(video_path),
            )

    def stream(self, video_path: str, source_indices: Sequence[int] | None) -> Iterable[DecodedFrame]:
        return _PyAVStream(video_path, source_indices)

    def decode_one(self, video_path: str, source_index: int) -> DecodedFrame:
        for frame in self.stream(video_path, [source_index]):
            return frame
        raise IndexError(f"frame {source_index} not found in {video_path}")


_DEFAULT_DECODER = PyAVDecoder()


def default_decoder() -> Decoder:
    """Return the process-wide default PyAV decoder."""
    return _DEFAULT_DECODER


def resolve_video_info(video_path: str, decoder: Decoder, fallback: VideoInfo) -> VideoInfo:
    """Probe ``video_path`` for accurate info, falling back on probe failure.

    PyAV gives the true container ``time_base`` and pixel dimensions. If the
    probe raises (unreadable/odd container), we fall back to ``fallback`` --
    built from the loader's stored metadata plus an fps-derived time base --
    so a single bad video degrades gracefully instead of aborting access.
    """
    try:
        return decoder.info(video_path)
    except Exception as exc:
        logger.warning("Video probe failed for %s (%s); using fallback metadata", video_path, exc)
        return fallback
