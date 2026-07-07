"""Video integrity checks for HMIE/Scale datasets."""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datamaite._types import Finding, Severity
from datamaite._upath import is_remote_path, local_open_target

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoProperties:
    """Cached video metadata read from a single av.open() container.

    Avoids the double-open cost when both integrity and consistency
    checks need the same video. At 60K datasets this saves ~100 min.
    """

    path: Path
    opened: bool
    fps: float = 0.0
    frame_count: int = 0
    width: int = 0
    height: int = 0
    first_frame_decodable: bool = False


# Frame sampling parameters -- tuned for cheap FMV quality probes.
# Each sampled frame is ~20ms of hardware decode, so 10 samples ~= 200ms
# which is small relative to the initial open + first-frame read cost.
_FRAME_SAMPLE_COUNT = 10
# Variance floor below which a frame is considered flat (stuck/black/white).
# Numpy std on uint8 pixel values across all channels.
#
# Tuning note: 1.0 only catches truly solid-color frames (stuck codec
# output, dead sensor, corrupted decoder state). Real FMV in IR or
# low-light conditions can legitimately have std around 3-8 on uniform
# sky/terrain -- a stricter threshold like 2.0 was producing ~3-5%
# false positives on real CDAO data.
_FLAT_FRAME_STD_THRESHOLD = 1.0
# If more than this fraction of sampled frames are flat, the video is bad.
# Raised from 0.3 to 0.5 because short title cards, IR warmup sequences,
# and intentional fade-to-black intros are common in real FMV and should
# not fail an otherwise-valid snippet.
_FLAT_FRAME_FAIL_RATIO = 0.5

# Block size handed to fsspec when opening a remote video for probing.
# fsspec's default 5 MiB read-ahead over-fetches badly on the probe's
# seek-heavy access pattern (measured >100% of file size on ~90 MB clips;
# see tools/probe_bench/README.md); a 1 MiB block bounds a whole probe to
# ~13 MB regardless of file size.
_REMOTE_READ_BLOCK_SIZE = 1 << 20

_av_silenced = False


def _silence_av_logging(av) -> None:  # type: ignore[no-untyped-def]
    """Set PyAV/FFmpeg log level to PANIC once per process. Idempotent.

    FFmpeg writes decoder warnings straight to stderr. At tens of
    thousands of pairs with any corrupt videos, that makes CLI output
    unreadable; the findings carry the signal instead.

    ``av.logging.PANIC`` is the highest FFmpeg log-level threshold (only
    unrecoverable-crash messages pass it), used here purely to suppress
    decoder warning noise. The setting is process-wide, global FFmpeg/PyAV
    state, not scoped to this probe -- it also silences log output for any
    other PyAV consumer running in the same process (e.g. a notebook cell
    or another library opening containers alongside datamaite).
    """
    global _av_silenced
    if _av_silenced:
        return
    logging_mod = getattr(av, "logging", None)
    set_level = getattr(logging_mod, "set_level", None) if logging_mod is not None else None
    panic = getattr(logging_mod, "PANIC", None) if logging_mod is not None else None
    if set_level is not None and panic is not None:
        set_level(panic)
    _av_silenced = True


def _av_source(video_path: Path) -> Any:
    """What to hand :func:`av.open` for ``video_path``.

    Local paths open by plain filesystem string. Remote paths open as a
    seekable fsspec file object with a 1 MiB read-ahead block
    (``_REMOTE_READ_BLOCK_SIZE``), so PyAV's demuxer fetches only the byte
    ranges the probe actually reads (container header plus the sampled
    frames' packets) -- no full-file download, no presigned URLs, and
    identical behavior on every backend. The caller owns closing a
    returned file object.
    """
    if is_remote_path(video_path):
        return video_path.open("rb", block_size=_REMOTE_READ_BLOCK_SIZE)  # type: ignore[union-attr]
    return local_open_target(video_path)


def probe_video(video_path: Path) -> tuple[VideoProperties, list[Finding]]:
    """Open a video once and extract cached properties plus integrity findings.

    This is the single-open entry point. Callers that need both integrity
    and consistency checks should use this + pass the returned VideoProperties
    to check_video_annotation_consistency() to avoid a second open.

    Frame-level integrity checks (stuck/black frames, mid-video decode,
    last-frame decode) are performed here while the capture is already open.

    ``video_path`` may be a remote ``UPath`` (``s3://``, ``gs://``,
    ``az://``): the probe streams the container through a seekable fsspec
    file object, transferring only the ranges it reads in 1 MiB blocks
    (``_REMOTE_READ_BLOCK_SIZE``). Findings always report the logical
    dataset path.

    Requires the ``fmv`` extra: pip install datamaite[fmv]
    """
    findings: list[Finding] = []

    try:
        import av  # type: ignore[import-untyped]
    except ImportError:
        findings.append(
            Finding(
                severity=Severity.WARNING,
                path=video_path,
                check="video_dependency",
                message="av (PyAV) not installed; install datamaite[fmv] to run video checks",
            )
        )
        return VideoProperties(path=video_path, opened=False), findings

    _silence_av_logging(av)

    props = _probe_with_container(av, video_path, findings)
    if not props.opened:
        return props, findings

    _append_metadata_findings(props, findings)
    return props, findings


def _probe_with_container(av, video_path: Path, findings: list[Finding]) -> VideoProperties:  # type: ignore[no-untyped-def]
    """Open the video with av.open and collect raw properties.

    Returns an un-opened VideoProperties when the container cannot open or
    throws while reading metadata. Frame-sample and mid/last-frame
    integrity checks run here while the container is still open.
    """
    source: Any = None
    container = None
    try:
        source = _av_source(video_path)
        container = av.open(source)
    except Exception:
        # Bad bytes, missing remote object, or transport failure: all
        # collapse to "cannot open", mirroring the old capture semantics.
        if source is not None and hasattr(source, "close"):
            source.close()
        findings.append(
            Finding(
                severity=Severity.ERROR,
                path=video_path,
                check="video_open",
                message="Cannot open video file",
            )
        )
        return VideoProperties(path=video_path, opened=False)

    try:
        stream = next((s for s in container.streams if s.type == "video"), None)
        if stream is None:
            findings.append(
                Finding(
                    severity=Severity.ERROR,
                    path=video_path,
                    check="video_open",
                    message="Cannot open video file",
                )
            )
            return VideoProperties(path=video_path, opened=False)

        fps = float(stream.average_rate) if stream.average_rate else 0.0
        frame_count = int(stream.frames or 0)
        if frame_count <= 0 and stream.duration is not None and stream.time_base is not None and fps > 0:
            # Some containers omit the frame count; derive it from duration.
            frame_count = int(float(stream.duration * stream.time_base) * fps)
        width = int(stream.codec_context.width or 0)
        height = int(stream.codec_context.height or 0)

        first_frame = next(container.decode(stream), None)
        first_frame_decodable = first_frame is not None

        _check_frame_samples(container, stream, video_path, frame_count, fps, findings)
        _check_mid_and_last_frame(container, stream, video_path, frame_count, fps, findings)
    except Exception as e:
        # PyAV raises on malformed containers, OSError on flaky transports,
        # etc. Surface it as a finding instead of letting the worker crash
        # -- a single bad video must not kill a whole 60K-pair run.
        findings.append(
            Finding(
                severity=Severity.ERROR,
                path=video_path,
                check="video_probe_error",
                message=f"Error probing video: {type(e).__name__}: {e}",
            )
        )
        return VideoProperties(path=video_path, opened=False)
    finally:
        with contextlib.suppress(Exception):
            container.close()
        if hasattr(source, "close"):
            source.close()

    return VideoProperties(
        path=video_path,
        opened=True,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        first_frame_decodable=first_frame_decodable,
    )


def _append_metadata_findings(props: VideoProperties, findings: list[Finding]) -> None:
    """Emit findings for suspicious video-level metadata values."""
    if props.frame_count <= 0:
        findings.append(
            Finding(
                severity=Severity.ERROR,
                path=props.path,
                check="video_frame_count",
                message=f"Video reports {props.frame_count} frames",
            )
        )

    if props.fps <= 0 or props.fps > 240:
        findings.append(
            Finding(
                severity=Severity.WARNING,
                path=props.path,
                check="video_fps",
                message=f"Unusual FPS: {props.fps}",
            )
        )

    if props.width <= 0 or props.height <= 0:
        findings.append(
            Finding(
                severity=Severity.ERROR,
                path=props.path,
                check="video_resolution",
                message=f"Invalid resolution: {props.width}x{props.height}",
            )
        )

    if not props.first_frame_decodable:
        findings.append(
            Finding(
                severity=Severity.ERROR,
                path=props.path,
                check="video_first_frame",
                message="Cannot decode first frame",
            )
        )


def _decode_frame_at(container, stream, frame_index: int, fps: float):  # type: ignore[no-untyped-def]
    """Seek near ``frame_index`` and decode the next frame, or None.

    PyAV seeks land on the nearest preceding keyframe, not the exact
    frame index; the first decoded frame after the seek stands in for the
    requested index. That's inherent to keyframe-based seeking over the
    same FFmpeg decoder underneath (codec-dependent granularity), which
    is why the mid/last checks are WARNING-level.
    """
    if fps <= 0 or stream.time_base is None:
        return None
    try:
        container.seek(int(frame_index / fps / stream.time_base), stream=stream)
        return next(container.decode(stream), None)
    except Exception:
        return None


def _check_frame_samples(
    container,  # type: ignore[no-untyped-def]
    stream,  # type: ignore[no-untyped-def]
    video_path: Path,
    frame_count: int,
    fps: float,
    findings: list[Finding],
) -> None:
    """Sample frames across the video and flag stuck/black/flat content.

    For each sampled frame compute numpy std across all pixels. A very
    low std means the frame is solid color (black, white, or stuck).
    If more than _FLAT_FRAME_FAIL_RATIO of samples are flat, emit an
    ERROR finding.
    """
    if frame_count <= 1:
        return

    import numpy as np

    # Sample uniformly across the video, but start from frame 1 rather than
    # frame 0. Frame 0 is often a title card, fade-in, or black leader -- a
    # legitimate "flat" frame that shouldn't count toward the failure ratio.
    # Skipping it removes a common source of false positives on real FMV.
    n_samples = min(_FRAME_SAMPLE_COUNT, frame_count - 1)
    if n_samples <= 0:
        return
    start_idx = 1
    end_idx = frame_count - 1
    step = max((end_idx - start_idx) // max(n_samples - 1, 1), 1)
    sample_indices = [min(start_idx + i * step, end_idx) for i in range(n_samples)]

    flat_count = 0
    sampled_count = 0
    for idx in sample_indices:
        frame = _decode_frame_at(container, stream, idx, fps)
        if frame is None:
            continue
        sampled_count += 1
        pixels = frame.to_ndarray(format="bgr24")
        if float(np.std(pixels[::4, ::4])) < _FLAT_FRAME_STD_THRESHOLD:
            flat_count += 1

    if sampled_count == 0:
        return  # other checks will catch an unreadable video

    flat_ratio = flat_count / sampled_count
    if flat_ratio > _FLAT_FRAME_FAIL_RATIO:
        findings.append(
            Finding(
                severity=Severity.ERROR,
                path=video_path,
                check="video_flat_frames",
                message=(
                    f"{flat_count}/{sampled_count} sampled frames are flat "
                    f"(std < {_FLAT_FRAME_STD_THRESHOLD}); video is likely corrupted or stuck"
                ),
            )
        )


def _check_mid_and_last_frame(
    container,  # type: ignore[no-untyped-def]
    stream,  # type: ignore[no-untyped-def]
    video_path: Path,
    frame_count: int,
    fps: float,
    findings: list[Finding],
) -> None:
    """Verify that the middle and last frames of the video can be decoded.

    Catches mid-video corruption that the first-frame check misses --
    HMIE videos commonly decode frame 0 cleanly and die halfway.

    WARNING, not ERROR: PyAV's keyframe-based seek semantics are codec-
    and container-dependent. With H.264 + B-frame reordering, a seek to
    a non-keyframe index lands on the nearest preceding keyframe rather
    than the exact frame -- the same FFmpeg decoder underneath any
    seek-based reader behaves this way, it isn't a PyAV shortcut. Until
    we have real CDAO data to measure the false-positive rate, these
    checks are advisory only -- they surface a suspicious video without
    failing its dataset. The video_first_frame and video_flat_frames
    checks still fire at ERROR level and catch catastrophic breakage.
    """
    if frame_count <= 1:
        return

    mid_idx = frame_count // 2
    last_idx = frame_count - 1

    frame = _decode_frame_at(container, stream, mid_idx, fps)
    if frame is None:
        findings.append(
            Finding(
                severity=Severity.WARNING,
                path=video_path,
                check="video_mid_frame",
                message=f"Cannot decode middle frame (index {mid_idx})",
            )
        )

    frame = _decode_frame_at(container, stream, last_idx, fps)
    if frame is None:
        findings.append(
            Finding(
                severity=Severity.WARNING,
                path=video_path,
                check="video_last_frame",
                message=f"Cannot decode last frame (index {last_idx})",
            )
        )
