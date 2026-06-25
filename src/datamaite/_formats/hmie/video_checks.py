"""Video integrity checks for HMIE/Scale datasets."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from datamaite._types import Finding, Severity

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoProperties:
    """Cached video metadata read from a single cv2.VideoCapture open.

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

_cv2_silenced = False


def _silence_cv2_logging(cv2) -> None:  # type: ignore[no-untyped-def]
    """Set cv2 log level to SILENT once per process. Idempotent.

    opencv's libavcodec backend writes decoder warnings directly to
    stderr, bypassing Python's logging. At 60K pairs this makes the
    CLI output unreadable. cv2.setLogLevel is process-global so we
    only need to call it once.
    """
    global _cv2_silenced
    if _cv2_silenced:
        return
    # Walk the cv2.utils.logging chain defensively: opencv-python exposes
    # it but some variant builds (headless, custom, older wheels) omit
    # cv2.utils entirely. Silencing is cosmetic; if the API is missing we
    # still mark as silenced so we don't keep re-probing on every call.
    utils = getattr(cv2, "utils", None)
    logging_mod = getattr(utils, "logging", None) if utils is not None else None
    if logging_mod is not None:
        silent = getattr(logging_mod, "LOG_LEVEL_SILENT", None)
        set_log_level = getattr(logging_mod, "setLogLevel", None)
        if silent is not None and set_log_level is not None:
            set_log_level(silent)
    _cv2_silenced = True


def probe_video(video_path: Path) -> tuple[VideoProperties, list[Finding]]:
    """Open a video once and extract cached properties plus integrity findings.

    This is the single-open entry point. Callers that need both integrity
    and consistency checks should use this + pass the returned VideoProperties
    to check_video_annotation_consistency() to avoid a second open.

    Frame-level integrity checks (stuck/black frames, mid-video decode,
    last-frame decode) are performed here while the capture is already open.

    Requires the `video` extra: pip install datamaite[video]
    """
    findings: list[Finding] = []

    try:
        import cv2  # type: ignore[import-untyped]
    except ImportError:
        findings.append(
            Finding(
                severity=Severity.WARNING,
                path=video_path,
                check="video_dependency",
                message="opencv-python-headless not installed, skipping video checks",
            )
        )
        return VideoProperties(path=video_path, opened=False), findings

    # Silence opencv's direct-to-stderr decoder warnings. At 60K pairs
    # with any corrupt videos, stderr becomes unreadable because opencv
    # writes libavcodec warnings past Python's logging. Set the log level
    # to SILENT on the first call per process; cv2 stores this globally.
    _silence_cv2_logging(cv2)

    props = _probe_with_capture(cv2, video_path, findings)
    if not props.opened:
        return props, findings

    _append_metadata_findings(props, findings)
    return props, findings


def _probe_with_capture(cv2, video_path: Path, findings: list[Finding]) -> VideoProperties:  # type: ignore[no-untyped-def]
    """Open the video with cv2.VideoCapture and collect raw properties.

    Returns an un-opened VideoProperties when the capture cannot open
    or throws while reading metadata. Frame-sample and mid/last-frame
    integrity checks run here while the capture is still open.
    """
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
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
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            ret, first_frame = cap.read()
            first_frame_decodable = bool(ret and first_frame is not None)

            # Deeper integrity: sample frames across the video and check
            # mid/last decodability while the capture is still open.
            _check_frame_samples(cv2, cap, video_path, frame_count, findings)
            _check_mid_and_last_frame(cv2, cap, video_path, frame_count, findings)
        except Exception as e:
            # cv2 can raise on malformed containers, ValueError on NaN ints,
            # OSError on flaky network FS, etc. Surface it as a finding
            # instead of letting the worker crash -- a single bad video
            # must not kill a whole 60K-pair run.
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
        cap.release()

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


def _check_frame_samples(cv2, cap, video_path: Path, frame_count: int, findings: list[Finding]) -> None:  # type: ignore[no-untyped-def]
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
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        sampled_count += 1
        if float(np.std(frame[::4, ::4])) < _FLAT_FRAME_STD_THRESHOLD:
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


def _check_mid_and_last_frame(cv2, cap, video_path: Path, frame_count: int, findings: list[Finding]) -> None:  # type: ignore[no-untyped-def]
    """Verify that the middle and last frames of the video can be decoded.

    Catches mid-video corruption that the first-frame check misses --
    HMIE videos commonly decode frame 0 cleanly and die halfway.

    WARNING, not ERROR: cv2.VideoCapture.set(CAP_PROP_POS_FRAMES) seek
    semantics are codec- and build-dependent. With H.264 + B-frame
    reordering, a direct frame-index seek to a non-keyframe can return
    the nearest preceding keyframe or fail the read entirely, even on
    a perfectly valid video. Until we have real CDAO data to measure
    the false-positive rate, these checks are advisory only -- they
    surface a suspicious video without failing its dataset. The
    video_first_frame and video_flat_frames checks still fire at
    ERROR level and catch catastrophic breakage.
    """
    if frame_count <= 1:
        return

    mid_idx = frame_count // 2
    last_idx = frame_count - 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_idx)
    ret, frame = cap.read()
    if not ret or frame is None:
        findings.append(
            Finding(
                severity=Severity.WARNING,
                path=video_path,
                check="video_mid_frame",
                message=f"Cannot decode middle frame (index {mid_idx})",
            )
        )

    cap.set(cv2.CAP_PROP_POS_FRAMES, last_idx)
    ret, frame = cap.read()
    if not ret or frame is None:
        findings.append(
            Finding(
                severity=Severity.WARNING,
                path=video_path,
                check="video_last_frame",
                message=f"Cannot decode last frame (index {last_idx})",
            )
        )
