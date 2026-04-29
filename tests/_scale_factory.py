"""Builders for hand-rolled Scale annotation JSON used in tests.

Many test cases manually construct a Scale annotation dict with the same
``params``/``response`` skeleton and vary only a handful of fields
(frames, label, afr, fps, task_id, status). These helpers centralize
that skeleton so tests read as "one-track annotation with this frame
list" instead of the full nested dict.

This is distinct from ``_hmie_factory.make_annotation_dict``, which
builds a realistic fixture from an ``AnnotationSpec``. Tests that need
to exercise the Scale schema directly (or inject deliberately malformed
data) use these builders instead.
"""

from __future__ import annotations

from typing import Any

# A frame dict: {"key": int, "left": float, "top": float, "height": float, "width": float, ...}
Frame = dict[str, Any]


def default_frame(key: int = 0, *, left: float = 10, top: float = 10, height: float = 20, width: float = 30) -> Frame:
    """Return a single well-formed bbox frame dict."""
    return {"key": key, "left": left, "top": top, "height": height, "width": width}


def default_frames(count: int = 1) -> list[Frame]:
    """Return ``count`` sequential well-formed bbox frames (keys 0..count-1)."""
    return [default_frame(key=i) for i in range(count)]


def one_track_annotation(
    *,
    task_id: str = "test",
    label: str = "car",
    geometry: str = "box",
    frames: list[Frame] | None = None,
    afr: float | None = 5.0,
    fps: float | None = 30.0,
    status: str | None = None,
    track_id: str = "t1",
    extra_tracks: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a Scale annotation dict with a single track.

    Defaults match the most common inline fixture in the test suite:
    one ``car``/``box`` track with one bbox frame at key 0, afr=5, fps=30.
    Pass ``frames`` to override the frame list, or ``extra_tracks`` to
    add more tracks under ``response.annotations``. Pass ``afr=None`` and
    ``fps=None`` together to omit the ``params`` block entirely.
    """
    if frames is None:
        frames = [default_frame()]
    data: dict[str, Any] = {
        "task_id": task_id,
        "response": {
            "annotations": {
                track_id: {"label": label, "geometry": geometry, "frames": frames},
            },
        },
    }
    if afr is not None or fps is not None:
        data["params"] = {"annotation_frame_rate": afr, "videoMetadata": {"video": {"fps": fps}}}
    if status is not None:
        data["status"] = status
    if extra_tracks:
        data["response"]["annotations"].update(extra_tracks)
    return data
