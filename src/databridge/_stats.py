"""Summary statistics over a loaded :class:`~databridge.model.BoxTrackDataset`.

Answers "what does this data actually look like?" -- snippet duration, frame
count, boxes per sequence, and fps -- as distributions, not just averages.
Pure-Python (no numpy), so it stays in core and runs without the ``video`` or
``maite`` extras.

Note on accuracy: ``duration``/``num_frames`` come from the Scale annotation
metadata when present, else from a frame-index *estimate* (max annotated frame
+ 1), which understates true length. Load with ``require_video=True`` for the
real per-file frame counts. The numbers describe the ~10s *snippets* paired by
discovery, not the hours-long full-length source.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from databridge.model import BoxTrackDataset


def _percentiles(values: list[float]) -> dict[str, float] | None:
    """Return count/min/p50/p90/p99/max/mean for ``values`` (None if empty).

    Uses linear interpolation between order statistics (the common
    "type 7"/numpy default percentile), implemented in pure Python.
    """
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)

    def pct(p: float) -> float:
        if n == 1:
            return ordered[0]
        rank = (n - 1) * p
        low = int(rank)
        high = min(low + 1, n - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)

    return {
        "count": n,
        "min": ordered[0],
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p99": pct(0.99),
        "max": ordered[-1],
        "mean": sum(ordered) / n,
    }


def dataset_stats(ds: BoxTrackDataset) -> dict[str, Any]:
    """Compute summary statistics over a loaded dataset.

    Returns a plain dict (JSON-serializable) with overall counts and
    percentile distributions for snippet duration (seconds), frame count,
    boxes per sequence, and fps. Distribution entries are ``None`` when no
    sequence carries that value.
    """
    sequences = ds.sequences
    durations = [s.duration for s in sequences if s.duration is not None and s.duration > 0]
    frame_counts = [float(s.num_frames) for s in sequences if s.num_frames is not None and s.num_frames > 0]
    fps_values = [s.fps for s in sequences if s.fps and s.fps > 0]
    box_counts = [float(len(s.boxes)) for s in sequences]

    return {
        "sequences": len(sequences),
        "sequences_with_video": sum(1 for s in sequences if s.video_path is not None),
        "sequences_with_duration": len(durations),
        "categories": len(ds.categories),
        "total_boxes": ds.num_boxes,
        "duration_s": _percentiles(durations),
        "num_frames": _percentiles(frame_counts),
        "boxes_per_sequence": _percentiles(box_counts),
        "fps": _percentiles(fps_values),
    }


_DISTRIBUTIONS = (
    ("duration_s", "Duration (s)", "{:.2f}"),
    ("num_frames", "Frame count", "{:.0f}"),
    ("boxes_per_sequence", "Boxes/seq", "{:.0f}"),
    ("fps", "FPS", "{:.2f}"),
)


def format_stats(stats: dict[str, Any], *, root: str | None = None) -> str:
    """Render :func:`dataset_stats` output as a human-readable table."""
    lines: list[str] = [""]
    if root:
        lines.append(f"  {root}")
        lines.append("  " + "=" * 68)
    lines.append(
        f"  {stats['sequences']:,} sequences "
        f"({stats['sequences_with_video']:,} with video, "
        f"{stats['sequences_with_duration']:,} with duration metadata)"
    )
    lines.append(f"  {stats['categories']:,} categories · {stats['total_boxes']:,} boxes total")
    lines.append("")
    header = f"  {'metric':<14s}{'count':>8s}{'min':>10s}{'p50':>10s}{'p90':>10s}{'p99':>10s}{'max':>10s}{'mean':>10s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for key, label, fmt in _DISTRIBUTIONS:
        dist = stats.get(key)
        if dist is None:
            lines.append(f"  {label:<14s}{'(no data)':>8s}")
            continue
        lines.append(
            f"  {label:<14s}{dist['count']:>8,}"
            f"{fmt.format(dist['min']):>10s}{fmt.format(dist['p50']):>10s}"
            f"{fmt.format(dist['p90']):>10s}{fmt.format(dist['p99']):>10s}"
            f"{fmt.format(dist['max']):>10s}{fmt.format(dist['mean']):>10s}"
        )
    lines.append("")
    return "\n".join(lines)
