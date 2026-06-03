#!/usr/bin/env python
"""Profile the databridge MAITE MOT surface: time, peak RSS, and file opens.

Why subprocesses: peak RSS (``resource.ru_maxrss``) is monotonic over a process
lifetime, so each (policy, num_frames) config runs in its own child to isolate
its peak. We also patch ``av.open`` in the child to count how many times each
run touches the video file -- this shows the MOT path opens once per video
(stream pass) plus one probe, regardless of length.

Run:
    poetry run python benchmarks/profile_maite_adapters.py
"""

from __future__ import annotations

import json
import os
import subprocess  # nosec B404 - trusted self-reexec for isolated measurement, no external input
import sys
import tempfile
import time
from pathlib import Path

# (policy, num_frames). The dataset IS a MAITE MOT dataset; we index it directly.
CONFIGS = [
    ("annotated", 300),  # realistic 10s snippet, default policy
    ("all", 100),
    ("all", 200),
    ("all", 400),
]
BOX_STRIDE = 6  # a labeled box every 6 frames (~5 fps labels over 30 fps video)
WIDTH, HEIGHT = 64, 48
WIDGET = "http://example.com/ontology/a/widget"


def _make_video(path: Path, num_frames: int) -> None:
    import cv2  # type: ignore[import-untyped]
    import numpy as np

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (WIDTH, HEIGHT))
    try:
        for i in range(num_frames):
            row = np.roll(np.linspace(0, 255, WIDTH, dtype=np.uint8), i)
            frame = np.stack([np.tile(row, (HEIGHT, 1))] * 3, axis=-1)
            writer.write(frame)
    finally:
        writer.release()


def _run_child(policy: str, video_path: str, num_frames: int, stride: int) -> None:
    import resource

    import av

    opens = {"n": 0}
    _orig_open = av.open

    def _counting_open(*a, **k):  # type: ignore[no-untyped-def]
        opens["n"] += 1
        return _orig_open(*a, **k)

    av.open = _counting_open  # type: ignore[assignment]

    from databridge.model import BoxAnnotation, BoxTrackDataset, VideoSequence

    boxes = [
        BoxAnnotation(
            track_uuid="u",
            track_id=0,
            category_id=1,
            category_uri=WIDGET,
            category_name="widget",
            bbox=(1, 2, 10, 20),
            attributes={},
            frame_index=i,
            timestamp=None,
            keyframe_type="start",
            is_inferred=False,
        )
        for i in range(0, num_frames, stride)
    ]
    seq = VideoSequence(
        video_id=0,
        video_path=video_path,
        fps=30.0,
        num_frames=num_frames,
        duration=num_frames / 30.0,
        annotation_path="a.json",
        width=WIDTH,
        height=HEIGHT,
        size_bytes=os.path.getsize(video_path),
        boxes=boxes,
        num_frames_exact=True,  # the video really has num_frames frames -> "all" may trust it
    )
    ds = BoxTrackDataset(sequences=[seq], categories={WIDGET: 1}).with_mot_options(empty_frame_policy=policy)

    start = time.perf_counter()
    items = 0
    for i in range(len(ds)):
        stream, _target, _meta = ds[i]
        for frame in stream:  # consume the lazy stream
            _ = frame.pixels.shape
            items += 1
    elapsed = time.perf_counter() - start

    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    peak_rss_mb = max_rss / (1024 * 1024) if sys.platform == "darwin" else max_rss / 1024
    print(json.dumps({"items": items, "time_s": elapsed, "peak_rss_mb": peak_rss_mb, "opens": opens["n"]}))


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        videos = {}
        for num_frames in sorted({n for _, n in CONFIGS}):
            path = tmpdir / f"v{num_frames}.mp4"
            _make_video(path, num_frames)
            videos[num_frames] = str(path)

        print(f"{'policy':<11}{'frames':>7}{'items':>7}{'time_s':>10}{'peak_RSS_MB':>13}{'av.opens':>10}")
        print("-" * 58)
        for policy, num_frames in CONFIGS:
            proc = subprocess.run(  # noqa: S603  # nosec B603 - fixed argv, sys.executable, no shell
                [
                    sys.executable,
                    __file__,
                    "--child",
                    policy,
                    videos[num_frames],
                    str(num_frames),
                    str(BOX_STRIDE),
                ],
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                print(f"{policy:<11}{num_frames:>7}  FAILED: {proc.stderr.strip().splitlines()[-1:]}")
                continue
            d = json.loads(proc.stdout.strip().splitlines()[-1])
            print(
                f"{policy:<11}{num_frames:>7}{d['items']:>7}"
                f"{d['time_s']:>10.3f}{d['peak_rss_mb']:>13.1f}{d['opens']:>10}"
            )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        _run_child(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
    else:
        main()
