"""Benchmark probe transports against a byte-counting HTTP range server.

Measures four transports, each replicating the probe's access pattern
(open + metadata + first frame + 10 sampled seeks + mid + last frame)
against ``range_server.py``:

  stream_pyav      -- REAL shipped code: datamaite.probe_video() on
                      UPath("http://...") with default fsspec block size.
                      PyAV over a seekable fsspec file object.
  stream_pyav_1mb  -- REAL shipped code, same call but with
                      UPath(url, block_size=1 << 20): the storage_options
                      tune recommended for probe-style seek-heavy access.
  full_download    -- EMULATION of a download-first design: fsspec
                      get_file() to a temp file, then probe_video() on
                      the local temp copy.
  cv2_url          -- EMULATION of a URL-handoff design: cv2.VideoCapture
                      directly on the URL, replicating the probe's seek
                      pattern by hand (FFmpeg's http demuxer does the
                      range requests). Requires opencv-python; skipped
                      with a printed note if it is not installed.

Server /stats gives bytes+requests actually served between /reset calls.
Loopback wall-clock time is NOT bandwidth-representative -- see README.md.
Bytes transferred and request counts are the durable, host-independent
metrics; wall time is recorded for reference only.

Usage:
    python bench.py <base_url> <local_video_path> <label> [reps]
"""

import json
import statistics
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

import fsspec
from upath import UPath

from datamaite._formats.hmie.video_checks import probe_video

try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

BASE = sys.argv[1]  # e.g. http://127.0.0.1:8123
VIDEO_LOCAL = Path(sys.argv[2])
LABEL = sys.argv[3]
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 3
URL = f"{BASE}/video.mp4"


_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def _ctl(cmd: str) -> dict | None:
    # Talks only to the localhost benchmark server started by this same
    # harness -- not user-controlled input. The guard below enforces exactly
    # two things: the scheme is http(s) (never file:// or another local
    # handler) and the hostname is loopback (never a remote or attacker-
    # controlled host) -- what the B310-family scanners audit for.
    url = f"{BASE}/{cmd}"
    split = urllib.parse.urlsplit(url)
    if split.scheme not in ("http", "https") or split.hostname not in _LOOPBACK_HOSTS:
        raise ValueError(f"benchmark server URL must be http(s) and loopback host: {url!r}")
    with urllib.request.urlopen(url, timeout=10) as r:  # noqa: S310  # nosec: B310
        return json.loads(r.read()) if cmd == "stats" else None


def _require_opened(props, findings, approach: str) -> None:
    if not props.opened:
        raise RuntimeError(f"{approach}: probe_video failed to open: {findings}")


def run_stream_pyav() -> None:
    props, findings = probe_video(UPath(URL))
    _require_opened(props, findings, "stream_pyav")


def run_stream_pyav_1mb() -> None:
    # Same shipped code path, tuned via storage_options (block_size=1 MiB) --
    # exactly what a user can pass without any datamaite change.
    props, findings = probe_video(UPath(URL, block_size=1 << 20))
    _require_opened(props, findings, "stream_pyav_1mb")


def run_full_download() -> None:
    fs = fsspec.filesystem("http")
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_name = tmp.name
    try:
        fs.get_file(URL, tmp_name)
        props, findings = probe_video(Path(tmp_name))
        _require_opened(props, findings, "full_download")
    finally:
        Path(tmp_name).unlink(missing_ok=True)


def run_cv2_url() -> None:
    cap = cv2.VideoCapture(URL)
    try:
        if not cap.isOpened():
            raise RuntimeError("cv2_url: VideoCapture failed to open URL")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.read()  # first frame
        # 10 uniform samples from frame 1, then mid + last (probe pattern)
        n = min(10, frame_count - 1)
        step = max((frame_count - 2) // max(n - 1, 1), 1)
        for i in range(n):
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(1 + i * step, frame_count - 1))
            cap.read()
        for idx in (frame_count // 2, frame_count - 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            cap.read()
    finally:
        cap.release()


APPROACHES = {
    "stream_pyav": run_stream_pyav,
    "stream_pyav_1mb": run_stream_pyav_1mb,
    "full_download": run_full_download,
    "cv2_url": run_cv2_url,
}

size = VIDEO_LOCAL.stat().st_size
results = []
for name, fn in APPROACHES.items():
    if name == "cv2_url" and not HAS_CV2:
        print(f"{LABEL:7s} {name:16s} SKIPPED (opencv-python not installed)", flush=True)
        continue
    times, bytes_list, reqs_list = [], [], []
    for _ in range(REPS):
        _ctl("reset")
        t0 = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - t0
        stats = _ctl("stats")
        times.append(elapsed)
        bytes_list.append(stats["bytes"])
        reqs_list.append(stats["requests"])
    results.append(
        {
            "label": LABEL,
            "file_bytes": size,
            "approach": name,
            "wall_s_median": round(statistics.median(times), 3),
            "wall_s_all": [round(t, 3) for t in times],
            "bytes_transferred": int(statistics.median(bytes_list)),
            "requests": int(statistics.median(reqs_list)),
        }
    )
    r = results[-1]
    print(
        f"{LABEL:7s} {name:16s} wall={r['wall_s_median']:7.3f}s "
        f"bytes={r['bytes_transferred']:>12,} ({r['bytes_transferred'] / size:6.1%} of file) "
        f"reqs={r['requests']}",
        flush=True,
    )

out = Path(__file__).parent / f"results_{LABEL}.json"
out.write_text(json.dumps(results, indent=2))
