"""Generate a synthetic noise-heavy mp4 for probe-transport benchmarking.

Every frame is random noise with a thin gradient overlay for realism. A
codec cannot compress noise away, so the encoded file stays large -- this
is what makes the full-download vs. ranged-read difference in
``bench.py`` show up at all. A video of mostly-static frames would
compress to a few hundred KB regardless of frame count, defeating the
benchmark.

Usage:
    python gen_video.py <out_path> [n_frames]
"""

import os
import sys

import cv2
import numpy as np

OUT = sys.argv[1]
N_FRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
WIDTH, HEIGHT = 640, 480
FPS = 30.0

rng = np.random.default_rng(42)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUT, fourcc, FPS, (WIDTH, HEIGHT))
try:
    for i in range(N_FRAMES):
        noise = rng.integers(0, 256, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)
        gradient = np.linspace(0, 255, WIDTH, dtype=np.uint8)
        row = np.roll(gradient, i * 5)
        overlay = np.tile(row, (HEIGHT, 1))
        frame = noise.copy()
        frame[:, :, 0] = overlay  # keep some structure so decode is realistic
        writer.write(frame)
finally:
    writer.release()

print(f"wrote {OUT}: {os.path.getsize(OUT) / 1e6:.1f} MB, {N_FRAMES} frames @ {FPS}")
