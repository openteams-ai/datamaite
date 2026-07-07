# probe_bench

Developer-only benchmark comparing transports for `datamaite.probe_video()`
against a remote (`http://`) video, used to validate the streaming design
for the cloud object-storage branch. Not shipped in the wheel or sdist --
see "Build exclusion" below.

## What it measures

Four transports, each replicating the probe's real access pattern (open +
metadata + first frame + 10 sampled seeks + mid frame + last frame):

| Transport | What it is |
|---|---|
| `stream_pyav` | **Real shipped code.** `datamaite.probe_video(UPath(url))` -- PyAV decoding over a seekable fsspec file object, fsspec's default block size. |
| `stream_pyav_1mb` | **Real shipped code**, same call with `UPath(url, block_size=1 << 20)` -- the `storage_options` tune recommended for probe-style seek-heavy access. |
| `full_download` | **Emulation** of a rejected download-first design: `fsspec.get_file()` to a temp file, then `probe_video()` on the local copy. |
| `cv2_url` | **Emulation** of a rejected URL-handoff design: `cv2.VideoCapture` directly on the URL, replicating the seek pattern by hand (FFmpeg's http demuxer issues the actual range requests). Requires `opencv-python`; the script guards the import and prints a skip note if it's missing. |

Only `stream_pyav` and `stream_pyav_1mb` exercise datamaite's actual code
path (`src/datamaite/_formats/hmie/video_checks.py::probe_video`). The
other two approaches model designs that were considered and rejected, to
quantify what they would have cost.

## Prerequisites

```bash
poetry install --extras dev --extras aws --extras fmv
```

`fmv` supplies OpenCV + PyAV for decoding; `aws` is what pulls in
`aiohttp` (transitively, via `s3fs`), which fsspec's `http://` filesystem
needs — the `stream_pyav*` and `cv2_url` transports all fetch the benchmark
video over `http://`, so without the `aws` extra fsspec cannot open the URL.

## Running it

1. Generate synthetic videos at a few sizes (noise-heavy frames so the
   codec can't compress the benchmark away -- see the docstring in
   `gen_video.py`):

   ```bash
   python gen_video.py /tmp/video_small.mp4 750    # ~93 MB
   python gen_video.py /tmp/video_medium.mp4 2500   # ~308 MB
   python gen_video.py /tmp/video_large.mp4 5600    # ~692 MB
   ```

2. Start the range server against one of the videos, with a small added
   per-request latency to make wall-clock differences visible on
   loopback (30ms used for the results below):

   ```bash
   python range_server.py /tmp/video_small.mp4 8123 30
   ```

3. Run the benchmark against the running server, once per video size:

   ```bash
   python bench.py http://127.0.0.1:8123 /tmp/video_small.mp4 small
   ```

   Repeat step 2 (pointing at the medium/large file) and step 3 (with
   matching label) for each size. Each run writes `results_<label>.json`
   next to the script and prints a summary table to stdout.

## Caveat: wall-clock time is not bandwidth-representative

The server and client run on loopback in these runs, so wall time mostly
reflects the injected latency and local disk/CPU, not real network
bandwidth or congestion behavior. Treat `wall_s_median` in the results as
indicative only. **Bytes transferred and request counts are the durable,
host-independent metrics** -- they measure what each transport actually
pulls over the wire regardless of network conditions, and that's what the
conclusions below are based on.

## Results (completed run)

Request counts reported by `/stats` include HEAD metadata probes (e.g. a
transport's initial size check), not just range-fetching GETs -- `do_HEAD`
in `range_server.py` increments the same counter as `do_GET`. The bytes and
request figures below were captured before `do_HEAD` counted itself toward
the request tally, so their request counts undercount HEAD probes relative
to a fresh run; the byte totals (the metric the conclusions are based on)
are unaffected and are left as originally measured.

Bytes transferred per approach, as a percentage of the full file size:

| Transport | 93 MB | 308 MB | 692 MB |
|---|---|---|---|
| PyAV streaming, 1 MiB blocks | 12.7 MB (13.7%) | 12.9 MB (4.2%) | 13.3 MB (1.9%) |
| PyAV streaming, default blocks | 97.9 MB (105.7%) | 59.0 MB (19.2%) | 59.4 MB (8.6%) |
| cv2 over URL (emulation) | 49.0 MB (52.9%) | 47.6 MB (15.5%) | 50.1 MB (7.2%) |
| Full download (emulation) | 92.6 MB (100%) | 307.7 MB (100%) | 691.7 MB (100%) |

### Conclusions

1. **Streaming cost is O(frames sampled), not O(file).** Bytes
   transferred for both PyAV streaming rows stay roughly flat as file
   size grows (~13 MB for the 1 MiB-block tune), while full download
   scales linearly with file size by definition. The probe only ever
   reads the container header plus the packets for the frames it
   actually samples.
2. **fsspec's default 5 MiB read-ahead over-fetches on seek-heavy
   access.** The default-block-size row transfers more than the whole
   file on the smallest video (each seek re-triggers a 5 MiB read-ahead
   that mostly gets discarded) and stays well above the 1 MiB-tuned row
   at every size. Pass `storage_options={"block_size": 1 << 20}` (or
   `UPath(url, block_size=1 << 20)`) for probe-style workloads.

## Scan and packaging policy

These are developer-only scripts: they are excluded from built distributions
(the wheel packages only `src/datamaite`; the sdist builds from an explicit
`src` + `tests` allowlist) and from the CI compliance SAST tier alongside
`tests/` (`sast_excluded_paths` in `.gitlab-ci.yml`). They are still linted
by the local toolchain (ruff security rules, bandit hook), so keep code here
to the same hygiene bar as the rest of the repo.
