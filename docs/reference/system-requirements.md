# System requirements

DataMAITE is distributed as a Python library. There is no separate DataMAITE
container image.

## Software

DataMAITE requires Python 3.10, 3.11, 3.12, 3.13, or 3.14. The installer pulls
in every dependency. Features that decode pixels need an extra: `fmv` for video,
`od` and `ic` for still images, and `maite` for MAITE protocol support. See
[Installation](../getting-started/installation.md). Cloud storage needs `aws`,
`gcs`, `azure`, or `cloud`. See
[Cloud object storage](../getting-started/cloud-storage.md).

## Hardware

| Resource | Minimum                     | Recommended                                   |
| -------- | --------------------------- | --------------------------------------------- |
| CPU      | 1 core                      | 4 or more cores for HMIE validation           |
| Memory   | 4 GB                        | 8 GB or more                                  |
| Disk     | 500 MB free, plus your data | Room for a full copy of a dataset you convert |
| GPU      | None                        | None. DataMAITE does not use a GPU            |

- **Memory** grows with the number of annotations, not the size of the media.
  Each loaded dataset keeps its annotations and per-sample records in memory,
  while images and video frames are decoded only when you access an item. HMIE
  validation runs one worker process per CPU core by default, and each worker
  adds about 300 MB with 1080p video, so 12 cores need about 4 GB. Lower it
  with `workers=` in Python or `--workers N` on the command line. For
  reference, loading a COCO dataset the size of `train2017` (118,000 images,
  860,000 boxes) peaks at about 1.4 GB, and converting it at about 1.9 GB.
- **Disk** holds the Python environment, your data, and anything you write.
  A core install is about 70 MB. With every extra (`all`, `maite`, and `cloud`)
  it is about 450 MB. `write()` and `convert()` copy media into the
  destination, so converting a dataset needs free space roughly equal to its
  size. Validation results are cached in `~/.cache/datamaite/validation.db`.
- **A GPU is not used.** All decoding and validation run on the CPU.

## Supported architectures

Verified on Linux x86_64 with Python 3.10 to 3.14: every test job in CI runs
there. The core library is pure Python, so it installs anywhere CPython 3.10 to
3.14 runs. `od` and `ic` depend on OpenCV. `fmv` depends on OpenCV and PyAV.
`maite` depends on PyAV. PyPI publishes OpenCV wheels for Linux (x86_64,
aarch64), macOS (x86_64, arm64), and Windows x86_64. No OpenCV wheel exists for
Windows on arm64, so `od`, `ic`, `fmv`, and `all` do not install there.
Platforms other than Linux x86_64 are not tested.

## Internet access

- **Installation** needs PyPI (or a mirror).
- **The tutorials** need the
  [example datasets](https://gitlab.jatic.net/jatic/orchestration-interoperability/datamaite-example-datasets),
  cloned from `gitlab.jatic.net`.
- **Cloud storage** is used only when you pass an `s3://`, `gs://`, or `az://`
  dataset root. See [Cloud object storage](../getting-started/cloud-storage.md).

Everything else runs offline. DataMAITE makes no other network requests and
sends no usage statistics.
