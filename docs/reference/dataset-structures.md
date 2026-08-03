# Dataset structures — what datamaite accepts, and what it rejects

This page documents **how datamaite behaves when pointed at a directory**:
which on-disk structures load, which raise errors, and what happens when a
structure is arbitrary, unlabeled, compressed, or unrecognized (issue #40).

The guiding policy is **fail loud, never guess silently**: an unrecognized
structure raises an error rather than being coerced into a dataset, and
loose/unlabeled image handling is an explicit, named opt-in.

## How a format is chosen

`load()` / `load_mot()` / `load_od()` / `load_ic()` / `load_vc()` select a
loader in one of two ways:

- **Explicit** — you pass `dataset_format=` (and `task=` where a format serves
  more than one task, e.g. YOLO). This is the normal, recommended path: you
  asked for a format, datamaite uses it.
- **Autodetect** — `load(root, dataset_format=None)` asks every registered
  loader to `sniff()` the root and picks the unique match. Autodetect is the
  only place the arbitrary-structure rules below apply, because it is the only
  place datamaite has to decide a format for you. Not every format implements
  `sniff()`, so a valid root can still fail autodetect — pass
  `dataset_format=` explicitly when it does.

Loaders are **best-effort by contract**: given the right format, malformed or
partial *content* is skipped with a warning, not raised (see
[Architecture](architecture.md)). The rules here govern structure and format
selection, which is a separate concern from content quality.

## Accepted structures

Each supported format expects a specific on-disk layout; the loader docstrings
hold the exact rules.

| Format | Task | Entry point | Expected layout (sketch) |
|---|---|---|---|
| `hmie` | MOT | `load_mot(dataset_format="hmie")` | `<video>/<snippet>/{<labeler>/*.json, seq_mp4/*.mp4}` — Scale JSON + snippet videos under one root |
| `motchallenge` | MOT | `load_mot(dataset_format="motchallenge")` | `<split>/<seq>/{seqinfo.ini, gt/gt.txt, img1/*.jpg}` |
| `visdrone_video` | MOT | `load_mot(dataset_format="visdrone_video")` | `sequences/<seq>/*.jpg` + `annotations/<seq>.txt` |
| `tao` | MOT | `load_mot(dataset_format="tao")` | `annotations/<split>.json` (COCO-style tracks) + `frames/<split>/<video>/*.jpg` |
| `flat_mp4` | MOT | `load_mot(dataset_format="flat_mp4")` | a flat directory of `*.mp4` clips (no annotations) |
| `coco` | OD | `load_od(dataset_format="coco")` | `annotations/instances.json` + image files |
| `yolo` | OD | `load_od(dataset_format="yolo")` | `data.yaml` + `images/<split>/…` + `labels/<split>/…` (absent/empty label file ⇒ background image) |
| `yolo` | IC | `load_ic(dataset_format="yolo")` | `<split>/<class>/*.{jpg,png,…}` folder tree — folder names become class labels |
| `visdrone` | OD / IC | `load_od` / `load_ic(dataset_format="visdrone")` | `images/*.jpg` + `annotations/*.txt` (VisDrone-DET layout) |
| `huggingface_vision` | OD / IC | `load_od` / `load_ic(dataset_format="huggingface_vision")` | Hugging Face ImageFolder / parquet vision dataset |
| `huggingface_video_classification` | VC | `load_vc(dataset_format="huggingface_video_classification")` | VideoFolder: `<split>/<class>/*.mp4` (+ optional metadata) |
| `flat_images` | OD | `load_od(dataset_format="flat_images")` | a flat directory of loose images (`.jpg`/`.png`/`.tif` + aliases) → an unlabeled OD dataset (zero detections, no taxonomy). **Explicit opt-in only; never autodetected.** |

Cloud (fsspec) roots are supported for the `hmie` format only; any other
format with a remote root raises `ValueError`.

## Rejected and error structures

| Scenario | Behavior |
|---|---|
| Path does not exist | `FileNotFoundError` |
| Root is a file — including archives (`.zip`, `.tar`, `.gz`, …) | `NotADirectoryError`. datamaite reads **directories only**; archives are not supported and will not be. Extract the archive yourself and point datamaite at the extracted directory. |
| Unknown / arbitrary folder structure (autodetect) | `ValueError: Could not autodetect…` listing the registered loaders; pass `dataset_format=` explicitly |
| Empty directory (autodetect) | same `ValueError` |
| Empty-but-valid root with an **explicit** format | warning + empty dataset (the documented best-effort contract) |
| Loose images at the root (autodetect) | `ValueError: Could not autodetect…` — to load them, opt in with `dataset_format="flat_images"` |
| Loose images in **named subfolders** (autodetect) | matches the YOLO image-classification layout, so it **loads with folder names as class labels**. Autodetect logs a warning when this happens; if the folder names are not class labels, move the images into a flat directory and load it with `dataset_format="flat_images"` |
| Structure matching more than one format (autodetect) | `ValueError: Ambiguous autodetect…` — pass `dataset_format`/`task` explicitly |

### Note on arbitrary image subfolders

A directory of arbitrary named subfolders containing images is
indistinguishable from a valid YOLO image-classification dataset — that *is*
the YOLO IC layout. datamaite does not try to outsmart this: autodetect will
pick YOLO IC and emit a warning that folder names were used as class labels,
so a mislabeled guess is visible rather than silent. When in doubt, always
pass `dataset_format=` explicitly.
