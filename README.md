# datamaite

A unified framework for dataset loading, conversion, and quality validation.

## Quick Start

```bash
# Clone and install
git clone https://gitlab.jatic.net/jatic/orchestration-interoperability/datamaite.git
cd datamaite
poetry install --extras dev --extras all

# Validate a dataset
datamaite validate /path/to/dataset

# Validate multiple batches at once
datamaite validate /path/to/batches/

# Verbose output (individual findings)
datamaite -v validate /path/to/dataset

# Save full report to file
datamaite validate /path/to/dataset -o report.txt
```

## Supported formats

datamaite is a bridge: a **loader** reads an input format into a
source-preserving in-memory dataset (`BoxTrackDataset` for MOT/video box tracks;
`ObjectDetectionDataset` for still-image object detection;
`ImageClassificationDataset` for still-image classification;
`VideoClassificationDataset` for video-level labels), a **validator** checks an
HMIE/Scale dataset on disk, and a **writer** serialises supported datasets out to
an output format (`convert` pairs a loader and a writer of the same task for
on-disk → on-disk conversion). HMIE/Scale, flat MP4 video folders, Hugging Face
Video Classification, MOTChallenge, TAO, VisDrone Video, VisDrone still images,
COCO OD, YOLO image classification, and YOLO object detection are implemented
input formats; HMIE/Scale, Hugging Face Video Classification, MOTChallenge,
TAO, VisDrone Video, COCO OD, YOLO image classification, and YOLO object
detection are implemented output formats. Validation is currently **HMIE/Scale only**;
non-HMIE formats load and write but are not validated by `datamaite validate`
yet.

| Format | Load | Validate | Write |
|---|---|---|---|
| HMIE / Scale (FMV) | ✅ | ✅ | ✅ |
| Flat folder MP4 video (H.264 / MPEG-2) | ✅ | — | planned |
| Hugging Face Video Classification | ✅ | — | ✅ |
| MOTChallenge | ✅ | — | ✅ |
| TAO | ✅ | — | ✅ |
| VisDrone Video (VID/MOT) | ✅ | — | ✅ |
| VisDrone still images (OD) | ✅ | — | — |
| VisDrone still images (IC, object crops) | ✅ | — | — |
| YOLO image classification | ✅ | — | ✅ |
| COCO object detection | ✅ | — | ✅ |
| YOLO object detection | ✅ | — | ✅ |

See [docs/architecture.md](docs/architecture.md) for the loader / writer design
and how to add a new loader or writer.

## Installation extras

Core install is deliberately lean:

```bash
pip install datamaite
```

Core installs only direct runtime dependencies needed for annotation loading,
validation, conversion dispatch, and MAITE target arrays: `pydantic` and
`numpy`. Pixel/media decoding is selected by task extras:

| Install | Adds | Enables |
|---|---|---|
| `datamaite` | `pydantic`, `numpy` | load/convert IRs, HMIE structure/annotation validation, MAITE target objects |
| `datamaite[fmv]` | OpenCV + PyAV | FMV integrity checks, flat MP4 probing, video-backed MOT decode/export |
| `datamaite[od]` | OpenCV | still-image OD pixel decode |
| `datamaite[ic]` | OpenCV | still-image IC pixel decode |
| `datamaite[all]` | union of task extras | all task pixel/media paths |
| `datamaite[maite]` | MAITE + PyAV | optional MAITE package plus MOT-video runtime for interoperability/conformance checks |
| `datamaite[aws]` | `s3fs` | load/validate HMIE from `s3://` roots |
| `datamaite[gcs]` | `gcsfs` | load/validate HMIE from `gs://` roots |
| `datamaite[azure]` | `adlfs` | load/validate HMIE from `az://` roots |
| `datamaite[cloud]` | `s3fs` + `gcsfs` + `adlfs` | all three cloud backends |

`maite` itself is not a core runtime dependency; datamaite datasets conform to
MAITE protocols structurally. The `maite` extra is available for consumers and
conformance tests that want the MAITE package installed alongside the adapters;
it includes PyAV because the MOT MAITE view decodes video-backed sequences.
See [docs/packaging.md](docs/packaging.md) for the dependency contract.

### Cloud object storage

Dataset roots can be cloud URLs — `s3://`, `gs://`, or `az://` — with the
matching extra installed (`datamaite[aws]`, `[gcs]`, `[azure]`, or
`[cloud]` for all three). Cloud roots are supported for **HMIE only**;
other format loaders raise a clear error on a cloud URL. Video integrity
checks over cloud data additionally need the `fmv` extra (e.g.
`datamaite[aws,fmv]`) — without it, video checks are skipped with a
`video_dependency` warning:

```python
import datamaite

result = datamaite.validate("s3://my-bucket/datasets/batch-a")
```

See the cloud storage guide in the docs for credentials and how video
integrity checks work remotely.

## Loading datasets (Python)

Load an HMIE/Scale dataset into the neutral `BoxTrackDataset` model:

```python
from datamaite import load_mot

ds = load_mot("/path/to/dataset")  # HMIE is the default MOT format
print(ds.sequence_count, "sequences,", ds.num_boxes, "boxes")

for seq in ds.iter_sequences():
    for box in seq.boxes:
        print(box.frame_index, box.track_id, box.category_name, box.bbox)
```

> `ds` is also a MAITE dataset (see below), so `len(ds)` / `ds[i]` / `for x in ds`
> are the **MAITE item** view (one per video). Use `ds.sequence_count` /
> `ds.iter_sequences()` / `ds.sequences` for the **record** view shown above.

The lower-level generic dispatching entry point works too (same result):

```python
from datamaite import load

ds = load("/path/to/dataset", dataset_format="hmie")
```

For non-standard layouts (flat annotation/video directories) and true frame
counts probed from the videos:

```python
ds = load_mot(
    "/path/to/dataset",
    dataset_format="hmie",
    annotation_dir="annotations/",
    video_dir="videos/",
    require_video=True,  # needs the `fmv` extra
)
```

Load a flat folder of `.mp4` videos (immediate children only, H.264 or
MPEG-2 codec) as video-backed sequences:

```python
from datamaite import load_mot

ds = load_mot(
    "/path/to/mp4-folder",
    dataset_format="flat_mp4",
)  # requires datamaite[fmv]

for seq in ds.iter_sequences():
    print(seq.video_path, seq.video_meta["codec"], seq.num_frames)
```

The flat MP4 loader does not recurse into subdirectories and carries no
annotations, so `seq.boxes` is empty and `ds.categories == {}`.

Load a Hugging Face VideoFolder-style video classification repository (class
folders, optional `train` / `validation` / `test` splits, or `metadata.csv` /
`metadata.jsonl` with `file_name` and an optional `label` column). Optional
`metadata.parquet` loading is experimental and requires `pyarrow` or `pandas`;
if parquet cannot be read, the loader warns and falls back to folder discovery.

```python
from datamaite import load_vc

# Example layout: train/dog/clip.mp4, train/cat/clip.mp4, test/dog/clip.mp4
# Metadata-file layouts use file_name paths relative to the metadata file.
ds = load_vc("/path/to/hf-video-dataset")

for sample in ds.iter_samples():
    print(sample.video_path, sample.split, sample.label)
```

Video classification labels are video-level metadata, not per-frame boxes, and
MAITE 0.9.5 has no video-classification protocol. The loader returns a
`VideoClassificationDataset` (not a MAITE MOT `BoxTrackDataset`): `len(ds)` /
`ds[i]` / iteration are plain sample-record accessors, `ds.label_names()` maps
stable label IDs to raw labels, and `ds.categories` uses injective percent-encoded
label URIs so labels such as `"a b"`, `"a/b"`, and `"a_b"` remain distinct.

Load a standard MOTChallenge benchmark root (with `train/` and/or `test/`
splits) the same way:

```python
from datamaite import load_mot

ds = load_mot(
    "/path/to/MOT17",
    dataset_format="motchallenge",
    annotation_source="gt",       # or "det"
    include_ignored=False,
    classes={1, 42},              # optional allowlist; omit/None to keep all classes
    class_names={42: "vehicle"},  # optional names for non-standard class IDs
)

for seq in ds.iter_sequences():
    print(seq.frame_dir, seq.frame_filename(0), len(seq.boxes))
```

MOTChallenge is image-sequence based, so loaded sequences have
`video_path=None` and carry their frame image location in `frame_dir`. Use
`seq.frame_filename(frame_index)` or `seq.frame_path(frame_index)` with the
model's 0-based `frame_index`; those helpers handle MOTChallenge's 1-based
image filenames. Standard MOTChallenge class IDs get their canonical names;
unknown/non-standard IDs default to `class_<id>`. For MOT-style datasets with
custom labels, pass `class_names={42: "vehicle"}`. Missing IDs, or an empty
`class_names={}`, still fall back to the built-in names and `class_<id>`.
To optionally probe the first frame image with OpenCV for metadata, pass
`probe_images=True` after installing `datamaite[fmv]`.

Load an official TAO dataset root (COCO-style `annotations/*.json` plus frame
files) similarly:

```python
from datamaite import load_mot

ds = load_mot("/path/to/TAO", dataset_format="tao", probe_images=False)

for seq in ds.iter_sequences():
    print(seq.video_meta["sequence_name"], seq.frame_path(0), len(seq.boxes))
```

TAO `images.file_name` entries are resolved under `<root>/frames` (with an
already-present leading `frames/` supported for derived datasets). Category IDs
are preserved as their raw sparse IDs. Non-box annotation fields such as
`segmentation`, `area`, and `iscrowd` are preserved in each box's `attributes`.

Load an official VisDrone video split (VID object detection in videos or MOT
multi-object tracking), or a parent containing multiple split roots:

```python
from datamaite import load_mot

ds = load_mot(
    "/path/to/VisDrone2019-VID-train",
    dataset_format="visdrone_video",
    variant="auto",          # inferred from VID/MOT in the split name; or "vid" / "mot"
    include_ignored=False,   # skip score=0 and ignored-region category rows
    classes={1, 4, 9},       # optional category allowlist
)

for seq in ds.iter_sequences():
    print(seq.video_meta["variant"], seq.frame_filename(0), len(seq.boxes))
```

VisDrone Video uses image sequences with seven-digit, 1-based `.jpg` filenames
(e.g. `0000001.jpg`). Loaded sequences set `video_path=None`, `frame_dir`,
`frame_pattern="{frame:07d}.jpg"`, and `frame_number_base=1`; use
`seq.frame_filename(frame_index)` / `seq.frame_path(frame_index)` with the
model's 0-based frame index. The loader preserves raw VisDrone category IDs
(`0` ignored region, `1` pedestrian, ..., `11` others) in `category_id`.

Load still-image object detection from COCO or YOLO, and still-image
classification from YOLO/Ultralytics folder layouts, through task-first entry
points:

```python
from datamaite import load_ic, load_od

od = load_od("/path/to/coco", dataset_format="coco")
print(od.sample_count, od.num_detections, od.index2label())

# Example YOLO OD layout: images/train/a.jpg + labels/train/a.txt + data.yaml
od_yolo = load_od("/path/to/yolo-det", dataset_format="yolo")
print(od_yolo.sample_count, od_yolo.num_detections, od_yolo.index2label())

# Example YOLO IC layout: train/cat/a.jpg, train/dog/b.jpg, val/cat/c.jpg
ic = load_ic("/path/to/yolo-cls", dataset_format="yolo")
for sample in ic.iter_samples():
    print(sample.file_name, sample.split, sample.labels[0].category_name)
```

Load VisDrone's still-image DET layout (`images/` + `annotations/`) the same way:

```python
from datamaite import load_od, load_ic

# Object detection: one sample per image, VisDrone's DET classes.
od = load_od("/path/to/VisDrone2019-DET-train", dataset_format="visdrone")

# Image classification: one sample per labeled object (crop labeled by category).
# "ignored regions" (class 0) are excluded by default.
ic = load_ic("/path/to/VisDrone2019-DET-train", dataset_format="visdrone")
ic_with_ignored = load_ic(
    "/path/to/VisDrone2019-DET-train",
    dataset_format="visdrone",
    include_ignored_regions=True,
)
```

VisDrone ships no standard image-classification format; datamaite derives IC
samples from the DET annotations as object crops. **Limitation:** writing a
VisDrone-derived IC dataset through an image-copying IC writer would emit full
images, not crops — see the follow-up issue before round-tripping IC crops.

Both IC and OD use still-image records with first-class `split` and shared image
source fields (`path_or_uri`, `image_bytes`, `file_name`, `width`, `height`).
Because `yolo` is a shared format family, generic `load(..., dataset_format="yolo")`
or `convert(..., input_format="yolo", output_format="yolo")` calls should pass
`task="ic"` or `task="od"`; the task-first `load_ic` / `load_od` helpers set
that discriminator for you.

For a full load → verify → export-ready walkthrough on synthetic data, see
[docs/tool-usage/dataset_bridge_demo.ipynb](docs/tool-usage/dataset_bridge_demo.ipynb).

## MAITE interoperability

A loaded task dataset is MAITE-compatible by structure — no adapter or on-disk
conversion step. Video/FMV maps to MAITE's **multi-object-tracking** task (one
item per video). Still-image object detection maps to MAITE **object_detection**,
and still-image classification maps to MAITE **image_classification**. Core
install provides target arrays; pixel decoding needs the matching task extra
(`datamaite[fmv]`, `datamaite[od]`, or `datamaite[ic]`). Index the loaded
dataset directly:

```python
from datamaite import load_mot

ds = load_mot("/path/to/dataset")                # already a MAITE MOT Dataset

video_stream, target, metadata = ds[0]
frame = next(iter(video_stream))                 # VideoFrame: pixels (C,H,W), time_s, pts
boxes = target.frame_tracks[0].boxes             # xyxy, shape (N, 4)
```

The MOT MAITE surface probes each video's dimensions/time base itself via PyAV,
so the quick snippet above needs `datamaite[fmv]`. Loading with
`require_video=True` (true frame counts up front, which the
`empty_frame_policy="all"` view requires) additionally uses the OpenCV probe;
that is also included in `datamaite[fmv]`.

To configure the MOT view, copy the dataset with options (it's not a conversion —
the dataset is already MAITE):

```python
ds = ds.with_mot_options(empty_frame_policy="all", dataset_id="my-set")
```

Conventions: boxes are converted from `xywh` to MAITE's `xyxy`; ground-truth
`scores` are `1.0`; frames are decoded with PyAV (inject your own backend via
`with_mot_options(decoder=...)`); by default only annotated frames are emitted
(`empty_frame_policy="all"` emits every frame, and needs a probed frame count).
The `BoxTrackDataset` model is a box-track IR. Still-image tasks use separate
records (`ObjectDetectionDataset`, `ImageClassificationDataset`) and expose their
own MAITE surfaces. Indexing a still-image OD or IC dataset decodes images with
OpenCV, which ships in the task extras (`numpy` is part of the lean core), so
those tasks need `pip install datamaite[od]` or `datamaite[ic]` (or
`datamaite[all]`). `VideoClassificationDataset` is source-record-only until MAITE
grows a video-classification protocol.

## Writing & converting datasets (Python)

A **writer** serialises a loaded task dataset to an output format on disk;
`convert` pairs a loader and a writer of the same task for end-to-end on-disk →
on-disk conversion. MOT, VC, OD, and IC have writer surfaces for supported
output formats.

```python
from datamaite import load_mot, write, convert

# Write an in-memory dataset to disk (returns None; pass verbose=True for the file list)
ds = load_mot("/path/to/dataset")
write(ds, "/path/to/out", output_format="hmie")
files = write(ds, "/path/to/out", output_format="hmie", verbose=True)   # -> list of files written

# Or convert on disk → on disk in one call (verbose=True to get the file list back)
convert("/path/to/dataset", "/path/to/out", input_format="hmie", output_format="hmie")
```

Write MOTChallenge, TAO, or VisDrone Video with the same API. All three formats
are image-sequence based: video-backed inputs are decoded to frame images and
require the `fmv` extra, while existing image-sequence inputs copy their frame
files directly.

```python
from datamaite import load_mot, write

ds = load_mot(
    "/path/to/hmie",
    dataset_format="hmie",
    require_video=True,
)  # video-backed source
write(ds, "/path/to/mot-out", output_format="motchallenge", split="train")
write(ds, "/path/to/tao-out", output_format="tao", split="train")
write(ds, "/path/to/visdrone-out", output_format="visdrone_video", variant="mot")
```

Write Hugging Face Video Classification by loading the VC records and selecting
the matching output format:

```python
from datamaite import load_vc, write

vc = load_vc("/path/to/hf-video-dataset")
write(vc, "/path/to/hf-out", output_format="huggingface_video_classification")
```

For MOTChallenge, `annotation_source="gt"` (default) writes `gt/gt.txt` with
class and visibility columns; `annotation_source="det"` writes `det/det.txt`
with detection scores and optional world coordinates. For VisDrone Video,
`variant="vid"` writes Object Detection in Videos split roots, `variant="mot"`
writes Multi-Object Tracking split roots, and `variant="auto"` (default)
preserves the loaded sequence variant when present.

YOLO classification writes split/class/image directories; YOLO object detection
writes `images/<split>` and mirrored `labels/<split>` directories plus
`data.yaml`:

```python
from datamaite import load_ic, load_od, write

ic = load_ic("/path/to/yolo-cls", dataset_format="yolo")
write(ic, "/path/to/yolo-cls-out", output_format="yolo")

od = load_od("/path/to/yolo-det", dataset_format="yolo")
write(od, "/path/to/yolo-det-out", output_format="yolo")
```

HMIE, Hugging Face Video Classification, MOTChallenge, TAO, VisDrone Video,
COCO OD, YOLO OD, and YOLO IC have round-trip writers. The MOT routes recover
the same box/category/frame content represented by `BoxTrackDataset`; VC/OD/IC
routes preserve their task records within the constraints of each format (for
example, YOLO OD stores normalized boxes and class indices, not COCO
segmentation). Adding a new output format is a `Writer`
subclass + `@register_writer` — see [docs/architecture.md](docs/architecture.md).

## CLI Usage

```
datamaite validate <path> [options]

Options:
  -v, --verbose          Show individual findings per file
  -q, --quiet            Suppress progress output (for scripts)
  -o, --output FILE      Write full report to a file
  --skip-video-check     Skip FMV integrity checks (faster, JSON-only)
  --workers N            Number of parallel workers (default: CPU count)
  --json                 Emit results as JSON
  --jsonl                Emit results as newline-separated JSONL
  --format FORMAT        Dataset format (default: hmie)
  --debug                Enable debug logging
```

> **Note:** `--skip-video-check` (and `check_video_integrity=False`) reports
> FMV integrity as **SKIPPED**, not PASS — a skipped check is *not* a
> verified-clean check. Video↔annotation consistency is skipped too; the report
> shows a "Video checks disabled" banner so a skipped run is never mistaken for
> a clean one.

Exit codes: `0` = pass, `1` = warnings only, `2` = errors present.

## Validation Checks

Validation is currently implemented only for HMIE/Scale. For any non-HMIE
`--format`, `datamaite validate` raises `NotImplementedError`; loaders/writers
for COCO, YOLO IC/OD, MOTChallenge, TAO, VisDrone (video and still images), and
Hugging Face VC do not imply on-disk validation support yet.

For HMIE/Scale, the validator runs four checks against each dataset:

| Check | What it verifies |
|---|---|
| **Folder structure** | Snippet directories found with `seq_*` video containers |
| **FMV integrity** | Video files can be opened, frames decoded, not corrupted |
| **Annotation coverage** | Every annotation has a matching video and vice versa |
| **Scale spec compliance** | Annotations match the Scale Video Playback JSON format |

## Development

```bash
# Install with dev dependencies
poetry install --extras dev --extras all

# Run tests
poetry run pytest

# Lint and type check
poetry run pre-commit run --all-files
poetry run pyright src/

# Build wheel
poetry build
```

See [README_DEV.md](README_DEV.md) for alternative package managers (pixi, uv).

For a walk-through of how the code is organized — project layout,
reading order, and data-flow diagrams — see
[docs/architecture.md](docs/architecture.md).

## Dataset layout on disk

The validator is snippet-centric. Snippet dirs are identified by the
presence of a `seq_*/` video container; everything else is discovered
relative to that.

```
<batch_dir>/
    <snippet_name>_<id>_<seq>/           snippet directory
        <snippet_name>.json              snippet-level metadata (NOT a Scale annotation)
        scale/                           annotation dir (present in some families)
            *.json                       Scale Video Playback annotation
        <labeler>/                       alternative annotation dir (labeler subfolder)
            *.json
        seq_mp4/                         video container (always present)
            *.mp4
        seq_ts/                          alternative container (some datasets)
            *.ts
        mapp_metadata/ | 0601_metadata/  pipeline metadata (ignored)
            *.json
    scale/                               batch-level annotations (some families)
        *.json
    masks/                               batch-level masks (ignored)
```

Variations across families are tolerated: `scale/` vs labeler
subfolder, `seq_mp4/` vs `seq_ts/`, and the differing `*_metadata/`
directory names are all handled by discovery.
