# databridge

A unified framework for dataset loading, conversion, and quality validation.

## Quick Start

```bash
# Clone and install
git clone https://gitlab.jatic.net/jatic/orchestration-interoperability/databridge.git
cd databridge
poetry install --with dev --with video

# Validate a dataset
databridge validate /path/to/dataset

# Validate multiple batches at once
databridge validate /path/to/batches/

# Verbose output (individual findings)
databridge -v validate /path/to/dataset

# Save full report to file
databridge validate /path/to/dataset -o report.txt
```

## Supported formats

databridge is a bridge: a **loader** reads an input format into one neutral
in-memory model (`BoxTrackDataset`), a **validator** checks a format on disk,
and a **writer** serialises the model out to an output format (`convert` pairs a
loader and a writer for on-disk → on-disk conversion). HMIE/Scale,
MOTChallenge, TAO, and VisDrone Video are implemented input formats; HMIE/Scale
is the reference output format (proving the writer architecture via a load →
write → load round trip), and other writers are planned.

| Format | Load | Validate | Write |
|---|---|---|---|
| HMIE / Scale (FMV) | ✅ | ✅ | ✅ |
| MOTChallenge | ✅ | — | planned |
| TAO | ✅ | — | planned |
| VisDrone Video (VID/MOT) | ✅ | — | planned |
| YOLO | planned | planned | planned |
| COCO | planned | planned | planned |

See [docs/architecture.md](docs/architecture.md) for the loader / writer design
and how to add a new loader or writer.

## Loading datasets (Python)

Load an HMIE/Scale dataset into the neutral `BoxTrackDataset` model:

```python
from databridge import load_hmie

ds = load_hmie("/path/to/dataset")
print(ds.sequence_count, "sequences,", ds.num_boxes, "boxes")

for seq in ds.iter_sequences():
    for box in seq.boxes:
        print(box.frame_index, box.track_id, box.category_name, box.bbox)
```

> `ds` is also a MAITE dataset (see below), so `len(ds)` / `ds[i]` / `for x in ds`
> are the **MAITE item** view (one per video). Use `ds.sequence_count` /
> `ds.iter_sequences()` / `ds.sequences` for the **record** view shown above.

Or via the format-dispatching entry point (same result):

```python
from databridge import load

ds = load("/path/to/dataset", dataset_format="hmie")
```

For non-standard layouts (flat annotation/video directories) and true frame
counts probed from the videos:

```python
ds = load_hmie(
    "/path/to/dataset",
    annotation_dir="annotations/",
    video_dir="videos/",
    require_video=True,  # needs the `video` extra
)
```

Load a standard MOTChallenge benchmark root (with `train/` and/or `test/`
splits) the same way:

```python
from databridge import load_motchallenge

ds = load_motchallenge(
    "/path/to/MOT17",
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
To optionally probe the first frame image with OpenCV for metadata, call
`load_motchallenge(..., probe_images=True)` after installing `databridge[video]`.

Load an official TAO dataset root (COCO-style `annotations/*.json` plus frame
files) similarly:

```python
from databridge import load_tao

ds = load_tao("/path/to/TAO", probe_images=False)

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
from databridge import load_visdrone_video

ds = load_visdrone_video(
    "/path/to/VisDrone2019-VID-train",
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

For a full load → verify → export-ready walkthrough on synthetic data, see
[docs/tool-usage/dataset_bridge_demo.ipynb](docs/tool-usage/dataset_bridge_demo.ipynb).

## MAITE interoperability

A loaded dataset **is** a [MAITE](https://github.com/mit-ll-ai-technology/maite)
multi-object-tracking dataset — no adapter or on-disk conversion step — so it can
feed MAITE-compatible models, metrics, and augmentations directly. Install the extra:

```bash
pip install databridge[maite]
```

Video/FMV maps to MAITE's **multi-object-tracking** task (one item per video).
Index the loaded dataset directly:

```python
from databridge import load_hmie

ds = load_hmie("/path/to/dataset")               # already a MAITE MOT Dataset

video_stream, target, metadata = ds[0]
frame = next(iter(video_stream))                 # VideoFrame: pixels (C,H,W), time_s, pts
boxes = target.frame_tracks[0].boxes             # xyxy, shape (N, 4)
```

The MAITE surface probes each video's dimensions/time base itself (via PyAV), so the
quick snippet above needs only the `maite` extra. Loading with `require_video=True`
(true frame counts up front, which the `empty_frame_policy="all"` view requires)
additionally uses the OpenCV probe — install both extras for that:
`pip install databridge[maite,video]`.

To configure the MOT view, copy the dataset with options (it's not a conversion —
the dataset is already MAITE):

```python
ds = ds.with_mot_options(empty_frame_policy="all", dataset_id="my-set")
```

Conventions: boxes are converted from `xywh` to MAITE's `xyxy`; ground-truth
`scores` are `1.0`; frames are decoded with PyAV (inject your own backend via
`with_mot_options(decoder=...)`); by default only annotated frames are emitted
(`empty_frame_policy="all"` emits every frame, and needs a probed frame count).
The `BoxTrackDataset` model is a box-track IR — labels that are not bounding
boxes, and tasks other than multi-object tracking, are out of scope.

## Writing & converting datasets (Python)

A **writer** serialises a loaded `BoxTrackDataset` to an output format on disk;
`convert` pairs a loader and a writer for end-to-end on-disk → on-disk
conversion. Any registered loader can feed any registered writer.

```python
from databridge import load_hmie, write, convert

# Write an in-memory dataset to disk
ds = load_hmie("/path/to/dataset")
files = write(ds, "/path/to/out", output_format="hmie")   # -> list of files written

# Or convert on disk → on disk in one call
convert("/path/to/dataset", "/path/to/out", input_format="hmie", output_format="hmie")
```

HMIE is the reference writer: `load_hmie → write(output_format="hmie") →
load_hmie` recovers the same box/category content (the round trip that proves
the writer architecture and that `BoxTrackDataset` is a lossless hub). Adding a
new output format is a `Writer` subclass + `@register_writer` — see
[docs/architecture.md](docs/architecture.md).

## CLI Usage

```
databridge validate <path> [options]

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

Exit codes: `0` = pass, `1` = warnings only, `2` = errors present.

## Validation Checks

The validator runs four checks against each dataset:

| Check | What it verifies |
|---|---|
| **Folder structure** | Snippet directories found with `seq_*` video containers |
| **FMV integrity** | Video files can be opened, frames decoded, not corrupted |
| **Annotation coverage** | Every annotation has a matching video and vice versa |
| **Scale spec compliance** | Annotations match the Scale Video Playback JSON format |

## Development

```bash
# Install with dev dependencies
poetry install --with dev --with video

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
