# Tool-usage notebooks

Runnable notebooks for using `databridge`, organized to mirror the bridge
architecture (see [`../architecture.md`](../architecture.md)).

- **`dataset_bridge_demo.ipynb`** — start here. A guided tour of the whole
  pipeline on synthetic data: load a dataset into the neutral `BoxTrackDataset`
  model, verify it, convert it to MOTChallenge on disk, and consume it as a
  MAITE multi-object-tracking dataset.
- **`validators/<format>.ipynb`** — how to **validate** a dataset of a given
  input format (does it conform to that format's rules: folder structure,
  schema, annotation coverage, video integrity). One notebook per input format.
  - `validators/hmie.ipynb` — HMIE / Scale (implemented today).
- **`exporters/<format>.ipynb`** — how to **export** the neutral
  `BoxTrackDataset` to a given output format. One notebook per box-track output
  format. The MOT writers (HMIE, MOTChallenge, TAO, VisDrone) and the `write()`
  / `convert()` calls are implemented; `dataset_bridge_demo.ipynb` walks the
  MOTChallenge export end to end, and dedicated per-format walkthroughs are
  added as they get written.

This notebook set currently focuses on the box-track/MOT bridge. Registered
non-MOT loaders (for example Hugging Face video classification) return their own
task-specific dataset records and do not use the `BoxTrackDataset` writer or
MAITE-MOT surface until that task family gains a writer/protocol surface.

## Why this split

For box-track/MOT data, databridge is an N-to-M bridge. *Loaders* and
*validators* live on the **input** side (format-specific: HMIE, MOTChallenge,
TAO, …); *writers* live on the **output** side (MOTChallenge, TAO, VisDrone,
…); the neutral `BoxTrackDataset` model sits between them. Validation happens at
two distinct points:

1. **Input validation** (`validators/`) — per input format, on disk: *is the
   source dataset trustworthy?* This is what `databridge.validate()` does, and
   it dispatches by format (`DatasetFormat`). Each input format has its own
   rules, so each gets its own validator notebook.
2. **Model / export validation** — format-neutral, on the loaded
   `BoxTrackDataset`: *is the model internally consistent, and does a
   conversion preserve it?* Written once and shared across all formats.

So a new box-track input format adds a `validators/<format>.ipynb`; a new
box-track output format adds an `exporters/<format>.ipynb`; the top-level demo
ties them together.
