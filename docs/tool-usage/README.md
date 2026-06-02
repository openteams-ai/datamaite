# Tool-usage notebooks

Runnable notebooks for using `databridge`, organized to mirror the bridge
architecture (see [`../architecture.md`](../architecture.md)).

- **`dataset_bridge_demo.ipynb`** — start here. A guided tour of the whole
  pipeline on synthetic data: load a dataset into the neutral `Dataset` model,
  verify it, and see how it is ready to export.
- **`validators/<format>.ipynb`** — how to **validate** a dataset of a given
  input format (does it conform to that format's rules: folder structure,
  schema, annotation coverage, video integrity). One notebook per input format.
  - `validators/hmie.ipynb` — HMIE / Scale (implemented today).
- **`exporters/<format>.ipynb`** — how to **export** the neutral `Dataset` to a
  given output format. One notebook per output format. Added as converters land
  (MOTChallenge first); see the roadmap in `dataset_bridge_demo.ipynb`.

## Why this split

databridge is an N-to-M bridge. *Loaders* and *validators* live on the **input**
side (format-specific: HMIE, COCO, YOLO, …); *converters* live on the **output**
side (MOTChallenge, YOLO, COCO, …); the neutral `Dataset` model sits between
them. Validation happens at two distinct points:

1. **Input validation** (`validators/`) — per input format, on disk: *is the
   source dataset trustworthy?* This is what `databridge.validate()` does, and
   it dispatches by format (`DatasetFormat`). Each input format has its own
   rules, so each gets its own validator notebook.
2. **Model / export validation** — format-neutral, on the loaded `Dataset`: *is
   the model internally consistent, and does a conversion preserve it?* Written
   once and shared across all formats.

So a new input format adds a `validators/<format>.ipynb`; a new output format
adds an `exporters/<format>.ipynb`; the top-level demo ties them together.
