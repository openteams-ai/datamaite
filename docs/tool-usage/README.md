# Tool-usage notebooks

Runnable notebooks for using `datamaite`, organized to mirror the bridge
architecture (see [`../architecture.md`](../architecture.md)).

- **`dataset_bridge_demo.ipynb`** — start here. A guided tour of the whole
  pipeline on synthetic data: load a dataset into the neutral `Dataset` model,
  verify it, and see how it is ready to export.
- **`validators/hmie.ipynb`** — how to **validate** an HMIE / Scale dataset.
  HMIE is the only validation format implemented today; loaders/writers for
  COCO, YOLO IC/OD, MOTChallenge, TAO, VisDrone, and Hugging Face VC do not imply
  `datamaite.validate()` support.
- **`exporters/<format>.ipynb`** — how to **export** the neutral `Dataset` to a
  given output format. One notebook per output format. Added as converters land
  (MOTChallenge first); see the roadmap in `dataset_bridge_demo.ipynb`.

## Why this split

datamaite is an N-to-M bridge. *Loaders* live on the **input** side
(format/task-specific: HMIE, COCO OD, YOLO IC/OD, …); *writers/converters* live on
the **output** side (MOTChallenge, YOLO IC/OD, COCO OD, …); task-specific neutral
datasets sit between them. Validation is deliberately narrower today:

1. **Input validation** (`validators/hmie.ipynb`) — HMIE-only, on disk: *is the
   source dataset trustworthy?* This is what `datamaite.validate()` does today.
2. **Model / export checks** — format-neutral tests on loaded task datasets:
   *is the model internally consistent, and does a conversion preserve it?*

So a new input/output format adds loader/writer docs first. A new non-HMIE
validator should be added only as a deliberate separate feature.
