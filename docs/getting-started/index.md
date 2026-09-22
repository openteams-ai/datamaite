# Getting Started

`datamaite` is a unified framework for dataset loading, conversion, and quality
validation. It acts as a bridge: a **loader** reads an input format into a
source-preserving in-memory dataset, a **validator** checks a format on disk,
and a **writer** serialises supported datasets out to an output format.

New here? Start with [Installation](installation.md), then pick the
[tutorial](../tutorials/index.md) matching your data — full-motion video,
image classification, or object detection. The
[Reference](../reference/index.md) covers the architecture and CLI details.

- [Loading from cloud object storage](cloud-storage.md) — use `s3://` / `gs://` / `az://` URLs as dataset roots.

## Quick start

After [installing](installation.md), load up a dataset

```python
from datamaite import convert
from datamaite.loaders import load
from pathlib import Path

hmie_datadir = Path('path/to/datasets')  # replace this with your local dataset

hmie_dataset = load(hmie_datadir, dataset_format='hmie')
```

The same shape serves still images: `load_ic(root, dataset_format="yolo")`
for classification folders and `load_od(root, dataset_format="coco")` for
detection. [Working with IC Datasets](../tutorials/Working_with_IC_datasets.ipynb)
and [Working with OD Datasets](../tutorials/Working_with_OD_datasets.ipynb)
walk through both on synthetic data; the rest of this page stays with HMIE.

Convert to another format:

```python
visdrone_datadir = Path('output_dir/visdrone')

convert(hmie_datadir, visdrone_datadir, input_format="hmie", output_format="visdrone_video")
```

Run validation checks on the HMIE dataset:

```python
from datamaite import validate

single_result = validate(hmie_datadir)

single_result.summary()
```

```{toctree}
:maxdepth: 1
:hidden:

installation
cloud-storage
```
