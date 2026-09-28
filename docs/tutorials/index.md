# Tutorials

These tutorials walk you through `datamaite` via hands-on, runnable code. These
are also available as Jupyter Notebooks in the package repository if you'd like
to run them yourself.

Each tutorial covers one part of `datamaite`: pick the chapter for your task --
full-motion video, image classification, or object detection -- or go straight
to validation or cloud I/O.

For full-motion video datasets -- loading, writing, and converting between
formats on real example data -- see [Working with FMV
Datasets](Working_with_FMV_datasets.ipynb).

For still-image datasets, [Working with IC
Datasets](Working_with_IC_datasets.ipynb) covers image classification and
[Working with OD Datasets](Working_with_OD_datasets.ipynb) covers object
detection -- both run on tiny synthetic data, so they work in any checkout.

For validation of HMIE datasets (Scale Video Playback annotations in a snippet
folder layout -- see the [architecture
reference](../reference/architecture.md)), check out the [HMIE Validation
tutorial](HMIE_Validation.ipynb). It produces a shareable HTML rollup; see the
published [Example Validation
Report](../example-validation-report.html){.external} for what that looks like.

To validate and load HMIE datasets straight from cloud object storage, see [HMIE
Datasets from Cloud Storage](HMIE_Cloud_Storage.ipynb) -- it runs hermetically
against fsspec's `memory://` filesystem, which exercises the identical code path
as `s3://`/`gs://`/`az://`.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`arrow-switch` Working with FMV Datasets
:link: Working_with_FMV_datasets
:link-type: doc

Load, inspect, and convert full motion video datasets between formats with
MAITE-compliant dataset wrappers.
:::

:::{grid-item-card} {octicon}`tag` Working with IC Datasets
:link: Working_with_IC_datasets
:link-type: doc

Round-trip YOLO classification folders and load VisDrone crop samples, all on
tiny synthetic data.
:::

:::{grid-item-card} {octicon}`image` Working with OD Datasets
:link: Working_with_OD_datasets
:link-type: doc

Load, inspect, and convert object-detection datasets (COCO, YOLO, VisDrone) on
tiny synthetic data.
:::

:::{grid-item-card} {octicon}`checklist` HMIE Validation
:link: HMIE_Validation
:link-type: doc

Validate HMIE/Scale datasets for structure, coverage, video integrity, and spec
compliance, then roll the results into a shareable HTML report.
:::

:::{grid-item-card} {octicon}`cloud` HMIE Datasets from Cloud Storage
:link: HMIE_Cloud_Storage
:link-type: doc

Validate and load HMIE datasets directly from S3/GCS/Azure-style object storage,
with streaming video integrity checks.
:::

::::

```{toctree}
:maxdepth: 1
:hidden:

Working_with_FMV_datasets
Working_with_IC_datasets
Working_with_OD_datasets
HMIE_Validation
HMIE_Cloud_Storage
```
