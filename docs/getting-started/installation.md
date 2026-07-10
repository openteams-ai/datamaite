# Installation

`datamaite` ships as one package with task-oriented extras. The core install is
kept deliberately small so annotation loading, conversion dispatch, and HMIE
structure/annotation validation do not pull in native media stacks. `datamaite` supports Python 3.10–3.13.

## From source with Poetry

```bash
git clone https://github.com/openteams-ai/datamaite.git
cd datamaite
poetry install --extras all
```

## Optional dependency extras

Optional dependencies are exposed as PEP 621 extras; install only the ones you
need with `--extras <name>` (repeat the flag to combine):

| Extra | Adds | Enables |
|---|---|---|
| _(none)_ | `pydantic`, `numpy` | load registered datasets into the in-memory model, build MAITE target arrays, run HMIE structure/annotation validation |
| `all` | union of task extras | all supported media/pixel decode paths |
| `fmv` | core + OpenCV + PyAV | FMV/video integrity checks, flat MP4 probing, video-backed MOT MAITE decode, video-backed MOT writer frame extraction |
| `od` | core + OpenCV | still-image object-detection pixel decode |
| `ic` | core + OpenCV | still-image image-classification pixel decode |
| `maite` | core + MAITE + PyAV | optional MAITE package plus MOT-video runtime for interoperability/conformance checks |
| `notebook` | `ipykernel` | Jupyter kernel for running the tutorial notebooks |
| `docs` | Sphinx, MyST-NB, … | building this documentation |
| `dev` | pytest, ruff, pyright, … | test, lint, and type-check tooling |

`maite` itself is not a core runtime dependency. Datamaite datasets conform to
MAITE protocols structurally; the `maite` extra is available for consumers and
conformance tests that want the MAITE package installed alongside the adapters.
It includes PyAV because the MOT MAITE view decodes video-backed sequences.

For example, to install everything needed to run the tutorial notebooks:

```bash
poetry install --extras all --extras maite --extras notebook
```
