# Packaging and extras

Datamaite ships as one package with task-oriented extras. The core install is
kept deliberately small so annotation loading, conversion dispatch, and HMIE
structure/annotation validation do not pull native media stacks.

## Install matrix

| Install command | Direct runtime deps | Enables |
|---|---|---|
| `pip install datamaite` | `pydantic`, `numpy` | load registered datasets into IRs, build MAITE target arrays, run HMIE structure/annotation validation |
| `pip install datamaite[fmv]` | core + OpenCV + PyAV | FMV/video integrity checks, flat MP4 probing, video-backed MOT MAITE decode, video-backed MOT writer frame extraction |
| `pip install datamaite[od]` | core + OpenCV | still-image object-detection pixel decode |
| `pip install datamaite[ic]` | core + OpenCV | still-image image-classification pixel decode |
| `pip install datamaite[all]` | union of task extras | all supported media/pixel decode paths |
| `pip install datamaite[maite]` | core + MAITE + PyAV | optional MAITE package plus MOT-video runtime for interoperability/conformance checks |

`maite` itself is not a core runtime dependency. Datamaite datasets conform to
MAITE protocols structurally; the `maite` extra is available for consumers and
conformance tests that want the MAITE package installed alongside the adapters.
It includes PyAV because the MOT MAITE view decodes video-backed sequences.

## Dependency direction

Core code may depend on pure-Python annotation/model utilities and `numpy` target
arrays. Native media dependencies stay behind lazy imports:

- OpenCV (`cv2`) is imported only by FMV integrity/probing paths, image probing,
  image decode, or frame extraction writers.
- PyAV (`av`) is imported only by the MOT video decoder.
- Optional parquet metadata readers (`pyarrow` / `pandas`) remain opportunistic;
  the Hugging Face video-classification loader falls back when unavailable.

Importing `datamaite`, loading annotations, and constructing task datasets must
not import `cv2`, `av`, or `maite`.

## CI contract

CI installs PEP 621 extras instead of Poetry dependency groups. Lint uses
`--extras dev`; typecheck/tests compose `--extras dev --extras maite --extras all`
so tooling, MAITE conformance deps, and task media decoders are present. Docs
jobs install `--extras docs`; that extra lists Sphinx, the notebook kernel, and
runtime imports directly because Poetry extras cannot include other extras. A
dedicated packaging smoke check should continue to use the base project metadata
so `poetry check` and `poetry build` verify the wheel without optional extras.

## Deprecated extras

The previous dependency-named `video` extra was removed before the public API
stabilized. Use task extras instead:

- `video` → `fmv` for video/MOT media, or `all` for every task media stack

The `maite` extra remains as an interoperability/conformance convenience; it is
not required for core annotation loading or target construction.
