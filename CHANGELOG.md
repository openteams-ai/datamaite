# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.2] - 2026-07-16

### Added

- VisDrone Static-Images loaders (IR-3.2-S-3): object detection
  (`load_od(root, dataset_format="visdrone")`) reads the official
  VisDrone-DET layout (`images/` + eight-field `annotations/*.txt`), and
  image classification (`load_ic(root, dataset_format="visdrone")`) derives
  one classification sample per labeled box (the object crop, labeled by its
  VisDrone category) from the same on-disk layout (#4).
- Cloud object storage support for HMIE: dataset roots can be `s3://`,
  `gs://`, or `az://` URLs with the matching extra installed
  (`datamaite[aws]`, `[gcs]`, `[azure]`, or `[cloud]` for all three).
  Cloud roots are HMIE-only; other format loaders raise a clear error on a
  cloud URL. Video integrity checks over cloud data additionally need the
  `fmv` extra — without it they are skipped with a `video_dependency`
  warning.
- MOTChallenge/VisDrone writers warn (aggregated, once per write) when
  falling back from generic `category_id` values to their fixed class
  tables, and accept an explicit `class_map=` option; categories missing
  from `class_map` are dropped with an aggregated warning (#55).
- Python 3.13 support (`requires-python = ">=3.10,<3.14"`), tested in CI
  alongside 3.10/3.11/3.12.

### Changed

- **Breaking:** `write()`/`convert()` now refuse a non-empty destination by
  default (`mode="error"`). Pass `mode="replace"` to clear the destination
  first or `mode="append"` for the previous write-into behavior (#55).

## [0.2.1] - 2026-06-30

### Added

- Object-detection and image-classification formats via task-aware registries: a
  COCO object-detection loader and writer (`load_od(..., dataset_format="coco")`,
  `write(..., output_format="coco")`) and a YOLO/Ultralytics image-classification
  loader and writer (`load_ic(..., dataset_format="yolo")`). A single
  `DatasetFormat` can now back more than one task, dispatched by `(task, format)`.
- YOLO/Ultralytics object-detection loader and writer
  (`load_od(..., dataset_format="yolo")`, `write(..., output_format="yolo")`) with
  standard `images/<split>` + `labels/<split>` and `<split>/images` +
  `<split>/labels` layouts, `data.yaml` discovery, and `load → write → load`
  round-tripping (boxes are clipped to the image on write).
- `datamaite.load_vc(root, dataset_format=…)`: a task-first public entry point
  for video classification, the analogue of `load_mot`. It pins the return type
  to `VideoClassificationDataset` and raises `TypeError` if the resolved format
  produces a different task's dataset. Defaults to
  `dataset_format="huggingface_video_classification"`.
- Hugging Face VideoFolder-style video-classification writer, completing the
  `load → write` round trip for the `VideoClassificationDataset` model.

### Changed

- **Breaking:** the package, distribution, and CLI have been renamed from
  `databridge` to `datamaite`. Update imports (`import datamaite`), the console
  entry point (`datamaite validate …`), and the dependency name; there is no
  `databridge` compatibility shim.
- **Breaking:** `load_huggingface_video_classification` is no longer part of the
  public `datamaite` API. Use the task-first `load_vc(...)` instead (the
  format-specific helper lives on internally in
  `datamaite._formats.huggingface_video_classification.loader`). This makes the
  public loader surface a consistent rule — generic `load` plus one
  `load_<task>` per task (`load_mot`, `load_od`, `load_ic`, `load_vc`) —
  matching how the per-format MOT `load_<format>` helpers were already made
  internal.
- **Breaking:** `load_yolo_image_classification` is no longer part of the public
  `datamaite` API. Use `load_ic(..., dataset_format="yolo")` instead.
- `datamaite.load` (and task-first loaders like `load_mot` / `load_vc`) now fail
  fast on a bad dataset root: a nonexistent path raises `FileNotFoundError` and
  a non-directory path raises
  `NotADirectoryError`, instead of silently returning an empty dataset. A root
  that exists but yields no loadable items still returns an empty dataset, now
  with a `WARNING` so an empty result (e.g. wrong format or wrong subdirectory)
  is never silent.
- `datamaite.load(..., dataset_format=None)` now raises on ambiguous sniff
  matches instead of picking the first registered loader, so multi-task formats
  like YOLO cannot be silently autodetected as the wrong task.
- `datamaite.write` (and `convert`, which forwards to it) now return ``None``
  by default and only return the ``list[Path]`` of files written when called
  with ``verbose=True``. The full list is one path per frame image, which
  floods interactive/REPL output; the file list is now opt-in. Side effects
  (the files written) are unchanged.
- Optional dependencies are declared once via PEP 621
  `[project.optional-dependencies]` instead of duplicated Poetry dependency
  groups, so the extras stay in sync across Poetry, uv, and pip.

### Fixed

- Corrected stale optional-dependency references (e.g. `datamaite[video]`) so
  installs resolve to the current task-oriented extras (`datamaite[fmv]`,
  `datamaite[all]`, …).

### Documentation

- Added a Sphinx documentation build (`docs/`).

## [0.2.0] - 2026-06-16

### Added

- Neutral in-memory dataset model (`datamaite.model`): `Dataset` /
  `VideoSequence` / `BoxAnnotation` form the format-agnostic hub that
  every loader produces and every converter will consume, so any loader
  can feed any output format (N-to-M bridge).
- Loader architecture (`datamaite.loaders`): a `Loader` base class
  defines the input-side contract, `register_loader` is the extension
  point, and `datamaite.load(root, dataset_format=…)` dispatches across
  registered formats (with a `sniff`-based autodetection hook). `HmieLoader`
  is the reference implementation; adding a format is additive (subclass +
  register). See the "Loader architecture" section in `docs/architecture.md`.
- HMIE dataloader (`datamaite.load_mot(..., dataset_format="hmie")`):
  loads an HMIE/Scale dataset into the neutral `BoxTrackDataset` model
  (`VideoSequence` / `BoxAnnotation`
  records with a dataset-wide ontology-URI → category-id map). Reuses the
  existing discovery + Scale-schema layers instead of the hard-coded
  notebook walk; supports `annotation_dir` / `video_dir` overrides for
  flat layouts and an opt-in `require_video` mode that reads true frame
  counts via the `fmv` or `all` extra.
- Writer architecture (`datamaite.writers`): a `Writer` base class defines
  the output-side contract (`BoxTrackDataset` → `list[Path]`), `register_writer`
  is the extension point, and `datamaite.write(ds, dest, output_format=…)`
  dispatches across registered formats. `datamaite.conversion.convert` pairs a
  loader and a writer for end-to-end on-disk → on-disk conversion. See the
  "Writer architecture" section in `docs/architecture.md`.
- HMIE writer (`HmieWriter`): the reference writer that serialises a
  `BoxTrackDataset` back to the HMIE on-disk layout. With the HMIE loader it
  closes a `load → write → load` round trip that recovers the same
  box/category content, proving the writer architecture and that
  `BoxTrackDataset` is a lossless hub.
- Flat-folder MP4 loader
  (`datamaite.load_mot(..., dataset_format="flat_mp4")`, IR-3.3-S-1):
  loads immediate `.mp4` children encoded as H.264 or MPEG-2 into video-backed
  `VideoSequence` records with media metadata and no annotations.
- MOTChallenge loader and writer (`dataset_format="motchallenge"` /
  `MotChallengeWriter`).
- TAO (Tracking Any Object) loader and writer (`dataset_format="tao"` /
  `TaoWriter`); video-backed TAO writes need the `datamaite[fmv]` or
  `datamaite[all]` extra.
- VisDrone video loader and writer (`dataset_format="visdrone"` /
  `VisDroneVideoWriter`).
- Hugging Face VideoFolder-style video-classification loader
  (`HuggingFaceVideoClassificationLoader` /
  `load_huggingface_video_classification`) into the
  `VideoClassificationDataset` model.
- Task / IC / OD foundation: `Task` taxonomy, source-preserving `Taxonomy` /
  `CategoryEntry` (`datamaite.taxonomy`), and canonical `xywh` bbox +
  conversions (`datamaite.geometry`).
- MAITE interoperability (`datamaite.maite`, optional `datamaite[maite]`
  extra): `BoxTrackDataset` conforms to the MAITE MOT protocol structurally;
  `load_mot` returns a MAITE-indexable dataset and `with_mot_options`
  configures the view.

### Changed

- Task-first loader API: `load_mot(root, dataset_format=…)` replaces the
  per-format `load_*` public functions.
- Format loaders/writers reorganised into per-format `datamaite._formats`
  packages.
- Skipped video checks are reported as `SKIPPED` instead of `PASS`.
- Validator notebook exposes the multi-dataset (collection) HTML report.
- `license` set to `Apache-2.0`, replacing the `LicenseRef-TBD` placeholder.

### Fixed

- Batch-level `scale/` discovery now **merges per batch** with the
  snippet-centric pass instead of being an all-or-nothing `root/scale`
  fallback, so per-batch `scale/` under a multi-batch parent and trees mixing
  both layouts are fully discovered. `SnippetPair` carries a `snippet_dir` so
  `snippet_count` no longer collapses centralized-`scale/` annotations onto
  the batch root; non-annotation JSON in a `scale/` dir is skipped; and
  `match_annotation_to_video` returns an orphan instead of guessing when two
  videos share a basename.
- Frame-key mapping snaps to a near integer before flooring (fixes a
  floating-point off-by-one on rates like 29.97/14.985); relative
  `annotation_dir`/`video_dir` overrides resolve against `root` not the CWD;
  non-finite bbox coordinates and NaN/Inf/non-positive fps & duration are
  rejected.

## [0.1.0] - 2026-05-20

### Added

- Initial `datamaite` package scaffold (pyproject, Poetry primary, uv/pixi
  alternatives, hatchling build backend with hatch-vcs versioning).
- GitLab CI pipeline: lint (pre-commit), typecheck (pyright), build (uv build),
  test matrix (Python 3.10 / 3.11 / 3.12 with 90% coverage gate), dr-compliance.
- HMIE / Scale Video Playback dataset validator (`datamaite validate <path>`):
  - Snippet-centric folder-structure discovery for the CDAO SUNet layout.
  - FMV integrity checks (open, frame count, resolution, first/mid/last frame,
    FPS, flat-frame detection) using OpenCV.
  - Annotation coverage checks (orphan annotations, orphan videos).
  - Scale schema conformance via Pydantic models derived from the Scale
    Video Playback reference PDF; handles both full-envelope and
    unwrapped annotation formats.
  - Consistency checks across annotation + video (FPS agreement, frame
    bounds, bbox bounds within image dimensions).
  - Parallel per-pair validation via `ProcessPoolExecutor` with
    `--workers` tuning and main-process cache lookup.
- `docs/schemas/scale-video-playback-v1.schema.json`: machine-readable
  JSON Schema of the Scale Video Playback annotation format.
- SQLite-backed validation cache (`ValidationCache`):
  - Per-file fingerprints (SHA-256 of first 1 MB + size + mtime).
  - WAL journal mode with batched commits (50 writes per flush).
  - `--no-cache` and `--clean` CLI flags; cache hit/miss reporting.
  - Per-user cache at `~/.cache/datamaite/validation.db`.
- CLI dashboard output:
  - 4-check status grid (Folder structure, FMV integrity, Annotation
    coverage, Scale spec compliance) with PASS / WARN / FAIL / N/A states.
  - Multi-batch table view when the path contains sibling batch
    directories; per-batch status indicators and totals.
  - Progress indicators on TTY (pair counts, phase status messages).
  - `--quiet`, `--verbose`, `--debug` modes; `NO_COLOR` env support.
- Output formats:
  - Text summary (default, and `.txt` via `-o`).
  - JSON (`--json` / `.json` extension).
  - JSONL (`--jsonl`) for streaming to `jq` / `grep`.
  - Self-contained HTML report (`.html` extension or default `-o` with
    no filename): inline CSS/JS, no external dependencies, interactive
    dashboard, light/dark theme, search/filter, lazy rendering, print-
    ready, WCAG AA contrast.
- Multi-batch HTML report: aggregated view + clickable batch table that
  swaps to per-batch detail; deep-linkable via `#batch=<name>` URL hash.
- Finding cap (`--max-findings-per-check`) to bound memory on
  pathological datasets; `finding_counts` stays accurate under caps.

### Documentation

- `README.md`: install / quick-start with Poetry.
- `README_DEV.md`: uv and pixi alternatives for contributors.
