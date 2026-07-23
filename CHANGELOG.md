# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Native IC/OD datasets now implement the MAITE `FieldwiseDataset` protocol
  (`get_input`/`get_target`/`get_metadata`) in addition to `__getitem__`.
  `get_target` and `get_metadata` avoid decoding the image when the target and
  dimensions are known without it, so fieldwise consumers (e.g. dataeval and
  other MAITE tooling) can read targets/metadata without the image extra (#77).

- Object-detection MAITE datum metadata now surfaces the source-preserving
  per-image passthrough (`sample.metadata`) as flat top-level keys, plus
  `file_name`. For COCO this exposes the full `images[]` extras
  (`license`/`date_captured`/`flickr_url`/`coco_url`/...) that MAITE metadata
  consumers (e.g. dataeval bias factors) read; YOLO/VisDrone provenance is
  surfaced the same way. The typed `id`/`height`/`width` keys always win over
  any same-named passthrough value, and a bare sample still yields exactly
  `{id, height, width}` (#79).
- Object-detection MAITE datum metadata now surfaces per-box source attributes
  as flat lists index-aligned to the target boxes, so metadata consumers
  (dataeval, which expands list-valued datum-metadata keys into per-object bias
  factors) can read them. This surfaces VisDrone `truncation`/`occlusion`/
  `visdrone_score` and other entries that detection records retain in their
  attributes. `target.scores` still carries ground-truth
  confidence; the raw VisDrone score stays a distinct `visdrone_score` factor
  (#80).

## [0.3.0] - 2026-07-23

### Added

- VisDrone Static-Images writers (IR-3.2-S-7): `write(dataset,
  output_format="visdrone")` serialises still-image object-detection and
  image-classification datasets to official `VisDrone2019-DET-<split>/`
  roots (`images/` + eight-field `annotations/*.txt`), with the
  detection/classification selection keyed on the dataset task. Class ids
  resolve through the shared fixed-taxonomy machinery (`class_map` option,
  aggregated #55 warnings); the static loaders now preserve
  `visdrone_category_id` in attributes so VisDrone-to-VisDrone round-trips
  are warning-free (#9).
- Flat-folder still-image loader (IR-3.2-S-1):
  `load_od(root, dataset_format="flat_images")` reads a flat directory of
  label-free `.jpg`/`.png`/`.tif` images as an unlabeled
  object-detection dataset (zero detections, no taxonomy). Explicit opt-in
  only — a bare folder of images is never autodetected. Images keep the
  lazy OpenCV decode (`datamaite[od]`). SafeTensors ingest, also named by
  IR-3.2-S-1, is deferred pending a program-standards change (#74) (#2).
- Hugging Face Vision still-image format (IR-3.2-S-2 loader + IR-3.2-S-6
  writer): `load_ic(root, dataset_format="huggingface_vision")` reads the
  ImageFolder classification convention (class folders, split folders, or
  `metadata.csv`/`metadata.jsonl` with `label`), and
  `load_od(root, dataset_format="huggingface_vision")` reads the
  object-detection convention (metadata `objects` column of parallel
  `bbox`/`categories` lists). `write(dataset,
  output_format="huggingface_vision")` mirrors both back out, with the
  detection/classification selection keyed on the dataset task (#3, #8).

### Fixed

- VisDrone still images: write -> reload is no longer lossy. The static
  loader reads `.tif`/`.tiff` images (the writer copies images verbatim, so
  a `.tif` source — e.g. arriving via `flat_images` — previously produced a
  root that reloaded as zero samples; suffixes the loader cannot read are
  now skipped with a warning at write time instead of silently vanishing on
  reload). The loader infers the official `test-dev`/`test-challenge`
  splits from split-root names instead of collapsing them onto `test`, so
  writer-emitted split roots round-trip their split identity. The writers
  no longer write a generic `category_id` 0 as VisDrone category 0
  ("ignored regions", score 0, excluded from evaluation) — such rows are
  dropped with an aggregated warning pointing at `class_map=` (#55
  provenance rules). The IC writer no longer copies an image whose
  annotation rows all drop, so an emitted root never contains images
  without annotation files (#9).

- Hugging Face Vision: the IC writer now preserves class-folder names with
  spaces/unicode/punctuation (e.g. `traffic light`, `café`) instead of skipping
  those samples, keeping write->reload label identity. The IC/OD loaders order
  integer category ids numerically (not lexically) and preserve original integer
  ids rather than re-indexing them; a metadata file no longer fabricates a label
  from a file_name's parent directory; and class folders literally named
  `train`/`test`/`val` are no longer mistaken for split directories (#3, #8).

- Hugging Face Vision: the OD writer now places `metadata.jsonl`/`.csv`
  *inside* each split directory (`train/metadata.jsonl`,
  `data/metadata.jsonl`) with directory-relative `file_name`s instead of one
  root-level file: `datasets.load_dataset("imagefolder", ...)` only
  associates metadata files within a split's directory tree, so the
  root-level file silently lost the `objects` column on the Hugging Face
  side (verified against `datasets` 5.x). The OD loader correspondingly
  scans all first-level directories (not just split-named ones) for
  metadata files; root-level metadata files remain supported for reading
  (#3, #8).

- Hugging Face Vision: custom split names can no longer break round-trips —
  the writers only emit ImageFolder-recognized split directories
  (`train`/`validation`/`test`; aliases such as `val`/`dev`/`eval` normalise,
  unknown split *options* raise, unknown *sample* splits fall back to the
  default split with a warning), since a directory like `holdout/` would
  reload as a class folder (IC) or lose its split (OD). The loaders now
  recognize the full Hugging Face split keyword set (`dev`, `testing`,
  `eval`, `evaluation`). The OD writer preserves category *names* in
  `objects.categories` when detections carry them (reloads keep `"person"`
  instead of a bare id; the numeric-id drop is declared in
  `lossy_without`). Docs now scope the format as the local
  ImageFolder-compatible layout (not general `datasets`/Hub support) and mark
  CSV OD metadata (JSON-encoded `objects`) as a datamaite extension; an
  optional test module verifies writer output loads with the real
  `datasets.load_dataset("imagefolder", ...)` when `datasets` is installed
  (#3, #8).

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
