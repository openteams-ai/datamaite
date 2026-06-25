# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- `datamaite.load` (and `load_mot`) now fail fast on a bad dataset root: a
  nonexistent path raises `FileNotFoundError` and a non-directory path raises
  `NotADirectoryError`, instead of silently returning an empty dataset. A root
  that exists but yields no loadable items still returns an empty dataset, now
  with a `WARNING` so an empty result (e.g. wrong format or wrong subdirectory)
  is never silent.
- `datamaite.write` (and `convert`, which forwards to it) now return ``None``
  by default and only return the ``list[Path]`` of files written when called
  with ``verbose=True``. The full list is one path per frame image, which
  floods interactive/REPL output; the file list is now opt-in. Side effects
  (the files written) are unchanged.

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
  counts via the `video` extra.
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
  `TaoWriter`); video-backed TAO writes need the `datamaite[video]` extra.
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
