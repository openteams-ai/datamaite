# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial `databridge` package scaffold (pyproject, Poetry primary, uv/pixi
  alternatives, hatchling build backend with hatch-vcs versioning).
- GitLab CI pipeline: lint (pre-commit), typecheck (pyright), build (uv build),
  test matrix (Python 3.10 / 3.11 / 3.12 with 90% coverage gate), dr-compliance.
- HMIE / Scale Video Playback dataset validator (`databridge validate <path>`):
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
  - Per-user cache at `~/.cache/databridge/validation.db`.
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
- Neutral in-memory dataset model (`databridge.model`): `Dataset` /
  `VideoSequence` / `BoxAnnotation` form the format-agnostic hub that
  every loader produces and every converter will consume, so any loader
  can feed any output format (N-to-M bridge).
- Loader architecture (`databridge.loaders`): a `Loader` base class
  defines the input-side contract, `register_loader` is the extension
  point, and `databridge.load(root, dataset_format=…)` dispatches across
  registered formats (with a `sniff`-based autodetection hook). `HmieLoader`
  is the reference implementation; adding a format is additive (subclass +
  register). See the "Loader architecture" section in `docs/architecture.md`.
- HMIE dataloader (`databridge.load_hmie`): loads an HMIE/Scale dataset
  into the neutral `Dataset` model (`VideoSequence` / `BoxAnnotation`
  records with a dataset-wide ontology-URI → category-id map). Reuses the
  existing discovery + Scale-schema layers instead of the hard-coded
  notebook walk; supports `annotation_dir` / `video_dir` overrides for
  flat layouts and an opt-in `require_video` mode that reads true frame
  counts via the `video` extra.
- Writer architecture (`databridge.writers`): a `Writer` base class defines
  the output-side contract (`BoxTrackDataset` → `list[Path]`), `register_writer`
  is the extension point, and `databridge.write(ds, dest, output_format=…)`
  dispatches across registered formats. `databridge.conversion.convert` pairs a
  loader and a writer for end-to-end on-disk → on-disk conversion. See the
  "Writer architecture" section in `docs/architecture.md`.
- HMIE writer (`HmieWriter`): the reference writer that serialises a
  `BoxTrackDataset` back to the HMIE on-disk layout. With the HMIE loader it
  closes a `load → write → load` round trip that recovers the same
  box/category content, proving the writer architecture and that
  `BoxTrackDataset` is a lossless hub.
- Flat-folder MP4 loader (`databridge.load_flat_mp4`, IR-3.3-S-1): loads
  immediate `.mp4` children encoded as H.264 or MPEG-2 into video-backed
  `VideoSequence` records with media metadata and no annotations.

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

### Documentation

- `README.md`: install / quick-start with Poetry.
- `README_DEV.md`: uv and pixi alternatives for contributors.

[Unreleased]: https://gitlab.jatic.net/jatic/orchestration-interoperability/databridge/-/compare/main...feature/hmie-validator
