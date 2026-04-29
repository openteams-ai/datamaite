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

### Documentation

- `README.md`: install / quick-start with Poetry.
- `README_DEV.md`: uv and pixi alternatives for contributors.

[Unreleased]: https://gitlab.jatic.net/jatic/orchestration-interoperability/databridge/-/compare/main...feature/hmie-validator
