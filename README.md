# databridge

A unified framework for dataset loading, conversion, and quality validation.

## Quick Start

```bash
# Clone and install
git clone https://gitlab.jatic.net/jatic/orchestration-interoperability/databridge.git
cd databridge
poetry install --with dev --with video

# Validate a dataset
databridge validate /path/to/dataset

# Validate multiple batches at once
databridge validate /path/to/batches/

# Verbose output (individual findings)
databridge -v validate /path/to/dataset

# Save full report to file
databridge validate /path/to/dataset -o report.txt
```

## CLI Usage

```
databridge validate <path> [options]

Options:
  -v, --verbose          Show individual findings per file
  -q, --quiet            Suppress progress output (for scripts)
  -o, --output FILE      Write full report to a file
  --skip-video-check     Skip FMV integrity checks (faster, JSON-only)
  --workers N            Number of parallel workers (default: CPU count)
  --json                 Emit results as JSON
  --jsonl                Emit results as newline-separated JSONL
  --format FORMAT        Dataset format (default: hmie)
  --debug                Enable debug logging
```

Exit codes: `0` = pass, `1` = warnings only, `2` = errors present.

## Validation Checks

The validator runs four checks against each dataset:

| Check | What it verifies |
|---|---|
| **Folder structure** | Snippet directories found with `seq_*` video containers |
| **FMV integrity** | Video files can be opened, frames decoded, not corrupted |
| **Annotation coverage** | Every annotation has a matching video and vice versa |
| **Scale spec compliance** | Annotations match the Scale Video Playback JSON format |

## Development

```bash
# Install with dev dependencies
poetry install --with dev --with video

# Run tests
poetry run pytest

# Lint and type check
poetry run pre-commit run --all-files
poetry run pyright src/

# Build wheel
poetry build
```

See [README_DEV.md](README_DEV.md) for alternative package managers (pixi, uv).

For a walk-through of how the code is organized — project layout,
reading order, and data-flow diagrams — see
[docs/architecture.md](docs/architecture.md).

## Dataset layout on disk

The validator is snippet-centric. Snippet dirs are identified by the
presence of a `seq_*/` video container; everything else is discovered
relative to that.

```
<batch_dir>/
    <snippet_name>_<id>_<seq>/           snippet directory
        <snippet_name>.json              snippet-level metadata (NOT a Scale annotation)
        scale/                           annotation dir (present in some families)
            *.json                       Scale Video Playback annotation
        <labeler>/                       alternative annotation dir (labeler subfolder)
            *.json
        seq_mp4/                         video container (always present)
            *.mp4
        seq_ts/                          alternative container (some datasets)
            *.ts
        mapp_metadata/ | 0601_metadata/  pipeline metadata (ignored)
            *.json
    scale/                               batch-level annotations (some families)
        *.json
    masks/                               batch-level masks (ignored)
```

Variations across families are tolerated: `scale/` vs labeler
subfolder, `seq_mp4/` vs `seq_ts/`, and the differing `*_metadata/`
directory names are all handled by discovery.
