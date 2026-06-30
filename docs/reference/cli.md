# HMIE Validation CLI

The `datamaite` CLI provides two commands — validate to check a dataset's structure, video integrity, annotation coverage, and schema compliance, and stats to summarize its duration/frame/box distributions — with global flags for verbosity and JSON/HTML output.

`datamaite [-v] [-q] [--debug] <command> ...`

**Global flags:** `-v/--verbose` (show individual findings), `-q/--quiet` (suppress progress, for scripts), `--debug` (detailed logging).

## **Commands**


### `datamaite validate <path> [options]`

Validate a dataset (folder structure, FMV integrity, annotation coverage, schema).

| Option | Description |
| :--- | :--- |
| `--skip-video-check` | Skip FMV integrity checks (OpenCV). |
| `--workers N` | Parallel worker processes for per-pair validation. |
| `--no-cache` | Bypass the validation cache. |
| `--clean` | Clear the cache before running. |
| `-o, --output <file>` | Write report to a file (format inferred from extension: `.txt`/`.json`/`.html`). |
| `--json` | Emit JSON to stdout. |
| `--jsonl` | Emit JSONL to stdout (stream to `jq`/`grep`). |

### `datamaite stats <path> [--json]`

Summarize a dataset's duration / frame / box distributions.
