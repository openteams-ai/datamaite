# Tests

Two tiers, by directory:

| Tier | Location | Runs | Needs |
|------|----------|------|-------|
| **Unit / hermetic** | `tests/*.py` | always (default `pytest`) | nothing — synthetic `tmp_path` fixtures, offline, in-process |
| **End-to-end (e2e)** | `tests/e2e/*.py` | opt-in only | a real external dataset checkout + the `video`/`maite` extras |

The split is intentional: the hermetic suite is the always-on safety net (fast,
offline, runs locally and in every pipeline), and the e2e suite is the
real-data confidence check layered on top. They can look similar — e.g. the
cross-format conversion matrix exists in **both** tiers — but they catch
different classes of bug:

- `tests/test_conversion_matrix.py` (hermetic) proves the conversion **logic**
  is correct, including malformed-input edge cases (fps mismatch,
  drop-with-warning, registry guardrails) that are impossible to express
  against clean real data.
- `tests/e2e/test_example_data_matrix.py` (e2e) proves datamaite loads and
  round-trips the **real on-disk datasets** and that counts match the
  example-data repo's `MANIFEST.md` — a contract check spanning two repos.

Neither replaces the other.

## Running

```bash
# Default: hermetic suite only (e2e is deselected by the `integration` marker)
poetry run pytest

# e2e: point at an example-data checkout, then opt in via the marker
git clone https://gitlab.jatic.net/jatic/orchestration-interoperability/datamaite-example-datasets.git
export DATAMAITE_DATASETS_ROOT=$PWD/datamaite-example-datasets/datasets
poetry run pytest tests/e2e -m integration -s    # -s shows the summary table
```

## How the opt-in works

Two independent gates keep e2e tests out of the default run:

1. **Marker** — e2e tests are marked `integration`, and `pyproject.toml`'s
   `addopts` deselects `not integration` by default. This is the mechanism;
   the `tests/e2e/` directory is the human-readable signal.
2. **Self-skip** — each e2e test skips when `DATAMAITE_DATASETS_ROOT` is unset
   or missing, so an accidental `-m integration` locally is a clean skip, not a
   failure.

CI runs the e2e tier in its own `integration` job (see `.gitlab-ci.yml`), which
clones the example-data repo over git-LFS and gates the pipeline
(`allow_failure: false`).

## Adding tests

- New unit test → `tests/test_<thing>.py`, no marker. Must run offline.
- New e2e test → `tests/e2e/test_<thing>.py`, `pytestmark = pytest.mark.integration`,
  and skip cleanly when its external dependency is unavailable.
