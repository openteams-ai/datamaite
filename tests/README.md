# Tests

Two tiers, by directory:

| Tier | Location | Runs | Needs |
|------|----------|------|-------|
| **Unit / hermetic** | `tests/*.py` | always (default `pytest`) | nothing — synthetic `tmp_path` fixtures, offline, in-process |
| **End-to-end (e2e)** | `tests/e2e/*.py` | opt-in only | a real external dataset checkout + the `video`/`maite` extras |
| **S3 end-to-end (e2e)** | `tests/e2e/test_s3_compatible.py` | opt-in only | a real S3-API server (SeaweedFS) + the `aws`/`fmv` extras |

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
uv run pytest

# e2e: point at an example-data checkout, then opt in via the marker
git clone https://gitlab.jatic.net/jatic/orchestration-interoperability/datamaite-example-datasets.git
export DATAMAITE_DATASETS_ROOT=$PWD/datamaite-example-datasets/datasets
uv run pytest tests/e2e -m integration -s    # -s shows the summary table
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

## S3 end-to-end tier

`tests/e2e/test_s3_compatible.py` is a separate e2e tier that runs datamaite
against a real S3-API server instead of the shared example-data checkout: it
proves the ranged-read video streaming and `storage_options` plumbing work
against real S3 semantics, not just against fsspec's `memory://` filesystem
(which is what the coverage-gated unit suite's cloud tests use instead, to
stay hermetic — see `tests/e2e/test_cloud_integration.py` and `tests/test_loaders.py`).
Like the rest of `tests/e2e/`, it carries the `integration` marker and
self-skips when its environment variables are unset, so it never affects the
default `pytest` run or the 90% coverage gate.

To run it locally, install the extras the tier needs, then start the same
pinned Apache-2.0 SeaweedFS image used by CI. The image below must match the
`S3_E2E_IMAGE` variable in the `e2e-s3` job in `.gitlab-ci.yml` — keep the two
in sync:

```bash
uv sync --extra dev --extra aws --extra fmv

docker run -d --rm --name datamaite-seaweedfs-e2e -p 9123:8333 \
  -e AWS_ACCESS_KEY_ID=datamaite-e2e \
  -e AWS_SECRET_ACCESS_KEY=datamaite-e2e-secret \
  ghcr.io/chrislusf/seaweedfs:4.46@sha256:08d516132314207d10c8e37cbffc1f32b147d870169688734cc61c6231625b62

export DATAMAITE_S3_E2E_ENDPOINT=http://127.0.0.1:9123
export DATAMAITE_S3_E2E_KEY=datamaite-e2e
export DATAMAITE_S3_E2E_SECRET=datamaite-e2e-secret
uv run pytest tests/e2e/test_s3_compatible.py -m integration --no-cov -v

docker stop datamaite-seaweedfs-e2e
```

CI runs this tier in its own `e2e-s3` job (in the `test` stage, concurrent with the matrix),
against a SeaweedFS service container, with coverage disabled — it is deliberately
kept out of the coverage-gated `test` matrix.

`TestExampleDatasetParity` (in the same file) additionally mirrors the shared
example-data repo's `hmie/valid` dataset into the SeaweedFS bucket and asserts
`validate()`/`load_mot()` give identical results locally and over S3; it needs
`DATAMAITE_DATASETS_ROOT` pointing at an example-datasets checkout and skips
cleanly without it.

## Adding tests

- New unit test → `tests/test_<thing>.py`, no marker. Must run offline.
- New e2e test → `tests/e2e/test_<thing>.py`, `pytestmark = pytest.mark.integration`,
  and skip cleanly when its external dependency is unavailable.
