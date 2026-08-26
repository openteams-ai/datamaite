# Development Workflow

The primary workflow uses **uv**, which is also what CI runs — a green local
run is the same resolution CI gets.

```bash
uv sync --all-extras
uv run pytest
uv run pre-commit run --all-files
uv run pyright src/
uv build
```

`uv.lock` is the lock file of record. CI verifies it is in sync with
`pyproject.toml` (`uv lock --check`), so run `uv lock` and commit the result
whenever you change a dependency.

A dependency change also needs the generated `requirements.txt` refreshed — it
is the DR-compliance scanner's only readable input, since that component does
not parse `uv.lock`. The `lint` job fails on drift:

```bash
scripts/export-requirements.sh
```

### Where dependencies are declared

One place, with two files derived from it:

| File | Role | Who maintains it |
|---|---|---|
| `pyproject.toml` `[project]` / `[project.optional-dependencies]` | **Source of truth** for runtime deps and every extra | edit by hand |
| `uv.lock` | Lock file of record; pins the resolved graph with hashes | `uv lock` |
| `requirements.txt` | Generated projection for DR-compliance dependency scanning only — never an install path | `scripts/export-requirements.sh` |

So: add or change a dependency in `pyproject.toml`, then run `uv lock` and
`scripts/export-requirements.sh` and commit all three. pip consumes the same
`pyproject.toml` metadata directly, so there is nothing extra to update for it.

## `datamaite.__version__` in development environments

The version is derived from the git tag at build time and read back from the
installed metadata at runtime. `uv sync` installs the root project through
hatchling + uv-dynamic-versioning, so an ordinary development environment
reports the SCM-derived version with no extra step.

A clone without tags still gets a real version, just an unflattering one
(`0.0.0.postN.devN+<sha>` — dunamai's no-tag serialization, not a `0.0.0`
release). The literal `0.0.0` fallback appears only when the backend cannot use
git at all, such as building from a tarball export.

An editable install bakes the version at install time, so it would go stale as
commits and tags land. `[tool.uv] cache-keys` in `pyproject.toml` declares the
git commit and tags as cache keys for exactly that reason, so `uv sync` rebuilds
the root when either moves — without it, `uv sync` reuses the cached editable
wheel and reports a stale version even after you delete `.venv`.

## Releasing (#85)

The git tag is the single source of truth for the version (uv-dynamic-versioning);
there is no version-bump commit. A release is three steps:

1. **Changelog MR** — move the `## [Unreleased]` content into a new
   `## [X.Y.Z] - YYYY-MM-DD` section and merge it. The tag pipeline's
   `changelog-check` job fails the release if this section is missing.
2. **Push the tag** — `git tag X.Y.Z <main-sha> && git push origin X.Y.Z`
   (no `v` prefix, no leading zeros). The tag pipeline re-runs full
   validation, then publishes to TestPyPI automatically, with post-upload
   digest verification.
3. **Approve `publish-pypi`** — the one manual gate, in the same tag pipeline.
   After it verifies, `gitlab-release` creates the GitLab Release with the
   changelog section as its notes.

No conda step yet: once a conda-forge feedstock exists, the autotick bot will
track PyPI and this repo will need no extra release step.

Every publish step is idempotent (`--skip-existing` + digest comparison), so
re-running a failed or interrupted publish job is safe — retrying the failed
job is the sanctioned move for infra or index flakes. Only release-shaped tags
(`X.Y.Z`, optionally `-rcN`/`-alphaN`/`-betaN`/`.postN`/`.devN`) start a
pipeline at all, and a wrong version derivation fails the build before
anything uploads. Tags must be canonically spelled — leading zeros
(`01.02.03`, `0.5.0-rc01`) are rejected, because PEP 440 would canonicalize
the built artifacts to a *different* spelling (`1.2.3`) than the
tag/changelog/Release record, and that spelling could collide with a later
genuine `1.2.3`.

### Recovery runbook

- **Tagged too early / wrong SHA, nothing published yet** (changelog-check or
  tests failed — publishing never ran): protected tags cannot be deleted while
  protected. Order matters: Settings → Repository → unprotect `*.*.*` →
  `git push --delete origin X.Y.Z` → **re-protect `*.*.*`** → re-tag. If you
  re-tag before re-protecting, the pipeline runs on an unprotected ref, the
  PyPI tokens don't flow, and publish fails with a token error.
- **Wrong content already on TestPyPI** (auto-publish ran before the mistake
  was caught): that version's filenames are burned on TestPyPI forever — the
  index never allows filename reuse, so the corrected build's digest check
  will fail permanently. Bump the patch version and release that instead.
  PyPI itself is untouched (the manual gate never ran).
- **Digest mismatch on PyPI** (partial upload, then a differing rebuild):
  never hand-upload over it. Yank the bad file on PyPI if it is wrong, bump
  the patch version, release again. The publish toolchain and the PEP 517
  build env are pinned (`scripts/publish-build-constraints.txt`) precisely so
  retry rebuilds stay byte-identical and this stays rare.
- **`gitlab-release` failed after PyPI succeeded**: retry just that job as a
  Maintainer, or create the Release by hand — the notes are
  `python scripts/release_notes.py extract X.Y.Z`.
- **Abandoned release** (tag pushed, PyPI gate never approved): cancel the
  blocked pipeline. GitLab's "prevent outdated deployment jobs" setting stops
  a stale gate from publishing an older version after a newer one shipped; a
  deliberate out-of-order/backport release needs a fresh pipeline run on the
  tag (CI/CD → Pipelines → Run pipeline → select the tag).
