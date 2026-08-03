# Alternative Package Managers

The primary workflow uses **Poetry** (required for CI). Two alternatives are
available for local development:

## uv (fast pip-based)

```bash
uv sync --all-extras
uv run pytest
uv run pre-commit run --all-files
uv run pyright src/
```

## pixi (conda-forge based)

Useful on machines where pip-installing opencv is difficult (e.g., SUNet).

```bash
pixi run install      # editable install into conda env
pixi run test         # pytest
pixi run lint         # pre-commit
pixi run typecheck    # pyright
pixi run check        # all of the above
```

Configuration lives in `pixi.toml` (separate from `pyproject.toml`).

## `datamaite.__version__` in development environments

The version is derived from the git tag at build time and read back from the
installed metadata at runtime. `poetry install` is the one path that bypasses
the build backend: Poetry's develop-install registers the literal
`[tool.poetry].version` placeholder (`0.0.0`) instead of building through
hatchling + uv-dynamic-versioning. `_version.py` detects the placeholder and
derives the version from git via dunamai (part of the `dev` extra), so a
`poetry install --extras dev` environment still reports the SCM version. In a
Poetry environment **without** the dev extra, `datamaite.__version__` reports
`0.0.0`; reinstall the root project through the backend if that matters:

```bash
poetry run pip install --no-deps -e .
```

(The docs CI jobs do exactly this so executed notebooks don't print `0.0.0`.
Note that a pip editable install bakes the version at install time — it goes
stale as commits/tags land until you reinstall.)

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
