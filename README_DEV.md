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

## Releasing (#85)

The git tag is the single source of truth for the version (uv-dynamic-versioning);
there is no version-bump commit. A release is three steps:

1. **Changelog MR** — move the `## [Unreleased]` content into a new
   `## [X.Y.Z] - YYYY-MM-DD` section and merge it. The tag pipeline's
   `changelog-check` job fails the release if this section is missing.
2. **Push the tag** — `git tag X.Y.Z <main-sha> && git push origin X.Y.Z`
   (no `v` prefix). The tag pipeline re-runs full validation, then publishes
   to TestPyPI automatically, with post-upload digest verification.
3. **Approve `publish-pypi`** — the one manual gate, in the same tag pipeline.
   After it verifies, `gitlab-release` creates the GitLab Release with the
   changelog section as its notes.

Every publish step is idempotent (`--skip-existing` + digest comparison), so
re-running a failed or interrupted publish job is safe — retrying the failed
job is the sanctioned move for infra or index flakes. Only release-shaped tags
(`X.Y.Z`, optionally `-rcN`/`-alphaN`/`-betaN`/`.postN`/`.devN`) start a
pipeline at all, and a wrong version derivation fails the build before
anything uploads.

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
