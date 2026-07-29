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
re-running a failed or interrupted pipeline is always safe. A `.devN`/`+dirty`
version derivation fails the build before anything uploads.
