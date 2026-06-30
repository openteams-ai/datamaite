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

## Documentation (HTML)

Sphinx source lives in `docs/`. Output goes to `public/` (git-ignored).

```bash
# Install docs dependencies
poetry install --extras docs

# Build HTML
poetry run sphinx-build -b html docs public
```

Open `public/index.html` to preview locally.
