## Documentation (HTML)

Sphinx source lives in `docs/`. Output goes to `public/` (git-ignored).

```bash
# Install docs dependencies
poetry install --extras docs

# Navigate to the docs directory
cd docs

# Build HTML
poetry run sphinx-build -b html . public
```

Open `docs/public/index.html` to preview locally.

### Live reload

For local editing, `sphinx-autobuild` watches the source, rebuilds on save, and
live-reloads the browser. Run from the `docs/` directory:

```bash
poetry run sphinx-autobuild -b html . public --open-browser
```

It serves at <http://127.0.0.1:8000>. Note that notebook execution uses
`nb_execution_mode = "cache"`, so only notebooks whose content changed are
re-executed on rebuild.
