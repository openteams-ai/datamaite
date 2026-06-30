# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "datamaite"
copyright = "2026, OpenTeams"  # noqa: A001
author = "Datamaite Team"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",  # replaces myst_parser; handles .md and .ipynb
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.mermaid",  # render ```mermaid fences as diagrams
    "sphinx_design",  # grid/card "tiles" for the landing page
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "README.md",  # top-level docs/README.md (build instructions, not a page)
    "**/datamaite-example-datasets",  # cloned example-datasets repo (docs/tutorials/), not part of the docs
    "**/output",  # generated notebook outputs (reports, converted datasets)
    "jupyter_execute",  # MyST-NB build artifact; excluding stops a re-read loop
    "**/.ipynb_checkpoints",
]

# MyST-NB: enable colon-fence syntax and auto-generate heading anchors
myst_enable_extensions = ["colon_fence", "attrs_inline"]
myst_heading_anchors = 3
# Treat ```mermaid fences as the mermaid directive (renders diagrams; also keeps
# the blocks GitHub-compatible). Requires the sphinxcontrib.mermaid extension.
myst_fence_as_directive = ["mermaid"]

# Notebook execution
# Execute every notebook with the kernel available in the current (Poetry)
# environment instead of the kernel name baked into the notebook metadata
# (e.g. "nebi-fmv-viewer-default"). The regex matches any saved kernel name
# and remaps it to the local "python3" kernel provided by ipykernel.
nb_kernel_rgx_aliases = {".*": "python3"}
nb_execution_mode = "cache"
nb_execution_cache_path = "_build/.jupyter_cache"
nb_execution_raise_on_error = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
# Widen the main content column (see _static/custom.css).
html_css_files = ["custom.css"]
# Publish the committed example validation report verbatim into the build output
# (it is a self-contained HTML page, not a Sphinx source document). It is copied
# to the build root and reachable at /example-validation-report.html.
html_extra_path = ["tutorials/example-validation-report.html"]

html_theme_options = {
    "show_toc_level": 2,
    "logo": {
        "text": "Datamaite",
    },
}
