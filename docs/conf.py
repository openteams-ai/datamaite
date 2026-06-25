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
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# MyST-NB: enable colon-fence syntax and auto-generate heading anchors
myst_enable_extensions = ["colon_fence"]
myst_heading_anchors = 3

# Notebook execution
nb_execution_mode = "cache"
nb_execution_cache_path = "_build/.jupyter_cache"
nb_execution_raise_on_error = True
# hmie.ipynb requires real HMIE datasets not available in CI
nb_execution_excludepatterns = ["tool-usage/validators/hmie.ipynb"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

html_theme_options = {
    "show_toc_level": 2,
    "logo": {
        "text": "Datamaite",
    },
}
