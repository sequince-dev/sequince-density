"""Sphinx configuration for the sequince-density documentation."""

from __future__ import annotations

from datetime import datetime
from importlib import metadata

project = "sequince-density"
copyright = f"{datetime.now():%Y}, Sequince Developers"
author = "Sequince Developers"

try:
    release = metadata.version("sequince-density")
except metadata.PackageNotFoundError:
    release = "dev"
version = release

extensions = [
    "autoapi.extension",
    "myst_parser",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

autodoc_typehints = "description"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_ivar = True

# Generate API pages without importing optional density backends.
autoapi_dirs = ["../src/sequince_density"]
autoapi_add_toctree_entry = False
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "sequince": ("https://sequince.readthedocs.io/en/latest/", None),
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/sequince-dev/sequince-density",
    "use_repository_button": True,
    "use_fullscreen_button": False,
}
