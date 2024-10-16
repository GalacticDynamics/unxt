"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

# ```{include} ../README.md
# :start-after: <!-- SPHINX-START -->
# ```

import importlib.metadata
import sys
from pathlib import Path
from typing import Any

import tomli

this_dir = Path(__file__).resolve().parent

sys.path.append(str(this_dir.parent / "src"))


# -- Project information -----------------------------------------------------


def read_pyproject() -> tuple[str, str]:
    """Get author information from package metadata."""
    with (this_dir.parent / "pyproject.toml").open("rb") as f:
        toml = tomli.load(f)

    project = dict(toml["project"])  # pylint: disable=W0621

    name = project["name"]
    authors = project["authors"][0]["name"]

    return name, authors


project, author = read_pyproject()

copyright = f"2024, {author}"

version = release = importlib.metadata.version(project)


# -- General configuration ---------------------------------------------------

# By default, highlight as Python 3.
highlight_language = "python3"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_design",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "inherited-members": True,
    "show-inheritance": True,
}

exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
    ".env",
    ".venv",
]

source_suffix = [".md"]


myst_enable_extensions = [
    "colon_fence",
    "dollarmath",  # for $, $$
    "amsmath",  # for direct LaTeX math
    "deflist",
]
myst_heading_anchors = 3

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "jaxtyping": ("https://docs.kidger.site/jaxtyping/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "quax": ("https://docs.kidger.site/quax/", None),
}

nitpick_ignore = [
    ("py:class", "_io.StringIO"),
    ("py:class", "_io.BytesIO"),
]

always_document_param_types = True


# -- Options for HTML output -------------------------------------------------

html_theme = "furo"

html_title = f"{project} v{release}"

html_static_path = ["_static"]

html_theme_options: dict[str, Any] = {
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/org/test",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,  # noqa: E501
            "class": "",
        },
    ],
    "source_repository": "https://github.com/org/test",
    "source_branch": "main",
    "source_directory": "docs/",
}
