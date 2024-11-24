"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

from datetime import datetime
from typing import Any

import pytz

from unxt import __version__

# -- Project information -----------------------------------------------------

author = "Unxt Developers"
project = "unxt"
copyright = f"{datetime.now(pytz.timezone('UTC')).year}, {author}"
version = __version__

master_doc = "index"
language = "en"

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",  # General MyST markdown support
    "sphinx_design",
    "sphinx.ext.autodoc",  # TODO: replace with autodoc2
    "sphinx.ext.autosummary",  # TODO: replace with autodoc2
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx-prompt",
    "sphinxext.opengraph",
    # "sphinxext.rediraffe",  # Add redirects
    "sphinx_togglebutton",
    "sphinx_tippy",
]

python_use_unqualified_type_names = True

exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
    ".env",
    ".venv",
]

source_suffix = [".md", ".rst"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "jaxtyping": ("https://docs.kidger.site/jaxtyping/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "quax": ("https://docs.kidger.site/quax/", None),
}

# -- Autodoc settings ---------------------------------------------------

autodoc_typehints = "description"
autodoc_typehints_format = "short"

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "inherited-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}

always_document_param_types = True
typehints_use_signature = True


nitpick_ignore = [
    ("py:class", "_io.StringIO"),
    ("py:class", "_io.BytesIO"),
    # TODO: fix these
    ("py:class", "P"),  # ParamSpec alias
    ("py:class", "unxt._src.experimental.R"),  # TypeVar alias
    ("py:class", "unxt._src.quantity.base._QuantityIndexUpdateHelper"),
    ("py:class", "unxt._src.quantity.mixins.AstropyQuantityCompatMixin"),
    ("py:class", "unxt._src.quantity.mixins.IPythonReprMixin"),
    ("py:class", "unxt._src.utils.SingletonMixin"),
    ("py:class", "ArrayLike"),
    ("py:class", "NoneType"),
    ("py:class", "quax._core.ArrayValue"),
    ("py:class", "PhysicalType"),
    ("py:class", "jaxtyping.Shaped[Array, '*shape']"),
    ("py:class", "astropy.units.core.Annotated"),
]

# -- MyST Setting -------------------------------------------------

myst_enable_extensions = [
    "amsmath",  # for direct LaTeX math
    "attrs_block",  # enable parsing of block attributes
    "attrs_inline",  # apply syntax highlighting to inline code
    "colon_fence",
    "deflist",
    "dollarmath",  # for $, $$
    # "linkify",  # identify “bare” web URLs and add hyperlinks:
    "smartquotes",  # convert straight quotes to curly quotes
    "substitution",  # substitution definitions
]
myst_heading_anchors = 3

# myst_substitutions = {
#     "ArrayLike": ":obj:`jaxtyping.ArrayLike`",
#     "Any": ":obj:`typing.Any`",
# }


# -- HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"
html_title = "unxt"
html_logo = "_static/favicon.png"  # TODO: an svg
html_copy_source = True
html_favicon = "_static/favicon.png"

html_static_path = ["_static"]
html_css_files = ["custom_toc.css", "custom_tooltip.css"]

html_theme_options: dict[str, Any] = {
    "home_page_in_toc": True,
    "repository_url": "https://github.com/GalacticDynamics/unxt",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_edit_page_button": False,
    "use_issues_button": True,
    "show_toc_level": 2,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/GalacticDynamics/unxt",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/unxt/",
            "icon": "https://img.shields.io/pypi/v/unxt",
            "type": "url",
        },
        {
            "name": "Zenodo",
            "url": "https://doi.org/10.5281/zenodo.10850455",
            "icon": "fa fa-quote-right",
        },
    ],
}
