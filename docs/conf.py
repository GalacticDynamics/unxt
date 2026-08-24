"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import pytz

from unxt import __version__

# Add package docs directories to the source paths
_docs_dir = Path(__file__).parent
_repo_root = _docs_dir.parent
sys.path.insert(0, str(_repo_root / "packages" / "unxt-api" / "docs"))
sys.path.insert(0, str(_repo_root / "packages" / "unxt-hypothesis" / "docs"))

# -- Project information -----------------------------------------------------

author = "Unxt Developers"
project = "unxt"
copyright = f"{datetime.now(pytz.timezone('UTC')).year}, {author}"
version = __version__

master_doc = "index"
language = "en"

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_nb",  # Jupyter notebook support via MyST (includes myst_parser functionality)
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
    "sphinxext.rediraffe",  # Add redirects
    "sphinx_togglebutton",
    "sphinx_tippy",
]

python_use_unqualified_type_names = True

# -- Redirects (rediraffe) ---------------------------------------------------

# The docs were reorganised into Diataxis sections; every pre-existing URL is
# redirected to its new home so external links keep working.
rediraffe_redirects = {
    "guides/quantity.md": "reference/quantity.md",
    "guides/dimensions.md": "reference/dimensions.md",
    "guides/units_and_systems.md": "reference/units.md",
    "guides/configuration.md": "reference/configuration.md",
    "guides/natural-units.md": "how-to/work-in-natural-units.md",
    "guides/type-checking.md": "how-to/check-types-at-runtime.md",
    "guides/perf.md": "how-to/optimize-performance.md",
    "guides/sharp-bits.md": "explanation/sharp-bits.md",
    "conventions.md": "explanation/api-conventions.md",
    "glossary.md": "reference/glossary.md",
    "migration.md": "how-to/migrate-to-v2.md",
    "contributing.md": "about/contributing.md",
    "dev.md": "about/contributing.md",
    "interop/astropy.md": "how-to/interoperate-with-astropy.md",
    "interop/dataclassish.md": "reference/dataclassish.md",
    "api/index.md": "reference/api/index.md",
    "api/quantity.md": "reference/api/quantity.md",
    "api/dims.md": "reference/api/dims.md",
    "api/units.md": "reference/api/units.md",
    "api/unitsystems.md": "reference/api/unitsystems.md",
    "api/experimental.md": "reference/api/experimental.md",
    # Package doc sets, reorganised in the same pass.
    "packages/unxts.parametric/dimensions.md": "packages/unxts.parametric/quantity.md",
}

exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
    ".env",
    ".venv",
    # Excluded: converted to optimize-performance.ipynb during the build
    "how-to/optimize-performance.md",
    "**/_data/**",  # Sample-data dirs (e.g. the xarray guide's), not doc pages
]

# Include package documentation directories
html_extra_path = [
    str(p) for p in (_repo_root / "packages").glob("*/docs") if p.is_dir()
]

source_suffix = [".md", ".rst", ".ipynb"]

_docs_path = Path(__file__).parent
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "jaxtyping": (
        "https://docs.kidger.site/jaxtyping/",
        str(_docs_path / "_static" / "jaxtyping.inv"),
    ),
    "equinox": (
        "https://docs.kidger.site/equinox/",
        str(_docs_path / "_static" / "equinox.inv"),
    ),
    # quax-blocks doesn't have hosted docs; links point to source code on GitHub
    "quax_blocks": (
        "https://github.com/GalacticDynamics/quax-blocks/",
        str(_docs_path / "_static" / "quax_blocks.inv"),
    ),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "quax": ("https://docs.kidger.site/quax/", None),
}

# -- Napoleon settings ---------------------------------------------------

napoleon_use_math = True

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
typehints_fully_qualified = False
simplify_optional_unions = True

# Map unqualified type names to fully qualified names for intersphinx resolution
autodoc_type_aliases = {
    "ndarray": "numpy.ndarray",
    "numpy.typing.ndarray": "numpy.ndarray",
    "numpy.ndarray": "numpy.ndarray",
    "NDArray": "numpy.typing.NDArray",
    "ArrayLike": "jaxtyping.ArrayLike",
    "jax._src.literals.TypedNdArray": "jaxtyping.TypedNdArray",
    "AbstractUnit": "unxt.AbstractUnit",
}


# -- Nitpick ignore patterns -------------------------------------------------

# Jaxtyping creates many shape-annotated types like Float[Array, "N 3"] that
# appear as class/data references. Instead of listing every combination, use
# regex patterns to ignore all shape annotations.

# Match single shape tokens: F (feature), N (batch), S (sequence), 1, 2, or "..."
_SHAPE_NAME_RE: Final[str] = r"^(?:F|N|S|1|2|\.\.\.)$"

# Match space-separated shape tuples like "N 3", "... 4", "F N S"
_SHAPE_TUPLE_RE: Final[str] = r"^(?:F|N|S|\d+|\.\.\.)(?:\s+(?:F|N|S|\d+|\.\.\.))*$"

nitpick_ignore_regex: Final[list[tuple[str, str]]] = [
    ("py:class", _SHAPE_TUPLE_RE),
    ("py:data", _SHAPE_TUPLE_RE),
    ("py:class", _SHAPE_NAME_RE),
    ("py:data", _SHAPE_NAME_RE),
]

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
    ("py:class", "NoneType"),
    ("py:class", "quax._core.ArrayValue"),
    ("py:class", "PhysicalType"),
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
myst_heading_anchors = 4

myst_substitutions = {
    "ArrayLike": ":obj:`jaxtyping.ArrayLike`",
    "Any": ":obj:`typing.Any`",
}

# -- MyST-NB settings (Jupyter notebook support) --------------------------

# Execute notebooks during build
nb_execution_mode = "cache"
# Store execution cache in _build directory
nb_execution_cache_path = "_build/.jupyter_cache"
# Raise errors if notebook execution fails
nb_execution_raise_on_error = True
# Timeout for each notebook cell (in seconds)
nb_execution_timeout = 100
# Add download links for notebooks
nb_html_download = True
# Enable .ipynb download specifically
nb_ipynb_download = True

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
            "name": "JOSS",
            "url": "https://doi.org/10.21105/joss.07771",
            "icon": "fa fa-quote-right",
        },
    ],
}
