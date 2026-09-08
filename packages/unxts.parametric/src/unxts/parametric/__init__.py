"""Dimension-parametrized quantities for unxt (canonical: unxts.parametric)."""

# pylint: disable=duplicate-code
# The `__init__.pyi` stub next to this file redeclares (most of) this
# module's public surface for mypy; pylint's duplicate-code checker (R0801)
# doesn't know about the PEP 561 stub relationship and flags the `.pyi` as a
# self-duplicate of this file's `__all__` tuple.

__all__ = (
    "__version__",
    "AbstractParametricQuantity",
    "ParametricQuantity",
    "PQ",
    "config",
)

# Import register modules for their dispatch/promotion side effects.
from ._src import (  # noqa: F401
    register_api,
    register_conversions,
    register_primitives,
)
from ._src.base_parametric import AbstractParametricQuantity
from ._src.config import config
from ._src.parametric import PQ, ParametricQuantity
from ._version import version as __version__
