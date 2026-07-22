"""Dimension-parametrized quantities for unxt (canonical: unxts.parametric)."""

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
    register_astropy,
    register_conversions,
    register_primitives,
    register_promotions,
)
from ._src.base_parametric import AbstractParametricQuantity
from ._src.config import config
from ._src.parametric import PQ, ParametricQuantity
from ._version import version as __version__
