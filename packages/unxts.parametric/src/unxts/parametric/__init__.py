"""Dimension-parametrized quantities for unxt (canonical: unxts.parametric)."""

__all__ = ["AbstractParametricQuantity", "ParametricQuantity", "PQ"]

from ._src.base_parametric import AbstractParametricQuantity
from ._src.parametric import PQ, ParametricQuantity

# Import register modules for their dispatch/promotion side effects.
from ._src import (  # noqa: F401
    register_api,
    register_conversions,
    register_promotions,
)
