"""Quantities in JAX.

This module provides a way to work with quantities in JAX.

The main features are:

- ``unxt.quantity.Quantity``: A class representing a quantity with units.
- ``unxt.quantity.uconvert``: A function to convert a quantity to a different
  unit.
- ``unxt.quantity.ustrip``: A function to strip the units from a quantity.

"""
# ruff:noqa: F403

from ._src.quantity.api import is_unit_convertible, uconvert, ustrip
from ._src.quantity.base import AbstractQuantity
from ._src.quantity.base_parametric import AbstractParametricQuantity
from ._src.quantity.quantity import Quantity
from ._src.quantity.unchecked import UncheckedQuantity

# isort: split
# Register dispatches and conversions
from ._src.quantity import (
    register_api,
    register_conversions,
    register_dispatches,
    register_primitives,
)

__all__: list[str] = [
    # Core
    "Quantity",
    # Base
    "AbstractQuantity",
    # Fast
    "UncheckedQuantity",
    # Base Parametric
    "AbstractParametricQuantity",
    # Functional
    "uconvert",
    "ustrip",
    "is_unit_convertible",
]


# Clean up namespace
del register_conversions, register_api, register_dispatches, register_primitives
