"""Quantities in JAX.

This module provides a way to work with quantities in JAX.

The main features are:

- ``unxt.quantity.Quantity``: A class representing a quantity with units.
- ``unxt.quantity.uconvert``: A function to convert a quantity to a different
  unit.
- ``unxt.quantity.ustrip``: A function to strip the units from a quantity.

"""

__all__ = [
    # Core
    "Quantity",
    # Base
    "AbstractQuantity",
    # Fast
    "BareQuantity",
    "UncheckedQuantity",
    # Base Parametric
    "AbstractParametricQuantity",
    # Functional
    "uconvert",
    "ustrip",
    "is_unit_convertible",
    "is_any_quantity",
    "convert_to_quantity_value",
    "AllowValue",
]


from jaxtyping import install_import_hook

from .setup_package import RUNTIME_TYPECHECKER

with install_import_hook("unxt.quantity", RUNTIME_TYPECHECKER):
    from ._src.quantity.api import is_unit_convertible, uconvert, ustrip
    from ._src.quantity.base import AbstractQuantity, is_any_quantity
    from ._src.quantity.base_parametric import AbstractParametricQuantity
    from ._src.quantity.flag import AllowValue
    from ._src.quantity.quantity import Quantity
    from ._src.quantity.unchecked import BareQuantity, UncheckedQuantity
    from ._src.quantity.value import convert_to_quantity_value

    # isort: split
    # Register dispatches and conversions
    from ._src.quantity import (
        register_api,
        register_conversions,
        register_dispatches,
        register_primitives,
    )

# Clean up namespace
del register_conversions, register_api, register_dispatches, register_primitives
