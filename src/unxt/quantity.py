"""unxt: Quantities in JAX.

Copyright (c) 2023 Galactic Dynamics. All rights reserved.
"""
# ruff:noqa: F403

from ._src.quantity.api import uconvert, ustrip
from ._src.quantity.base import AbstractQuantity, is_unit_convertible
from ._src.quantity.base_parametric import AbstractParametricQuantity
from ._src.quantity.core import Quantity
from ._src.quantity.fast import UncheckedQuantity

# isort: split
# Register dispatches and conversions
from ._src.quantity import compat, functional, register_dispatches, register_primitives

__all__: list[str] = [
    # Base
    "AbstractQuantity",
    # Fast
    "UncheckedQuantity",
    # Base Parametric
    "AbstractParametricQuantity",
    # Core
    "Quantity",
    # Functional
    "uconvert",
    "ustrip",
    "is_unit_convertible",
]


# Clean up namespace
del compat, functional, register_dispatches, register_primitives
