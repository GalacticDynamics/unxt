"""unxt: Quantities in JAX.

Copyright (c) 2023 Galactic Dynamics. All rights reserved.
"""
# ruff:noqa: F403

from ._src.quantity.api import uconvert, ustrip
from ._src.quantity.base import AbstractQuantity, can_convert_unit
from ._src.quantity.base_parametric import AbstractParametricQuantity
from ._src.quantity.core import Quantity
from ._src.quantity.distance import (
    AbstractDistance,
    Distance,
    DistanceModulus,
    Parallax,
)
from ._src.quantity.fast import UncheckedQuantity

# isort: split
# Register dispatches and conversions
from ._src.quantity import compat, functional, register_dispatches, register_primitives

__all__: list[str] = [
    # Base
    "AbstractQuantity",
    "can_convert_unit",
    # Fast
    "UncheckedQuantity",
    # Base Parametric
    "AbstractParametricQuantity",
    # Core
    "Quantity",
    # Distance
    "AbstractDistance",
    "Distance",
    "Parallax",
    "DistanceModulus",
    # Functional
    "uconvert",
    "ustrip",
]


# Clean up namespace
del compat, functional, register_dispatches, register_primitives
