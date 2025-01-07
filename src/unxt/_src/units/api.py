"""Units objects in JAX.

Copyright (c) 2023 Galactic Dynamics. All rights reserved.
"""

__all__ = ["unit", "unit_of"]

from typing import Any, TypeAlias

import astropy.units as apyu
from plum import dispatch

AbstractUnits: TypeAlias = apyu.UnitBase | apyu.Unit


@dispatch.abstract
def unit(obj: Any, /) -> AbstractUnits:
    """Construct the units from a units object."""


@dispatch.abstract
def unit_of(obj: Any, /) -> AbstractUnits:
    """Return the units of an object."""
