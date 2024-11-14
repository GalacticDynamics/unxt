"""Units objects in JAX.

Copyright (c) 2023 Galactic Dynamics. All rights reserved.
"""

__all__ = [
    "unit",
    "unit_of",
    "Unit",  # legacy
]

from typing import Any, TypeAlias

import astropy.units as apyu
from astropy.units import Unit
from plum import dispatch

from unxt._src.dimensions.core import AbstractDimension

AbstractUnits: TypeAlias = apyu.UnitBase | apyu.Unit

# ===================================================================
# Construct units


@dispatch
def unit(obj: AbstractUnits, /) -> AbstractUnits:
    """Construct the units from a units object.

    Examples
    --------
    >>> import unxt as u
    >>> m = u.unit("m")

    >>> u.unit(m) is m
    True

    """
    return obj


@dispatch  # type: ignore[no-redef]
def unit(obj: str, /) -> AbstractUnits:
    """Construct units from a string.

    Examples
    --------
    >>> import unxt as u
    >>> m = u.unit("m")
    >>> m
    Unit("m")

    """
    return apyu.Unit(obj)


# ===================================================================
# Get units


@dispatch.abstract
def unit_of(obj: Any, /) -> AbstractUnits:
    """Return the units of an object."""


@dispatch
def unit_of(obj: Any, /) -> None:
    """Return the units of an object.

    Examples
    --------
    >>> import unxt as u
    >>> print(u.unit_of(1))
    None

    """
    return None  # noqa: RET501


@dispatch
def unit_of(obj: AbstractUnits, /) -> AbstractUnits:
    """Return the units of an unit.

    Examples
    --------
    >>> import unxt as u
    >>> m = u.unit("m")

    >>> u.unit_of(m)
    Unit("m")

    """
    return obj


# ===================================================================
# Get dimensions


@dispatch  # type: ignore[misc]
def dimension_of(obj: AbstractUnits, /) -> AbstractDimension:
    """Return the dimensions of the given units.

    Examples
    --------
    >>> import unxt as u
    >>> u.dimension_of(u.unit("km"))
    PhysicalType('length')

    """
    return obj.physical_type
