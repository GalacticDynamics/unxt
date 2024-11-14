"""Units objects in JAX.

Copyright (c) 2023 Galactic Dynamics. All rights reserved.
"""

__all__ = [
    "units",
    "unit_of",
    "Unit",  # legacy
]

from typing import Any, TypeAlias

import astropy.units as u
from astropy.units import Unit
from plum import dispatch

AbstractUnits: TypeAlias = u.UnitBase | Unit

# ===================================================================
# Construct units


@dispatch
def units(obj: AbstractUnits, /) -> AbstractUnits:
    """Construct the units from a units object.

    Examples
    --------
    >>> from unxt import units
    >>> m = units("m")

    >>> units(m) is m
    True

    """
    return obj


@dispatch  # type: ignore[no-redef]
def units(obj: str, /) -> AbstractUnits:
    """Construct units from a string.

    Examples
    --------
    >>> from unxt import units
    >>> m = units("m")
    >>> m
    Unit("m")

    """
    return u.Unit(obj)


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
    >>> from unxt import unit_of
    >>> print(unit_of(1))
    None

    """
    return None  # noqa: RET501


@dispatch
def unit_of(obj: AbstractUnits, /) -> AbstractUnits:
    """Return the units of an unit.

    Examples
    --------
    >>> from unxt import units, unit_of
    >>> m = units("m")

    >>> unit_of(m)
    Unit("m")

    """
    return obj


# ===================================================================
# Get dimensions


@dispatch  # type: ignore[misc]
def dimensions_of(obj: AbstractUnits, /) -> u.PhysicalType:
    """Return the dimensions of the given units.

    Examples
    --------
    >>> from unxt import units
    >>> from unxt.dims import dimensions_of
    >>> dimensions_of(units("km"))
    PhysicalType('length')

    """
    return obj.physical_type
