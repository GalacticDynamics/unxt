"""Units objects in JAX.

Copyright (c) 2023 Galactic Dynamics. All rights reserved.
"""

__all__ = [
    "units",
    "units_of",
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


@dispatch
def units_of(obj: Any, /) -> AbstractUnits:
    """Return the units of an object.

    Examples
    --------
    >>> from unxt import units_of
    >>> units_of(1)
    Unit(dimensionless)

    """
    return u.dimensionless_unscaled


@dispatch
def units_of(obj: AbstractUnits, /) -> AbstractUnits:
    """Return the units of an unit.

    Examples
    --------
    >>> from unxt import units, units_of
    >>> m = units("m")

    >>> units_of(m)
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
    >>> from unxt import dimensions_of
    >>> dimensions_of(u.km)
    PhysicalType('length')

    """
    return obj.physical_type
