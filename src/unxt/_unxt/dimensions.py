"""Units objects in JAX.

Copyright (c) 2023 Galactic Dynamics. All rights reserved.
"""

__all__ = ["dimensions", "dimensions_of"]

from typing import TypeAlias

import astropy.units as u
from astropy.units import Unit
from plum import dispatch

AbstractUnits: TypeAlias = u.UnitBase | Unit
AbstractDimensions: TypeAlias = u.PhysicalType


# ===================================================================
# Construct the dimensions


@dispatch
def dimensions(obj: AbstractDimensions, /) -> AbstractDimensions:
    """Return the dimensions of the given units.

    Examples
    --------
    >>> from astropy import units as u
    >>> from unxt import dimensions

    >>> length = u.get_physical_type("length")
    >>> length
    PhysicalType('length')

    >>> dimensions(length) is length
    True

    """
    return obj


@dispatch
def dimensions(obj: str, /) -> AbstractDimensions:
    """Construct dimensions from a string.

    Examples
    --------
    >>> from unxt import dimensions
    >>> dimensions("length")
    PhysicalType('length')

    """
    return u.get_physical_type(obj)


# ===================================================================
# Get the dimensions


@dispatch  # type: ignore[misc]
def dimensions_of(obj: AbstractDimensions, /) -> AbstractDimensions:
    """Return the dimensions of the given units.

    Examples
    --------
    >>> from astropy import units as u
    >>> from unxt import dimensions_of

    >>> dimensions_of(u.get_physical_type("length"))
    PhysicalType('length')

    """
    return obj
