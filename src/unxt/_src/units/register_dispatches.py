"""Units objects in JAX.

Copyright (c) 2023 Galactic Dynamics. All rights reserved.
"""

__all__: list[str] = []

from typing import Any

import astropy.units as apyu
from plum import dispatch

from .api import AstropyUnits
from unxt.dims import AbstractDimension

# ===================================================================
# Construct units


@dispatch
def unit(obj: AstropyUnits, /) -> AstropyUnits:
    """Construct the units from a units object.

    Examples
    --------
    >>> import unxt as u
    >>> m = u.unit("m")

    >>> u.unit(m) is m
    True

    """
    return obj


@dispatch
def unit(obj: str, /) -> AstropyUnits:
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
def unit_of(obj: AstropyUnits, /) -> AstropyUnits:
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


@dispatch
def dimension_of(obj: AstropyUnits, /) -> AbstractDimension:
    """Return the dimensions of the given units.

    Examples
    --------
    >>> import unxt as u
    >>> u.dimension_of(u.unit("km"))
    PhysicalType('length')

    """
    return obj.physical_type
