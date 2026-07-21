"""Units objects in JAX.

Copyright (c) 2023 Galactic Dynamics. All rights reserved.
"""

__all__ = ("unit", "unit_of", "AbstractUnit")

from typing import Any, TypeAlias

import astropy.units as apyu
from plum import dispatch

import unxt_api as uapi
from unxt.dims import AbstractDimension

# ``FunctionUnitBase`` (mag/dex/dB) is a separate hierarchy from ``UnitBase``, so
# it is listed explicitly. ``StructuredUnit`` is intentionally excluded: it has
# no single dimension, so ``dimension_of`` cannot handle it.
AbstractUnit: TypeAlias = apyu.UnitBase | apyu.FunctionUnitBase


# ===================================================================
# Construct units


@dispatch
def unit(obj: AbstractUnit, /) -> AbstractUnit:
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
def unit(obj: str, /) -> AbstractUnit:
    """Construct units from a string.

    Examples
    --------
    >>> import unxt as u
    >>> m = u.unit("m")
    >>> m
    Unit("m")

    Astropy function units (magnitudes, dex, decibels) are also supported:

    >>> u.unit("mag(AB)")
    Unit("mag(AB)")

    >>> u.unit("dex(cm/s2)")
    Unit("dex(cm / s2)")

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
def unit_of(obj: AbstractUnit, /) -> AbstractUnit:
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
def dimension_of(obj: AbstractUnit, /) -> AbstractDimension:
    """Return the dimensions of the given units.

    Examples
    --------
    >>> import unxt as u
    >>> u.dimension_of(u.unit("km"))
    PhysicalType('length')

    """
    return uapi.dimension(obj.physical_type)
