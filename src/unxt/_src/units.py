"""Units objects in JAX.

Copyright (c) 2023 Galactic Dynamics. All rights reserved.
"""

__all__ = ("unit", "unit_of", "AbstractUnit")

from typing import TYPE_CHECKING, Any, TypeAlias, cast

import astropy.units as apyu
from plum import dispatch

import unxt_api as uapi
from unxt.dims import AbstractDimension

if TYPE_CHECKING:
    from collections.abc import Callable

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


def parse_unit(obj: AbstractUnit | str, /) -> AbstractUnit:
    """Typed converter for the ``unit`` field of the quantity classes.

    ``unit`` is a `plum` dispatch, which a static type checker (pyright) reads as
    only its last registered overload. Using it directly as an
    ``eqx.field(converter=...)`` therefore leaves the generated ``__init__``
    typing the ``unit`` argument as the field type ``AbstractUnit`` -- so
    ``Quantity(1, "m")`` is a type error, even though the string is accepted at
    runtime. This thin wrapper gives the checker a readable ``AbstractUnit | str``
    parameter while the field still reads back as ``AbstractUnit``. It delegates
    to ``unit``; the ``cast`` restores the real (union-accepting) signature that
    the checker cannot see through the dispatch.
    """
    construct = cast("Callable[[AbstractUnit | str], AbstractUnit]", unit)
    return construct(obj)


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
