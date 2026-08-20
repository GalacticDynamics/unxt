"""Units objects in JAX.

Copyright (c) 2023 Galactic Dynamics. All rights reserved.
"""

__all__ = ("unit", "unit_of", "AbstractUnit")

import contextlib
from typing import Any, TypeAlias

import astropy.units as apyu
from plum import dispatch

import unxt_api as uapi
from unxt._src import fmt
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

    Astropy units are passed through unchanged:

    >>> import astropy.units as apyu
    >>> u.unit(apyu.km)
    Unit("km")

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


@dispatch
def unit(obj: apyu.Quantity, /) -> AbstractUnit:
    """Construct units from an Astropy quantity, folding the value into the unit.

    Examples
    --------
    >>> import astropy.units as apyu
    >>> import unxt as u
    >>> u.unit(apyu.Quantity(2, "km"))
    Unit("2 km")

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

    >>> import astropy.units as apyu
    >>> u.unit_of(apyu.km)
    Unit("km")

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


# ===================================================================
# String formatting


@fmt.pparts.dispatch  # type: ignore[misc]
def pparts(
    obj: AbstractUnit, /, *, markup: str = "text", unit: str = "symbol", **kw: Any
) -> tuple[Any, ...]:
    r"""Decompose a unit for the `unxt._fmt` engine.

    A unit is just an object with parts, so there is no separate unit renderer
    and the engine's nesting rule covers it.

    ``unit`` is the engine's unit axis: which *spelling* to use.

    Examples
    --------
    >>> import unxt as u
    >>> from unxt._fmt import pparts

    >>> pparts(u.unit("m"))
    (PPart(role='unit', text='m', kind='content'),)

    In LaTeX the fragment carries rendered markup, so it is not escaped again:

    >>> pparts(u.unit("m/s2"), markup="latex")
    (PPart(role='unit', text='\\mathrm{\\frac{m}{s^{2}}}', kind='markup'),)

    A dimensionless unit contributes no fragment at all:

    >>> pparts(u.unit(""))
    ()

    ``"name"`` picks the spelled-out name, ``"dim"`` the physical dimension:

    >>> pparts(u.unit("m"), unit="name")
    (PPart(role='unit', text='meter', kind='content'),)

    >>> pparts(u.unit("m"), unit="dim")
    (PPart(role='unit', text='length', kind='content'),)

    Both fall back to the symbol when the unit has no such spelling -- a
    composite has no single long name, so asking for one is not an error:

    >>> pparts(u.unit("km/s"), unit="name")
    (PPart(role='unit', text='km / s', kind='content'),)

    """
    # A name or a dimension is a plain word, not astropy's own rendering, so it
    # goes through the engine's ordinary content escaping in every markup
    # rather than astropy's markup-specific ``to_string(markup)``.
    if unit == "name" and (names := getattr(obj, "long_names", None)):
        return (fmt.PPart("unit", names[0]),)
    if unit == "dim":
        with contextlib.suppress(Exception):
            return (fmt.PPart("unit", str(obj.physical_type)),)

    # Decide emptiness on the *plain* string: a dimensionless unit's LaTeX form
    # is ``$\mathrm{}$``, which is truthy after the ``$`` are stripped and would
    # otherwise emit a phantom unit.
    plain = obj.to_string()
    if not plain:
        return ()
    if markup == "latex":
        return (fmt.PPart("unit", fmt.unwrap_math(obj.to_string("latex")), "markup"),)
    return (fmt.PPart("unit", plain),)
