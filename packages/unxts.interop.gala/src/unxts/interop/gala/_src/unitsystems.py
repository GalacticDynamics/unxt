"""Unitsystem compatibility."""

__all__ = (
    "convert_gala_unitsystem_to_unxt_unitsystem",
    "convert_unxt_unitsystem_to_gala_unitsystem",
)

from typing import Any

import gala.units
from astropy.units import UnitBase as AstropyUnit
from plum import conversion_method, convert, dispatch

from unxt.unitsystems import AbstractUnitSystem, DimensionlessUnitSystem, dimensionless


@dispatch
def unitsystem(value: gala.units.UnitSystem, /) -> AbstractUnitSystem:
    """Return a `gala.units.UnitSystem` as a `unxt.AbstractUnitSystem`.

    Examples
    --------
    >>> import unxt as u
    >>> import gala.units as gu
    >>> import astropy.units as apyu

    >>> usys = gu.UnitSystem(apyu.km, apyu.s, apyu.Msun, apyu.radian)

    >>> u.unitsystem(usys)
    unitsystem(km, s, solMass, rad)

    """
    # Create a new unit system instance, and possibly class.
    return unitsystem(*value._core_units)  # noqa: SLF001


@dispatch
def unitsystem(_: gala.units.DimensionlessUnitSystem, /) -> DimensionlessUnitSystem:
    """Return a `gala.units.DimensionlessUnitSystem` as a `unxt.DimensionlessUnitSystem`.

    Examples
    --------
    >>> import unxt as u
    >>> from gala.units import DimensionlessUnitSystem

    >>> usys = DimensionlessUnitSystem()

    >>> u.unitsystem(usys)
    DimensionlessUnitSystem()

    """  # noqa: E501
    return dimensionless


# =============================================================================
# Convert


@conversion_method(type_from=gala.units.UnitSystem, type_to=AbstractUnitSystem)  # type: ignore[arg-type]
def convert_gala_unitsystem_to_unxt_unitsystem(
    usys: gala.units.UnitSystem, /
) -> AbstractUnitSystem:
    """Convert a `gala.units.UnitSystem` to a `unxt.AbstractUnitSystem`.

    This is a `plum.conversion_method` and is registered with `plum`'s
    dispatch table as a side effect of importing this module. Prefer calling
    `plum.convert(usys, AbstractUnitSystem)` over calling this function
    directly.

    Examples
    --------
    >>> import unxt as u
    >>> import gala.units as gu
    >>> import astropy.units as apyu

    >>> usys = gu.UnitSystem(apyu.km, apyu.s, apyu.Msun, apyu.radian)
    >>> usys
    <UnitSystem (km, s, solMass, rad)>

    >>> convert(usys, u.AbstractUnitSystem)
    unitsystem(km, s, solMass, rad)

    """
    return unitsystem(usys)


def _convert_unit_to_apyu(u: Any) -> AstropyUnit:
    """Convert a `unxt.AbstractUnit` to an astropy.unit."""
    return convert(u, AstropyUnit)


@conversion_method(type_from=AbstractUnitSystem, type_to=gala.units.UnitSystem)  # type: ignore[arg-type]
def convert_unxt_unitsystem_to_gala_unitsystem(
    usys: AbstractUnitSystem, /
) -> gala.units.UnitSystem:
    """Convert a `unxt.AbstractUnitSystem` to a `gala.units.UnitSystem`.

    This is a `plum.conversion_method` and is registered with `plum`'s
    dispatch table as a side effect of importing this module. Prefer calling
    `plum.convert(usys, gala.units.UnitSystem)` over calling this function
    directly.

    Examples
    --------
    >>> import unxt as u
    >>> import gala.units as gu
    >>> import astropy.units as apyu
    >>> from plum import convert

    >>> usys = u.unitsystem(apyu.km, apyu.s, apyu.Msun, apyu.radian)
    >>> usys
    unitsystem(km, s, solMass, rad)

    >>> convert(usys, gu.UnitSystem)
    <UnitSystem (km, s, solMass, rad)>

    """
    return gala.units.UnitSystem(*map(_convert_unit_to_apyu, usys.base_units))
