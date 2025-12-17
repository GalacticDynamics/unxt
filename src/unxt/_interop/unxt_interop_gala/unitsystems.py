"""Unitsystem compatibility."""
# TODO: move to a compatibility module

__all__: tuple[str, ...] = ()

from typing import Any

from astropy.units import UnitBase as AstropyUnit
from gala.units import (  # pylint: disable=import-error
    DimensionlessUnitSystem as GalaDimensionlessUnitSystem,
    UnitSystem as GalaUnitSystem,
)
from plum import conversion_method, convert, dispatch

from unxt.unitsystems import AbstractUnitSystem, DimensionlessUnitSystem, dimensionless


@dispatch
def unitsystem(value: GalaUnitSystem, /) -> AbstractUnitSystem:
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
def unitsystem(_: GalaDimensionlessUnitSystem, /) -> DimensionlessUnitSystem:
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


@conversion_method(type_from=GalaUnitSystem, type_to=AbstractUnitSystem)  # type: ignore[arg-type]
def convert_gala_unitsystem_to_unxt_unitsystem(
    usys: GalaUnitSystem, /
) -> AbstractUnitSystem:
    """Convert a `gala.units.UnitSystem` to a `unxt.AbstractUnitSystem`.

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


@conversion_method(type_from=AbstractUnitSystem, type_to=GalaUnitSystem)  # type: ignore[arg-type]
def convert_unxt_unitsystem_to_gala_unitsystem(
    usys: AbstractUnitSystem, /
) -> GalaUnitSystem:
    """Convert a `unxt.AbstractUnitSystem` to a `gala.units.UnitSystem`.

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

    def convert_unit_to_apyu(u: Any) -> AstropyUnit:
        return convert(u, AstropyUnit)

    return GalaUnitSystem(*map(convert_unit_to_apyu, usys.base_units))
