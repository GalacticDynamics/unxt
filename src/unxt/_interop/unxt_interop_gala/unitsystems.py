"""Unitsystem compatibility."""
# TODO: move to a compatibility module

__all__: list[str] = []

from gala.units import (  # pylint: disable=import-error
    DimensionlessUnitSystem as GalaDimensionlessUnitSystem,
    UnitSystem as GalaUnitSystem,
)
from plum import dispatch

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


@dispatch  # type: ignore[no-redef]
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
