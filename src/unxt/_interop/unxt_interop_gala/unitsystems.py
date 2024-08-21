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
    >>> from gala.units import UnitSystem
    >>> import astropy.units as u
    >>> usys = UnitSystem(u.km, u.s, u.Msun, u.radian)

    >>> from unxt import unitsystem
    >>> unitsystem(usys)
    unitsystem(km, s, solMass, rad)

    """
    # Create a new unit system instance, and possibly class.
    return unitsystem(*value._core_units)  # noqa: SLF001


@dispatch  # type: ignore[no-redef]
def unitsystem(_: GalaDimensionlessUnitSystem, /) -> DimensionlessUnitSystem:
    """Return a `gala.units.DimensionlessUnitSystem` as a `unxt.DimensionlessUnitSystem`.

    Examples
    --------
    >>> from gala.units import DimensionlessUnitSystem
    >>> import astropy.units as u
    >>> usys = DimensionlessUnitSystem()

    >>> from unxt import unitsystem
    >>> unitsystem(usys)
    DimensionlessUnitSystem()

    """  # noqa: E501
    return dimensionless
