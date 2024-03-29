"""Tools for representing systems of units using ``astropy.units``."""

__all__ = ["unitsystem"]


import astropy.units as u
from plum import dispatch

from ._base import AbstractUnitSystem
from ._core import DimensionlessUnitSystem, UnitSystem
from ._realizations import NAMED_UNIT_SYSTEMS, dimensionless
from unxt._quantity.base import AbstractQuantity
from unxt._typing import Unit


@dispatch
def unitsystem(units: AbstractUnitSystem, /) -> AbstractUnitSystem:
    """Convert a UnitSystem or tuple of arguments to a UnitSystem.

    Examples
    --------
    >>> import astropy.units as u
    >>> from unxt.unitsystems import UnitSystem, unitsystem
    >>> usys = UnitSystem(u.kpc, u.Myr, u.Msun, u.radian, u.km/u.s)
    >>> usys
    UnitSystem(kpc, Myr, solMass, rad)

    >>> unitsystem(usys)
    UnitSystem(kpc, Myr, solMass, rad)

    """
    return units


@dispatch  # type: ignore[no-redef]
def unitsystem(
    units: (
        tuple[Unit | u.Quantity | AbstractQuantity, ...]
        | list[Unit | u.Quantity | AbstractQuantity]
    ),
    /,
) -> AbstractUnitSystem:
    """Convert a UnitSystem or tuple of arguments to a UnitSystem.

    Examples
    --------
    >>> import astropy.units as u
    >>> from unxt.unitsystems import UnitSystem, unitsystem

    >>> unitsystem((u.kpc, u.Myr, u.Msun, u.radian, u.km/u.s))
    UnitSystem(kpc, Myr, solMass, rad)

    >>> unitsystem([u.kpc, u.Myr, u.Msun, u.radian, u.km/u.s])
    UnitSystem(kpc, Myr, solMass, rad)

    """
    return UnitSystem(*units) if len(units) > 0 else dimensionless


@dispatch  # type: ignore[no-redef]
def unitsystem(_: None, /) -> DimensionlessUnitSystem:
    """Dimensionless unit system from None.

    Examples
    --------
    >>> from unxt.unitsystems import unitsystem
    >>> unitsystem(None)
    DimensionlessUnitSystem()

    """
    return dimensionless


@dispatch  # type: ignore[no-redef]
def unitsystem(unit0: Unit, /, *units: Unit) -> UnitSystem:
    """Convert a set of arguments to a UnitSystem.

    Examples
    --------
    >>> import astropy.units as u
    >>> from unxt.unitsystems import UnitSystem, unitsystem

    >>> unitsystem(u.kpc, u.Myr, u.Msun, u.radian)
    UnitSystem(kpc, Myr, solMass, rad)

    """
    return UnitSystem(unit0, *units)


@dispatch  # type: ignore[no-redef]
def unitsystem(name: str, /) -> AbstractUnitSystem:
    """Return unit system from name.

    Examples
    --------
    >>> from unxt.unitsystems import unitsystem
    >>> unitsystem("galactic")
    UnitSystem(kpc, Myr, solMass, rad)

    >>> unitsystem("solarsystem")
    UnitSystem(AU, yr, solMass, rad)

    >>> unitsystem("dimensionless")
    DimensionlessUnitSystem()

    """
    return NAMED_UNIT_SYSTEMS[name]
