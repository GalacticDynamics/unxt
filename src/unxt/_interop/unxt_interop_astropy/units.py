"""Astropy units compatibility."""

__all__: list[str] = []

from typing import TypeAlias

import astropy.units as u
from plum import dispatch

AstropyUnits: TypeAlias = u.UnitBase

# ===================================================================
# Register dispatches


@dispatch
def units(obj: u.UnitBase | u.Unit, /) -> AstropyUnits:
    """Construct the units from an Astropy unit.

    Examples
    --------
    >>> import astropy.units as u
    >>> from unxt import units
    >>> units(u.km)
    Unit("km")

    """
    return u.Unit(obj)


@dispatch
def units(obj: u.Quantity, /) -> AstropyUnits:
    """Construct the units from an Astropy quantity.

    Examples
    --------
    >>> import astropy.units as u
    >>> from unxt import units
    >>> units(u.Quantity(2, "km"))
    Unit("2 km")

    """
    return u.Unit(obj)


# -------------------------------------------------------------------


@dispatch
def unit_of(obj: u.UnitBase | u.Unit, /) -> AstropyUnits:
    """Return the units of an object.

    Examples
    --------
    >>> import astropy.units as u
    >>> from unxt import unit_of

    >>> unit_of(u.km)
    Unit("km")

    """
    return obj


@dispatch
def unit_of(obj: u.Quantity, /) -> AstropyUnits:
    """Return the units of an Astropy quantity.

    Examples
    --------
    >>> import astropy.units as u
    >>> from unxt import unit_of

    >>> unit_of(u.Quantity(1, "km"))
    Unit("km")

    """
    return unit_of(obj.unit)
