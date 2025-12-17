"""Astropy units compatibility."""

__all__: tuple[str, ...] = ()


import astropy.units as apyu
from plum import dispatch

from .custom_types import APYUnits

# ===================================================================
# Register dispatches


@dispatch
def unit(obj: apyu.UnitBase | apyu.Unit, /) -> APYUnits:
    """Construct the units from an Astropy unit.

    Examples
    --------
    >>> import astropy.units as apyu
    >>> import unxt as u
    >>> u.unit(apyu.km)
    Unit("km")

    """
    return apyu.Unit(obj)


@dispatch
def unit(obj: apyu.Quantity, /) -> APYUnits:
    """Construct the units from an Astropy quantity.

    Examples
    --------
    >>> import astropy.units as apyu
    >>> import unxt as u
    >>> u.unit(apyu.Quantity(2, "km"))
    Unit("2 km")

    """
    return apyu.Unit(obj)


# -------------------------------------------------------------------


@dispatch
def unit_of(obj: apyu.UnitBase | apyu.Unit, /) -> APYUnits:
    """Return the units of an object.

    Examples
    --------
    >>> import astropy.units as apyu
    >>> import unxt as u

    >>> u.unit_of(apyu.km)
    Unit("km")

    """
    return obj


@dispatch
def unit_of(obj: apyu.Quantity, /) -> APYUnits:
    """Return the units of an Astropy quantity.

    Examples
    --------
    >>> import astropy.units as apyu
    >>> import unxt as u

    >>> u.unit_of(apyu.Quantity(1, "km"))
    Unit("km")

    """
    return unit_of(obj.unit)
