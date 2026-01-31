"""Astropy units compatibility."""

__all__: tuple[str, ...] = ()


import dataclasses

import astropy.units as apyu
import plum

from .custom_types import APYUnits

# ===================================================================
# Register dispatches


@plum.dispatch
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


@plum.dispatch
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


@plum.dispatch
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


@plum.dispatch
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


# ===================================================
# `Dataclassish` support


@plum.dispatch
def fields(obj: APYUnits, /) -> tuple[dataclasses.Field, ...]:
    """Return the fields of a dimension.

    Examples
    --------
    >>> import dataclassish as dc
    >>> import astropy.units as apyu

    >>> dim = apyu.Unit("m")
    >>> dc.fields(dim)
    (Field(name='_names',...)

    """
    st_field = dataclasses.Field(
        dataclasses.MISSING,
        dataclasses.MISSING,
        True,  # noqa: FBT003
        True,  # noqa: FBT003
        True,  # noqa: FBT003
        True,  # noqa: FBT003
        {},
        False,  # noqa: FBT003
    )
    st_field.name = "_names"
    return (st_field,)
