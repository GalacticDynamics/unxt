"""Astropy units compatibility."""

__all__: tuple[str, ...] = ()


import dataclasses

import astropy.units as apyu
import plum

from .custom_types import APYUnits

# ===================================================================
# Register dispatches


# Note: ``unit``/``unit_of`` for Astropy units live in ``unxt._src.units``
# (core), not here: astropy is unxt's unit backend, not an optional interop, so
# ``AbstractUnit`` already covers ``apyu.UnitBase``. The same goes for
# ``unit(apyu.Quantity)``, which must be registered before the built-in unit
# systems are constructed at import time.


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
    st_field = dataclasses.field(init=True, repr=True, hash=True, compare=True)  # pylint: disable=invalid-field-call
    st_field.name = "_names"
    return (st_field,)
