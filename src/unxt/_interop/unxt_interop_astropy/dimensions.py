"""Dimensions compatibility."""

__all__: tuple[str, ...] = ()


import dataclasses

import astropy.units as apyu
import plum

from unxt.dims import AbstractDimension


@plum.dispatch
def dimension_of(obj: apyu.Quantity, /) -> AbstractDimension:
    """Return the dimension of a quantity.

    Examples
    --------
    >>> import unxt as u
    >>> import astropy.units as apyu

    >>> q = apyu.Quantity(1, "m")
    >>> u.dimension_of(q)
    PhysicalType('length')

    """
    return dimension_of(obj.unit)


# ===================================================
# `Dataclassish` support


@plum.dispatch
def fields(obj: AbstractDimension, /) -> tuple[dataclasses.Field, ...]:
    """Return the fields of a dimension.

    Examples
    --------
    >>> import dataclassish as dc
    >>> import astropy.units as apyu

    >>> dim = apyu.get_physical_type("length")
    >>> dc.fields(dim)
    (Field(name='_unit',...), Field(name='_physical_type',...))

    """
    unit_field = dataclasses.field(init=True, repr=False, hash=False, compare=False)  # pylint: disable=invalid-field-call
    unit_field.name = "_unit"
    physical_type_field = dataclasses.field(  # pylint: disable=invalid-field-call
        init=True, repr=False, hash=False, compare=False
    )
    physical_type_field.name = "_physical_type"
    return (unit_field, physical_type_field)
