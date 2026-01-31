"""Dimensions compatibility."""

__all__: tuple[str, ...] = ()


from dataclasses import MISSING, Field

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
def fields(obj: AbstractDimension, /) -> tuple[Field, ...]:
    """Return the fields of a dimension.

    Examples
    --------
    >>> import dataclassish as dc
    >>> import astropy.units as apyu

    >>> dim = apyu.get_physical_type("length")
    >>> dc.fields(dim)
    (Field(name='_unit',...), Field(name='_physical_type',...))

    """
    unit_field = Field(MISSING, MISSING, True, False, False, False, {}, False)  # noqa: FBT003
    unit_field.name = "_unit"
    physical_type_field = Field(MISSING, MISSING, True, False, False, False, {}, False)  # noqa: FBT003
    physical_type_field.name = "_physical_type"
    return (unit_field, physical_type_field)
