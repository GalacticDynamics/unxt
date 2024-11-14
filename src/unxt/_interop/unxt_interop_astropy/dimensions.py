"""Unitsystem compatibility."""

__all__: list[str] = []


import astropy.units as u
from plum import dispatch

from unxt._src.dimensions.core import AbstractDimensions


@dispatch  # type: ignore[misc]
def dimension_of(obj: u.Quantity, /) -> AbstractDimensions:
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
