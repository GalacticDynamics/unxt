"""Unitsystem compatibility."""

__all__: list[str] = []


import astropy.units as u
from plum import dispatch

from unxt._src.dimensions.core import AbstractDimensions


@dispatch  # type: ignore[misc]
def dimensions_of(obj: u.Quantity, /) -> AbstractDimensions:
    """Return the dimensions of a quantity.

    Examples
    --------
    >>> import astropy.units as u
    >>> from unxt import dimensions_of

    >>> q = u.Quantity(1, "m")
    >>> dimensions_of(q)
    PhysicalType('length')

    """
    return dimensions_of(obj.unit)
