"""Compatibility for Quantity."""

__all__: list[str] = []

from plum import conversion_method

from .api import ustrip
from .base import AbstractQuantity
from .quantity import Quantity
from .unchecked import BareQuantity
from unxt.units import unit_of


@conversion_method(type_from=AbstractQuantity, type_to=BareQuantity)  # type: ignore[arg-type]
def quantity_to_unchecked(q: AbstractQuantity, /) -> BareQuantity:
    """Convert any quantity to an unchecked quantity.

    Examples
    --------
    >>> from plum import convert
    >>> from unxt.quantity import Quantity, BareQuantity

    >>> q = Quantity(1, "m")
    >>> convert(q, BareQuantity)
    BareQuantity(Array(1, dtype=int32, weak_type=True), unit='m')

    The self-conversion doesn't copy the object:

    >>> q = BareQuantity(1, "m")
    >>> convert(q, BareQuantity) is q
    True

    """
    if isinstance(q, BareQuantity):
        return q
    unit = unit_of(q)
    return BareQuantity(ustrip(unit, q), unit)


@conversion_method(type_from=AbstractQuantity, type_to=Quantity)  # type: ignore[arg-type]
def quantity_to_checked(q: AbstractQuantity, /) -> Quantity:
    """Convert any quantity to a checked quantity.

    Examples
    --------
    >>> from plum import convert
    >>> from unxt.quantity import Quantity, BareQuantity

    >>> q = BareQuantity(1, "m")
    >>> q
    BareQuantity(Array(1, dtype=int32, ...), unit='m')

    >>> convert(q, Quantity)
    Quantity['length'](Array(1, dtype=int32, ...), unit='m')

    The self-conversion doesn't copy the object:

    >>> q = Quantity(1, "m")
    >>> convert(q, Quantity) is q
    True

    """
    if isinstance(q, Quantity):
        return q
    u = unit_of(q)
    return Quantity(ustrip(u, q), u)
