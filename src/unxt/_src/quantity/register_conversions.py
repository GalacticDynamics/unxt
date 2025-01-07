"""Compatibility for Quantity."""

__all__: list[str] = []

from plum import conversion_method

from .api import ustrip
from .base import AbstractQuantity
from .quantity import Quantity
from .unchecked import UncheckedQuantity
from unxt._src.units.api import unit_of


@conversion_method(type_from=AbstractQuantity, type_to=UncheckedQuantity)  # type: ignore[arg-type]
def _quantity_to_unchecked(q: AbstractQuantity, /) -> UncheckedQuantity:
    """Convert any quantity to an unchecked quantity.

    Examples
    --------
    >>> from plum import convert
    >>> from unxt.quantity import Quantity, UncheckedQuantity

    >>> q = Quantity(1, "m")
    >>> convert(q, UncheckedQuantity)
    UncheckedQuantity(Array(1, dtype=int32, weak_type=True), unit='m')

    The self-conversion doesn't copy the object:

    >>> q = UncheckedQuantity(1, "m")
    >>> convert(q, UncheckedQuantity) is q
    True

    """
    if isinstance(q, UncheckedQuantity):
        return q
    unit = unit_of(q)
    return UncheckedQuantity(ustrip(unit, q), unit)


@conversion_method(type_from=AbstractQuantity, type_to=Quantity)  # type: ignore[arg-type]
def _quantity_to_checked(q: AbstractQuantity, /) -> Quantity:
    """Convert any quantity to a checked quantity.

    Examples
    --------
    >>> from plum import convert
    >>> from unxt.quantity import Quantity, UncheckedQuantity

    >>> q = UncheckedQuantity(1, "m")
    >>> q
    UncheckedQuantity(Array(1, dtype=int32, ...), unit='m')

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
