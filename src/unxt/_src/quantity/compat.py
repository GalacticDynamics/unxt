"""Compatibility for Quantity."""

__all__: list[str] = []

from plum import conversion_method, dispatch

from .base import AbstractQuantity
from .core import Quantity
from .fast import UncheckedQuantity
from unxt._src.dimensions.core import AbstractDimension
from unxt._src.units.core import AbstractUnits

# ===================================================================
# Get dimensions


@dispatch  # type: ignore[misc]
def dimension_of(obj: AbstractQuantity, /) -> AbstractDimension:
    """Return the dimension of a quantity.

    Examples
    --------
    >>> from unxt import dimension_of, Quantity
    >>> q = Quantity(1, "m")
    >>> dimension_of(q)
    PhysicalType('length')

    """
    return dimension_of(obj.unit)


# ===================================================================
# Get units


@dispatch  # type: ignore[misc]
def unit_of(obj: AbstractQuantity, /) -> AbstractUnits:
    """Return the units of an object.

    Examples
    --------
    >>> from unxt import unit_of, Quantity
    >>> q = Quantity(1, "m")
    >>> unit_of(q)
    Unit("m")

    """
    return obj.unit


#####################################################################
# Conversion


@conversion_method(type_from=AbstractQuantity, type_to=UncheckedQuantity)  # type: ignore[misc]
def _quantity_to_unchecked(q: AbstractQuantity, /) -> UncheckedQuantity:
    """Convert any quantity to an unchecked quantity.

    Examples
    --------
    >>> from plum import convert
    >>> from unxt.quantity import Quantity, UncheckedQuantity

    >>> q = Quantity(1, "m")
    >>> q
    Quantity['length'](Array(1, dtype=int32, ...), unit='m')

    The self-conversion doesn't copy the object:

    >>> q = UncheckedQuantity(1, "m")
    >>> convert(q, UncheckedQuantity) is q
    True

    """
    if isinstance(q, UncheckedQuantity):
        return q
    return UncheckedQuantity(q.value, q.unit)


@conversion_method(type_from=AbstractQuantity, type_to=Quantity)  # type: ignore[misc]
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
    return Quantity(q.value, q.unit)
