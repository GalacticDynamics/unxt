"""Conversion between Quantity types."""

__all__: list[str] = []

from plum import conversion_method

from .base import AbstractQuantity
from .core import Quantity
from .distance import Distance, DistanceModulus, Parallax
from .fast import UncheckedQuantity


@conversion_method(type_from=AbstractQuantity, type_to=UncheckedQuantity)  # type: ignore[misc]
def _quantity_to_unchecked(q: AbstractQuantity, /) -> UncheckedQuantity:
    """Convert any quantity to an unchecked quantity.

    Examples
    --------
    >>> from plum import convert
    >>> from unxt import Quantity, UncheckedQuantity
    >>> q = Quantity(1, "m")
    >>> q
    Quantity['length'](Array(1, dtype=int32, ...), unit='m')

    >>> convert(q, UncheckedQuantity)
    UncheckedQuantity(Array(1, dtype=int32, ...), unit='m')

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
    >>> from unxt import Quantity, UncheckedQuantity
    >>> q = UncheckedQuantity(1, "m")
    >>> q
    UncheckedQuantity(Array(1, dtype=int32, ...), unit='m')

    >>> convert(q, Quantity)
    Quantity['length'](Array(1, dtype=int32, ...), unit='m')

    """
    if isinstance(q, Quantity):
        return q
    return Quantity(q.value, q.unit)


@conversion_method(type_from=AbstractQuantity, type_to=Distance)  # type: ignore[misc]
def _quantity_to_distance(q: AbstractQuantity, /) -> Distance:
    """Convert any quantity to a Distance.

    Examples
    --------
    >>> from plum import convert
    >>> from unxt import Quantity, Distance
    >>> q = UncheckedQuantity(1, "m")
    >>> q
    UncheckedQuantity(Array(1, dtype=int32, ...), unit='m')

    >>> convert(q, Distance)
    Distance(Array(1, dtype=int32, ...), unit='m')

    """
    if isinstance(q, Distance):
        return q
    return Distance(q.value, q.unit)


@conversion_method(type_from=AbstractQuantity, type_to=Parallax)  # type: ignore[misc]
def _quantity_to_parallax(q: AbstractQuantity, /) -> Parallax:
    """Convert any quantity to a Parallax.

    Examples
    --------
    >>> from plum import convert
    >>> from unxt import Quantity, Parallax
    >>> q = Quantity(1, "mas")
    >>> q
    Quantity['angle'](Array(1, dtype=int32, ...), unit='mas')

    >>> convert(q, Parallax)
    Parallax(Array(1, dtype=int32, weak_type=True), unit='mas')

    """
    if isinstance(q, Parallax):
        return q
    return Parallax(q.value, q.unit)


@conversion_method(type_from=AbstractQuantity, type_to=DistanceModulus)  # type: ignore[misc]
def _quantity_to_distmod(q: AbstractQuantity, /) -> DistanceModulus:
    """Convert any quantity to a DistanceModulus.

    Examples
    --------
    >>> from plum import convert
    >>> from unxt import Quantity, DistanceModulus
    >>> q = Quantity(1, "mag")
    >>> q
    Quantity['mag'](Array(1, dtype=int32, ...), unit='mag')

    >>> convert(q, DistanceModulus)
    DistanceModulus(Array(1, dtype=int32, ...), unit='mag')

    """
    if isinstance(q, DistanceModulus):
        return q
    return DistanceModulus(q.value, q.unit)
