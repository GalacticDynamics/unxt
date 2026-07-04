"""Conversion methods between quantity classes."""

__all__: tuple[str, ...] = ()

from plum import conversion_method

from .angle import Angle
from .base import AbstractQuantity
from .quantity import Quantity
from unxt_api import unit_of, ustrip


@conversion_method(type_from=AbstractQuantity, type_to=Quantity)  # type: ignore[arg-type]
def quantity_to_default(q: AbstractQuantity, /) -> Quantity:
    """Convert any quantity to the default (non-parametric) ``Quantity``.

    Examples
    --------
    >>> from plum import convert
    >>> from unxt.quantity import ParametricQuantity, Quantity

    >>> q = ParametricQuantity(1, "m")
    >>> convert(q, Quantity)
    Quantity(Array(1, dtype=int32...), unit='m')

    The self-conversion doesn't copy the object:

    >>> q = Quantity(1, "m")
    >>> convert(q, Quantity) is q
    True

    """
    if isinstance(q, Quantity):
        return q
    u = unit_of(q)
    return Quantity(ustrip(u, q), u)


@conversion_method(type_from=AbstractQuantity, type_to=Angle)  # type: ignore[arg-type]
def convert_quantity_to_angle(q: AbstractQuantity, /) -> Angle:
    """Convert any quantity to an Angle.

    Examples
    --------
    >>> from plum import convert
    >>> from unxt.quantity import Angle, Quantity
    >>> q = Quantity(1, "rad")
    >>> q
    Quantity(Array(1, dtype=int32...), unit='rad')

    >>> convert(q, Angle)
    Angle(Array(1, dtype=int32...), unit='rad')

    The self-conversion doesn't copy the object:

    >>> q = Angle(1, "rad")
    >>> convert(q, Angle) is q
    True

    """
    if isinstance(q, Angle):
        return q

    unit = unit_of(q)
    return Angle(ustrip(unit, q), unit)
