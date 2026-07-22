"""Conversions to ParametricQuantity (registered on import)."""

__all__: tuple[str, ...] = ()

from plum import conversion_method

from unxts.api import unit_of, ustrip

from .parametric import ParametricQuantity
from unxt.quantity import AbstractQuantity


@conversion_method(type_from=AbstractQuantity, type_to=ParametricQuantity)  # type: ignore[arg-type]
def quantity_to_checked(q: AbstractQuantity, /) -> ParametricQuantity:
    """Convert any quantity to a checked (parametric) quantity.

    Examples
    --------
    >>> from plum import convert
    >>> from unxt.quantity import Quantity
    >>> from unxts.parametric import ParametricQuantity

    >>> q = Quantity(1, "m")
    >>> q
    Quantity(Array(1, dtype=int32...), unit='m')

    >>> convert(q, ParametricQuantity)
    ParametricQuantity(Array(1, dtype=int32...), unit='m')

    The self-conversion doesn't copy the object:

    >>> q = ParametricQuantity(1, "m")
    >>> convert(q, ParametricQuantity) is q
    True

    """
    if isinstance(q, ParametricQuantity):
        return q
    u = unit_of(q)
    return ParametricQuantity(ustrip(u, q), u)
