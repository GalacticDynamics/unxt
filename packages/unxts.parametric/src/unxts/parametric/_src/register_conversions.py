"""Conversions to `ParametricQuantity` (registered on import).

Both an `unxt` quantity and an `astropy.units.Quantity` convert the same way --
read the unit, strip the value in it, rebuild -- so the two `plum` conversion
methods share one body. ``astropy`` is a hard dependency of ``unxts.parametric``
(its ``PhysicalType`` machinery requires it), so its conversion is always
registered.
"""

__all__: tuple[str, ...] = ()

from typing import Any

from astropy.units import Quantity as AstropyQuantity
from plum import conversion_method

from unxts.api import unit_of, ustrip

from .parametric import ParametricQuantity
from unxt.quantity import AbstractQuantity


def _to_parametric(q: Any, /) -> ParametricQuantity:
    """Rebuild ``q`` as a `ParametricQuantity` in its own unit."""
    u = unit_of(q)
    return ParametricQuantity(ustrip(u, q), u)


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
    return _to_parametric(q)


@conversion_method(type_from=AstropyQuantity, type_to=ParametricQuantity)  # type: ignore[arg-type]
def astropy_quantity_to_parametric(q: AstropyQuantity, /) -> ParametricQuantity:
    """Convert an `astropy.units.Quantity` to a `ParametricQuantity`.

    Examples
    --------
    >>> from astropy.units import Quantity as AstropyQuantity
    >>> from plum import convert
    >>> from unxts.parametric import ParametricQuantity

    >>> convert(AstropyQuantity(1.0, "cm"), ParametricQuantity)
    ParametricQuantity(Array(1., dtype=float32), unit='cm')

    """
    return _to_parametric(q)
