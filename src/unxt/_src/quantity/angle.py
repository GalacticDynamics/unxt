"""Angular quantities."""

__all__ = ("Angle",)

from typing import final

import equinox as eqx
from jaxtyping import Array, Shaped

import unxt_api as api
from .base_angle import AbstractAngle
from .value import convert_to_quantity_value
from unxt.units import AbstractUnit


@final
class Angle(AbstractAngle):
    """Angular quantity.

    Examples
    --------
    >>> import unxt as u

    Create an Angle:

    >>> q = u.Angle(1, "rad")
    >>> q
    Angle(Array(1, dtype=int32, ...), unit='rad')

    Wrap an Angle to a range:

    >>> q = u.Angle(370, "deg")
    >>> q.wrap_to(u.Quantity(0, "deg"), u.Quantity(360, "deg"))
    Angle(Array(10, dtype=int32, ...), unit='deg')

    Create an Angle array:

    >>> q = u.Angle([1, 2, 3], "deg")
    >>> q
    Angle(Array([1, 2, 3], dtype=int32), unit='deg')

    Do math on an Angle:

    >>> 2 * q
    Angle(Array([2, 4, 6], dtype=int32), unit='deg')

    >>> q % u.Quantity(4, "deg")
    Angle(Array([1, 2, 3], dtype=int32), unit='deg')

    """

    value: Shaped[Array, "*shape"] = eqx.field(converter=convert_to_quantity_value)
    """The value of the `unxt.AbstractQuantity`."""

    unit: AbstractUnit = eqx.field(static=True, converter=api.unit)
    """The unit associated with this value."""
