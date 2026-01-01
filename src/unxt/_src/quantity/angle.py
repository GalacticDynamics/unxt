"""Angular quantities."""

__all__ = ("Angle",)

from typing import final

import equinox as eqx
from jaxtyping import Array, Real

import unxt_api as uapi
from .base_angle import AbstractAngle
from .value import StaticValue, convert_to_quantity_value
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
    >>> q.wrap_to(u.Q(0, "deg"), u.Q(360, "deg"))
    Angle(Array(10, dtype=int32, ...), unit='deg')

    Create an Angle array:

    >>> q = u.Angle([1, 2, 3], "deg")
    >>> q
    Angle(Array([1, 2, 3], dtype=int32), unit='deg')

    Do math on an Angle:

    >>> 2 * q
    Angle(Array([2, 4, 6], dtype=int32), unit='deg')

    >>> q % u.Q(4, "deg")
    Angle(Array([1, 2, 3], dtype=int32), unit='deg')

    """

    value: Real[Array | StaticValue, "*shape"] = eqx.field(
        converter=convert_to_quantity_value
    )
    """The value of the `unxt.AbstractQuantity`."""

    unit: AbstractUnit = eqx.field(static=True, converter=uapi.unit)
    """The unit associated with this value."""
