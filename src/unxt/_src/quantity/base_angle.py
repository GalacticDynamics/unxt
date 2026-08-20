"""Base classes for angular quantities."""

__all__ = ("AbstractAngle",)

import equinox as eqx
from jaxtyping import Array, Shaped
from plum import add_promotion_rule

import unxt_api as uapi
from .base import AbstractQuantity
from .quantity import Quantity
from unxt._src.dimensions import dimension, dimension_of
from unxt._src.quantity.value import StaticValue
from unxt.units import AbstractUnit

angle_dimension = dimension("angle")


#: Units already shown to be angular. `__check_init__` runs per angle
#: constructed, and its check is a dispatched call whose result depends only on
#: the unit -- an immutable value object -- so it is answered once per unit.
_ANGULAR_UNITS: set[AbstractUnit] = set()


class AbstractAngle(AbstractQuantity):
    """Angular quantity.

    See Also
    --------
    `unxt.Angle` : a concrete implementation of this class.

    Examples
    --------
    For this example, we will use the concrete implementation of
    `unxt.AbstractAngle`, `unxt.Angle`.

    >>> from unxt import Angle

    >>> Angle(90, "deg")
    Angle(Array(90, dtype=int32...), unit='deg')

    Angles have to have dimensions of angle.

    >>> try:
    ...     Angle(90, "m")
    ... except ValueError as e:
    ...     print(e)
    Angle must have units with angular dimensions.

    """

    value: eqx.AbstractVar[Shaped[Array | StaticValue, "*shape"]]
    """The value of the `unxt.AbstractQuantity`."""

    unit: eqx.AbstractVar[AbstractUnit]
    """The unit associated with this value."""

    def __check_init__(self) -> None:
        """Check the initialization."""
        # `dimension_of(self)` forwards to `dimension_of(self.unit)`, so going
        # straight to the unit skips a dispatch. The answer is then memoised:
        # units are immutable value objects, so one that is angular stays
        # angular, and the set is bounded by the units a program constructs.
        # This runs on every angle built, and the dispatch dominates it --
        # ~138us against ~0.9us for the `physical_type` lookup underneath.
        unit = self.unit
        if unit in _ANGULAR_UNITS:
            return
        if dimension_of(unit) != angle_dimension:
            msg = f"{type(self).__name__} must have units with angular dimensions."
            raise ValueError(msg)
        _ANGULAR_UNITS.add(unit)

    def wrap_to(
        self, /, min: AbstractQuantity, max: AbstractQuantity
    ) -> "AbstractAngle":
        """Wrap the angle to the range [min, max).

        Parameters
        ----------
        min, max
            The minimum, maximum value of the range.

        See Also
        --------
        `unxt.quantity.wrap_to` : functional version of this method.

        Examples
        --------
        >>> import unxt as u
        >>> angle = u.Angle(370, "deg")
        >>> angle.wrap_to(min=u.Q(0, "deg"), max=u.Q(360, "deg"))
        Angle(Array(10, dtype=int32...), unit='deg')

        """
        return uapi.wrap_to(self, min, max)


# Add a rule that when an AbstractAngle interacts with a Quantity, the angle
# degrades to a Quantity. This is necessary for many operations, e.g. division
# of an angle by a non-dimensionless quantity where the resulting units are not
# those of an angle.
add_promotion_rule(AbstractAngle, Quantity, Quantity)
