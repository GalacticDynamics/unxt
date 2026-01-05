"""Static Quantity Class."""

# pylint: disable=import-error, no-member, unsubscriptable-object
#    b/c it doesn't understand dataclass fields

__all__ = ("StaticQuantity",)

from typing import Any, final

import equinox as eqx
import wadler_lindig as wl
from numpy.typing import NDArray
from plum import add_promotion_rule, parametric

from .base_parametric import AbstractParametricQuantity
from .quantity import Quantity
from .unchecked import BareQuantity
from .value import StaticValue
from unxt.units import AbstractUnit, unit as parse_unit


@final
@parametric
class StaticQuantity(AbstractParametricQuantity):
    """Static quantities with associated units.

    This class is parametrized by the dimensions of the units, just like
    {class}`~unxt.quantity.Quantity`, but its value is always stored as a static
    NumPy array. It accepts Python scalars and array-like inputs that can be
    converted to NumPy arrays, and it rejects JAX arrays.

    Examples
    --------
    >>> import numpy as np
    >>> import unxt as u

    Basic construction:

    >>> q = u.StaticQuantity(np.array([1.0, 2.0]), "m")
    >>> q
    StaticQuantity(array([1., 2.]), unit='m')

    Values are static and hashable:

    >>> isinstance(hash(q), int)
    True

    JAX arrays are rejected:

    >>> import jax.numpy as jnp
    >>> try:
    ...     u.StaticQuantity(jnp.array([1.0, 2.0]), "m")
    ... except TypeError as e:
    ...     print(e)
    StaticQuantity does not accept JAX arrays. Use Quantity for traced values.

    The Wadler-Lindig representation hides the internal static wrapper:

    >>> import wadler_lindig as wl
    >>> wl.pprint(q, short_arrays=False)
    StaticQuantity(array([1., 2.]), unit='m')

    """

    value: StaticValue = eqx.field(  # type: ignore[assignment]
        static=True, converter=StaticValue.from_
    )
    """The static value of the `AbstractQuantity`."""

    unit: AbstractUnit = eqx.field(static=True, converter=parse_unit)
    """The unit associated with this value."""

    def __hash__(self) -> int:
        """Return the hash of the quantity."""
        return hash((self.value, self.unit))

    def __eq__(self, other: Any, /) -> bool | NDArray[bool]:  # type: ignore[override]
        """Return structural equality for static quantities."""
        if isinstance(other, StaticQuantity):
            return self.unit == other.unit and self.value == other.value
        return super().__eq__(other)

    def __pdoc__(self, *, show_wrapper: bool = False, **kwargs: Any) -> wl.AbstractDoc:
        """Return the Wadler-Lindig representation of this class."""
        return super().__pdoc__(show_wrapper=False, **kwargs)


add_promotion_rule(StaticQuantity, StaticQuantity, StaticQuantity)
add_promotion_rule(StaticQuantity, Quantity, Quantity)
add_promotion_rule(StaticQuantity, BareQuantity, BareQuantity)
