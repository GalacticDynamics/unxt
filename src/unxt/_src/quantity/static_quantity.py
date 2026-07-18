"""Static quantity class."""

# pylint: disable=import-error, no-member, unsubscriptable-object
#    b/c it doesn't understand dataclass fields

__all__ = ("StaticQuantity",)

from typing import Any, final

import equinox as eqx
import numpy as np
import wadler_lindig as wl
from jaxtyping import ArrayLike
from plum import add_promotion_rule

from .base import AbstractQuantity, ArrayLikeSequence
from .quantity import Quantity
from .value import StaticValue
from unxt.units import AbstractUnit, unit as parse_unit


@final
class StaticQuantity(AbstractQuantity):
    """A non-parametric quantity whose value is always a static NumPy array.

    Unlike `~unxt.Quantity`, its value is stored as a static (hashable) NumPy
    array, which lets a `StaticQuantity` be passed as a static argument to a
    `jax.jit`-compiled function. It accepts Python scalars and array-like
    inputs convertible to NumPy arrays; a concrete (eager) JAX array is
    materialised back to NumPy, and only a *traced* value -- which cannot be
    static -- is rejected.

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

    A concrete JAX array is materialised back to NumPy (a static value is just
    data); only *traced* values (under ``jit``/``vmap``/``grad``) are rejected:

    >>> import jax.numpy as jnp
    >>> u.StaticQuantity(jnp.array([1.0, 2.0]), "m")
    StaticQuantity(array([1., 2.], dtype=float32), unit='m')

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

    def __eq__(self, other: Any, /) -> bool | np.ndarray:  # type: ignore[override]
        """Return structural equality for static quantities."""
        if isinstance(other, StaticQuantity):
            return self.unit == other.unit and self.value == other.value
        return super().__eq__(other)

    def __pdoc__(self, *, show_wrapper: bool = False, **kwargs: Any) -> wl.AbstractDoc:
        """Return the Wadler-Lindig representation of this class."""
        return super().__pdoc__(show_wrapper=False, **kwargs)


add_promotion_rule(StaticQuantity, StaticQuantity, StaticQuantity)
add_promotion_rule(StaticQuantity, Quantity, Quantity)


@AbstractQuantity.from_.dispatch
def from_(
    cls: type[StaticQuantity],
    value: ArrayLike | ArrayLikeSequence,
    unit: Any,
    /,
    *,
    dtype: Any = None,
) -> StaticQuantity:
    """Construct a `StaticQuantity`, keeping the value on NumPy dtypes.

    The generic ``AbstractQuantity.from_`` routes the value through
    ``jnp.asarray``, which applies JAX's x64-disabled dtype rules and silently
    downcasts int64 / float64 to int32 / float32. A `StaticQuantity` stores its
    value verbatim, so use ``np.asarray`` here -- ``.from_`` then agrees with the
    ``__init__`` converter. (The keyword-``unit`` overload delegates here, so
    it is covered too.)

    Examples
    --------
    >>> import numpy as np
    >>> import unxt as u

    >>> u.StaticQuantity.from_(
    ...     np.array([1, 2, 3], dtype=np.int64), "m"
    ... ).value.array.dtype
    dtype('int64')

    """
    return cls(np.asarray(value, dtype=dtype), unit)
