"""Static quantity class."""

# pylint: disable=import-error, no-member, unsubscriptable-object
#    b/c it doesn't understand dataclass fields

__all__ = ("StaticQuantity",)

from typing import Any, final

import equinox as eqx
import jax
import jax.core
import numpy as np
import unxts.api as uapi
import wadler_lindig as wl
from jaxtyping import ArrayLike
from plum import add_promotion_rule

from .base import AbstractQuantity, ArrayLikeSequence, same_unit_label
from .quantity import Quantity
from .value import TRACED_VALUE_MSG, StaticValue
from unxt.units import AbstractUnit


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

    unit: AbstractUnit = eqx.field(static=True, converter=uapi.unit)
    """The unit associated with this value."""

    @classmethod
    def _mk(cls, *, value: Any, unit: AbstractUnit) -> "StaticQuantity":
        """Build a `StaticQuantity`, converting the value but not the unit.

        `AbstractQuantity._mk` writes the fields verbatim. That is unsound here
        for the *value*: primitive rules hand this class the output of a
        `jax.lax` operation like any other quantity, and that field is the one
        thing the whole class is built on being static, so a raw array must not
        reach it. The unit needs no such care -- ``_mk``'s contract is that
        callers pass an already-normalised unit, and both callers (`revalue`
        and ``enable_materialise``) take one straight off another quantity.

        Rather than route back through the checked constructor, which costs two
        `plum`-dispatched converters plus `equinox`'s ``__init__`` machinery
        (~29us on a NumPy value, ~114us on a JAX one), inline the type switch
        from `StaticValue.from_`. `StaticQuantity` is `~typing.final` and has
        exactly two fields, so the signature can name them rather than take
        ``**fields``, and the switch has only three arms. ~1-2us.

        Inlining is the one liberty taken here, so `StaticValue.from_` shares
        `TRACED_VALUE_MSG` with this method, and
        ``test_mk_matches_static_value_from_`` pins the two together over every
        arm.

        Examples
        --------
        >>> import jax, jax.numpy as jnp
        >>> import numpy as np
        >>> import unxt as u
        >>> from unxt._src.quantity.base import revalue

        A concrete value is materialised, not stored raw:

        >>> q = u.StaticQuantity(np.array([1.0, 2.0]), "m")
        >>> revalue(q, jnp.asarray([3.0, 4.0])).value
        StaticValue(array([3., 4.], dtype=float32))

        A traced value is still rejected:

        >>> try:
        ...     jax.jit(lambda v: revalue(q, v))(jnp.asarray([3.0, 4.0]))
        ... except TypeError as e:
        ...     print(e)
        StaticQuantity cannot hold a traced JAX value; use Quantity under jit/vmap/grad.

        """
        if not isinstance(value, StaticValue):
            if isinstance(value, jax.core.Tracer):
                raise TypeError(TRACED_VALUE_MSG)
            # `StaticValue.__init__` does the `np.asarray`, so this one arm
            # covers NumPy, array-likes, scalars and concrete `jax.Array`.
            value = StaticValue(value)
        return cls.__make__(value=value, unit=unit)

    def __hash__(self) -> int:
        """Return the hash of the quantity."""
        return hash((self.value, self.unit))

    def __eq__(self, other: Any, /) -> bool | np.ndarray:  # type: ignore[override]
        """Return structural equality for static quantities."""
        if isinstance(other, StaticQuantity):
            # Label, not physical, equality -- see `same_unit_label`.
            return same_unit_label(self.unit, other.unit) and self.value == other.value
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
    value verbatim, so hand the value straight to ``__init__`` and let the
    ``StaticValue.from_`` converter convert it -- that preserves the NumPy
    dtype. Delegating (rather than calling ``np.asarray`` here) also keeps
    ``.from_`` and ``__init__`` under the *same* policy for JAX inputs, instead
    of materialising an array the constructor would reject.
    (The keyword-``unit`` overload delegates here, so it is covered too.)

    Examples
    --------
    >>> import numpy as np
    >>> import unxt as u

    >>> u.StaticQuantity.from_(
    ...     np.array([1, 2, 3], dtype=np.int64), "m"
    ... ).value.array.dtype
    dtype('int64')

    ``.from_`` applies the *same* JAX-input policy as ``__init__`` -- it
    delegates rather than converting first, so the two cannot drift:

    >>> import jax.numpy as jnp
    >>> u.StaticQuantity.from_(jnp.array([1.0, 2.0]), "m")
    StaticQuantity(array([1., 2.], dtype=float32), unit='m')

    """
    # ``dtype=None`` means "keep the value's own dtype"; only re-cast when the
    # caller explicitly asks. The converter does the array conversion.
    if dtype is not None:
        value = np.asarray(value, dtype=dtype)
    return cls(value, unit)
