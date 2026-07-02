"""Conversion to quantity value types."""

__all__ = ("StaticValue", "convert_to_quantity_value")

import operator
import warnings
from typing import Any, final
from typing_extensions import override

import jax
import numpy as np
import plum
import quax
import wadler_lindig as wl
from jaxtyping import Array, ArrayLike

import quaxed.numpy as jnp


@final
class StaticValue:
    """Immutable static value wrapper for `StaticQuantity`.

    This stores a read-only NumPy array and is used to keep the value static
    while avoiding Equinox's static-array warnings. Arithmetic operations
    degrade to the wrapped array unless both operands are `StaticValue`, in
    which case a `StaticValue` is returned.

    Note that since the array is immutable, hashing is supported.
    The hash is computed from the array's dtype, shape, and bytes.

    .. rubric:: Equality semantics

    `StaticValue` equality returns a **scalar** `bool`, not an element-wise
    array.  This is intentional: it mirrors Python's structural-equality
    contract (like ``tuple.__eq__``) and is a prerequisite for using a
    ``ParametricQuantity(StaticValue, ...)`` as a ``static_argnames`` argument in
    :func:`jax.jit`.  JAX needs a scalar ``bool`` to decide whether a cached
    compilation can be reused.

    This is fundamentally different from a normal :class:`~unxt.ParametricQuantity`
    (backed by a JAX array), whose ``==`` operator follows NumPy broadcasting
    rules and returns an element-wise boolean array.

    Examples
    --------
    >>> import numpy as np
    >>> from unxt.quantity import StaticValue

    Equality between two `StaticValue` objects returns a plain `bool`:

    >>> StaticValue(np.array([1.0, 2.0])) == StaticValue(np.array([1.0, 2.0]))
    True
    >>> StaticValue(np.array([1.0, 2.0])) == StaticValue(np.array([9.0, 9.0]))
    False

    Equality with a NumPy array returns an element-wise boolean array
    (same as NumPy):

    >>> StaticValue(np.array([1.0, 2.0])) == np.array([1.0, 9.0])
    array([ True, False])

    """

    __slots__ = ("_array",)

    def __init__(self, array: object, /) -> None:
        value = np.asarray(array)
        # Copy only when np.asarray returned a view of the caller's buffer so
        # that (a) we don't make the caller's array read-only as a side-effect
        # and (b) the caller cannot later mutate our internal data.
        if isinstance(array, np.ndarray) and np.shares_memory(value, array):
            value = value.copy()
        value.setflags(write=False)
        self._array = value

    @property
    def array(self) -> np.ndarray:
        """Return the contained NumPy array."""
        return self._array

    @property
    def _jnparray(self) -> Array:
        """Return the contained array as a JAX array.

        JAX 0.7.2+ introduces TypedNdArray for type-preserving operations.
        These are not jax.Array instances but carry dtype information.
        We need to explicitly convert them to proper arrays while preserving dtype.
        """
        out = jnp.asarray(self._array)
        if not isinstance(out, jax.Array):
            out = jnp.asarray(out, dtype=out.dtype)
        return out

    # Constructor
    @classmethod
    @plum.dispatch.abstract
    def from_(cls: type["StaticValue"], *args: Any, **kwargs: Any) -> "StaticValue":
        """Create a `StaticValue` from given arguments."""
        raise NotImplementedError  # pragma: no cover

    # ==================================================
    # NumPy API

    def __array__(self, dtype: Any = None) -> np.ndarray:
        if dtype is None:
            return np.asarray(self._array)
        return np.asarray(self._array, dtype=dtype)

    # ==================================================
    # Wadler-Lindig API

    def __pdoc__(self, *, show_wrapper: bool = True, **kwargs: Any) -> wl.AbstractDoc:
        """Return the Wadler-Lindig representation of this class."""
        if not show_wrapper:
            return wl.pdoc(self._array, **kwargs)
        return (
            wl.TextDoc("StaticValue(")
            + wl.pdoc(self._array, **kwargs)
            + wl.TextDoc(")")
        )

    # ==================================================
    # Python container API

    def __len__(self) -> int:
        return len(self._array)

    def __iter__(self) -> Any:
        return iter(self._array)

    def __getitem__(self, key: Any) -> Any:
        return self._array[key]

    def __getattr__(self, name: str) -> Any:
        return getattr(jnp.asarray(self._array), name)

    def __repr__(self) -> str:
        return wl.pformat(self, short_arrays=False, max_width=80)

    @override
    def __eq__(self, other: object, /) -> bool | np.ndarray:  # type: ignore[override]
        if isinstance(other, StaticValue):
            return np.array_equal(self._array, other._array)
        if isinstance(other, Array):
            return self._array == other
        if isinstance(other, (np.ndarray, list, tuple)):
            return np.equal(self._array, other)
        return NotImplemented

    def __ne__(self, other: object, /) -> bool | np.ndarray:  # type: ignore[override]
        if isinstance(other, StaticValue):
            return not bool(np.array_equal(self._array, other._array))
        if isinstance(other, Array):
            return self._array != other
        if isinstance(other, (np.ndarray, list, tuple)):
            return np.not_equal(self._array, other)
        return NotImplemented

    def __lt__(self, other: Any, /) -> np.ndarray:
        if isinstance(other, StaticValue):
            other = other._array
        return self._array < other

    def __le__(self, other: Any, /) -> np.ndarray:
        if isinstance(other, StaticValue):
            other = other._array
        return self._array <= other

    def __gt__(self, other: Any, /) -> np.ndarray:
        if isinstance(other, StaticValue):
            other = other._array
        return self._array > other

    def __ge__(self, other: Any, /) -> np.ndarray:
        if isinstance(other, StaticValue):
            other = other._array
        return self._array >= other

    def __hash__(self) -> int:
        return hash((self._array.dtype.str, self._array.shape, self._array.tobytes()))

    def _binary_op(self, other: Any, op: Any) -> Any:
        if isinstance(other, StaticValue):
            result = op(self._jnparray, jnp.asarray(other.array))
            return StaticValue(np.asarray(result))
        return op(self._jnparray, other)

    def _rbinary_op(self, other: Any, op: Any) -> Any:
        if isinstance(other, StaticValue):
            result = op(jnp.asarray(other.array), self._jnparray)
            return StaticValue(np.asarray(result))
        return op(other, self._jnparray)

    def __add__(self, other: Any) -> Any:
        return self._binary_op(other, operator.add)

    def __radd__(self, other: Any) -> Any:
        return self._rbinary_op(other, operator.add)

    def __sub__(self, other: Any) -> Any:
        return self._binary_op(other, operator.sub)

    def __rsub__(self, other: Any) -> Any:
        return self._rbinary_op(other, operator.sub)

    def __mul__(self, other: Any) -> Any:
        return self._binary_op(other, operator.mul)

    def __rmul__(self, other: Any) -> Any:
        return self._rbinary_op(other, operator.mul)

    def __truediv__(self, other: Any) -> Any:
        return self._binary_op(other, operator.truediv)

    def __rtruediv__(self, other: Any) -> Any:
        return self._rbinary_op(other, operator.truediv)

    def __floordiv__(self, other: Any) -> Any:
        return self._binary_op(other, operator.floordiv)

    def __rfloordiv__(self, other: Any) -> Any:
        return self._rbinary_op(other, operator.floordiv)

    def __pow__(self, other: Any) -> Any:
        return self._binary_op(other, operator.pow)

    def __rpow__(self, other: Any) -> Any:
        return self._rbinary_op(other, operator.pow)

    def __mod__(self, other: Any) -> Any:
        return self._binary_op(other, operator.mod)

    def __rmod__(self, other: Any) -> Any:
        return self._rbinary_op(other, operator.mod)

    def __matmul__(self, other: Any) -> Any:
        return self._binary_op(other, operator.matmul)

    def __rmatmul__(self, other: Any) -> Any:
        return self._rbinary_op(other, operator.matmul)

    def __neg__(self) -> Any:
        return -self._jnparray

    def __pos__(self) -> Any:
        return +self._jnparray

    def __abs__(self) -> Any:
        return abs(self._jnparray)


@StaticValue.from_.dispatch
def from_(cls: type[StaticValue], value: object, /) -> StaticValue:
    """Convert a value for `StaticQuantity`."""
    return cls(np.asarray(value))


@StaticValue.from_.dispatch
def from_(cls: type[StaticValue], value: StaticValue, /) -> StaticValue:
    """Convert a value for `StaticQuantity`."""
    return value


@StaticValue.from_.dispatch
def from_(cls: type[StaticValue], value: jax.Array | jax.core.Tracer, /) -> StaticValue:
    """Reject JAX arrays for `StaticQuantity`."""
    msg = "StaticQuantity does not accept JAX arrays. Use ParametricQuantity for traced values."
    raise TypeError(msg)


# ==================================================================
# Converters for quantity value field


@plum.dispatch.abstract
def convert_to_quantity_value(obj: Any, /) -> Any:
    """Convert for the value field of an `AbstractQuantity` subclass."""
    raise NotImplementedError  # pragma: no cover


@plum.dispatch
def convert_to_quantity_value(obj: StaticValue, /) -> StaticValue:
    """Allow static values in `ParametricQuantity`-like classes."""
    return obj


@plum.dispatch
def convert_to_quantity_value(obj: quax.ArrayValue, /) -> Any:
    """Convert a `quax.ArrayValue` for the value field.

    >>> import warnings
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue):
    ...     value: Array
    ...
    ...     def aval(self):
    ...         return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...
    ...     def materialise(self):
    ...         return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> with warnings.catch_warnings(record=True, action="always") as w:
    ...     y = convert_to_quantity_value(x)
    >>> y
    MyArray(value=i32[3])
    >>> print(f"Warning caught: {w[-1].message}")
    Warning caught: 'quax.ArrayValue' subclass 'MyArray' ...

    """
    warnings.warn(
        f"'quax.ArrayValue' subclass {type(obj).__name__!r} does not have a registered "
        "converter. Returning the object as is.",
        category=UserWarning,
        stacklevel=2,
    )
    return obj


@plum.dispatch
def convert_to_quantity_value(obj: ArrayLike | list[Any] | tuple[Any, ...], /) -> Array:
    """Convert an array-like object to a `jax.numpy.ndarray`.

    >>> import jax.numpy as jnp
    >>> from unxt.quantity import convert_to_quantity_value

    >>> convert_to_quantity_value([1, 2, 3])
    Array([1, 2, 3], dtype=int32)

    """
    out = jnp.asarray(obj)

    # JAX 0.7.2+ introduces TypedInt, TypedFloat, TypedComplex for
    # type-preserving operations. These are not jax.Array instances but carry
    # dtype information. We need to explicitly convert them to proper arrays
    # while preserving dtype.
    if not isinstance(out, jax.Array):
        out = jnp.asarray(out, dtype=out.dtype)

    return out
