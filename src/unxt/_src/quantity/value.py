"""Conversion to quantity value types."""

__all__ = ("convert_to_quantity_value",)

import warnings
from typing import Any, NoReturn

import quax
from jaxtyping import Array, ArrayLike
from plum import dispatch

import quaxed.numpy as jnp

from .base import AbstractQuantity


@dispatch.abstract
def convert_to_quantity_value(obj: Any, /) -> Any:
    """Convert for the value field of an `AbstractQuantity` subclass."""
    raise NotImplementedError  # pragma: no cover


@dispatch
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


@dispatch
def convert_to_quantity_value(obj: ArrayLike | list[Any] | tuple[Any, ...], /) -> Array:
    """Convert an array-like object to a `jax.numpy.ndarray`.

    >>> import jax.numpy as jnp
    >>> from unxt.quantity import convert_to_quantity_value

    >>> convert_to_quantity_value([1, 2, 3])
    Array([1, 2, 3], dtype=int32)

    """
    return jnp.asarray(obj)


@dispatch
def convert_to_quantity_value(obj: AbstractQuantity, /) -> NoReturn:
    """Disallow conversion of `AbstractQuantity` to a value.

    >>> import unxt as u
    >>> from unxt.quantity import convert_to_quantity_value

    >>> try:
    ...     convert_to_quantity_value(u.Quantity(1, "m"))
    ... except TypeError as e:
    ...     print(e)
    Cannot convert 'Quantity[PhysicalType('length')]' to a value.
    For a Quantity, use the `.from_` constructor instead.

    """
    msg = (
        f"Cannot convert '{type(obj).__name__}' to a value. "
        "For a Quantity, use the `.from_` constructor instead."
    )
    raise TypeError(msg)
