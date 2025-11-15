"""Flags for quantity operations."""

__all__ = ("AllowValue",)

from typing import Any, NoReturn

from plum import dispatch

from . import api
from .base import AbstractQuantity


class AllowValue:
    """A flag to allow a value to be passed through `unxt.ustrip`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> from unxt.quantity import AllowValue

    >>> x = jnp.array(1)
    >>> y = u.ustrip(AllowValue, "km", x)
    >>> y is x
    True

    >>> u.ustrip(AllowValue, "km", u.Quantity(1000, "m"))
    Array(1., dtype=float32, ...)

    This is a flag, so it cannot be instantiated.

    >>> try:
    ...     AllowValue()
    ... except TypeError as e:
    ...     print(e)
    Cannot instantiate AllowValue

    """

    def __new__(cls) -> NoReturn:
        msg = "Cannot instantiate AllowValue"
        raise TypeError(msg)


@dispatch
def ustrip(flag: type[AllowValue], unit: Any, x: Any, /) -> Any:
    """Strip the units from a value. This is a no-op.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> from unxt.quantity import AllowValue

    >>> x = jnp.array(1)
    >>> y = u.ustrip(AllowValue, "km", x)
    >>> y is x
    True

    >>> x = 1_000
    >>> y = u.ustrip(AllowValue, "km", x)
    >>> y is x
    True

    >>> x = "hello"
    >>> y = u.ustrip(AllowValue, "km", x)
    >>> y is x
    True

    """
    return x


@dispatch
def ustrip(flag: type[AllowValue], x: Any, /) -> Any:
    """Strip the units from a value. This is a no-op.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> from unxt.quantity import AllowValue

    >>> x = jnp.array(1)
    >>> y = u.ustrip(AllowValue, x)
    >>> y is x
    True

    >>> x = 1_000
    >>> y = u.ustrip(AllowValue, x)
    >>> y is x
    True

    >>> x = "hello"
    >>> y = u.ustrip(AllowValue, x)
    >>> y is x
    True

    """
    return x


@dispatch  # TODO: type annotate by value
def ustrip(flag: type[AllowValue], unit: Any, x: AbstractQuantity, /) -> Any:
    """Strip the units from a quantity.

    Examples
    --------
    >>> import unxt as u
    >>> from unxt.quantity import AllowValue
    >>> q = u.Quantity(1000, "m")
    >>> u.ustrip(AllowValue, "km", q)
    Array(1., dtype=float32, ...)

    """
    return api.ustrip(unit, x)


@dispatch
def ustrip(flag: type[AllowValue], x: AbstractQuantity, /) -> Any:
    """Strip the units from a quantity.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> from unxt.quantity import AllowValue

    >>> x = u.Quantity(1, "kpc")
    >>> y = u.ustrip(AllowValue, x)
    >>> not isinstance(y, u.Quantity)
    True
    >>> y == 1
    Array(True, dtype=bool, ...)

    """
    return api.ustrip(x)
