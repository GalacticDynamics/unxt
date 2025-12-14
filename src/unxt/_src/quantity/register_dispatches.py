"""Register dispatches for `quaxed.numpy`."""
# pylint: disable=import-error

__all__: tuple[str, ...] = ()

from typing import Any

import jax
import jax.numpy as jax_xp
from jaxtyping import ArrayLike
from plum import dispatch
from plum.parametric import type_unparametrized as type_np

from .api import ustrip
from .base import AbstractQuantity
from .quantity import Quantity

# -----------------------------------------------


@dispatch
def arange(
    start: AbstractQuantity,
    stop: AbstractQuantity | None = None,
    step: AbstractQuantity | None = None,
    **kwargs: Any,
) -> AbstractQuantity:
    """Return evenly spaced values within a given interval.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from unxt import Quantity

    >>> jnp.arange(Quantity(5, "m"))
    Quantity(Array([0, 1, 2, 3, 4], dtype=int32), unit='m')

    >>> jnp.arange(Quantity(5, "m"), Quantity(10, "m"))
    Quantity(Array([5, 6, 7, 8, 9], dtype=int32), unit='m')

    >>> jnp.arange(Quantity(5, "m"), Quantity(10, "m"), Quantity(2, "m"))
    Quantity(Array([5, 7, 9], dtype=int32), unit='m')

    """
    unit = start.unit
    return type_np(start)(
        jax_xp.arange(
            start.value,
            stop=ustrip(unit, stop) if stop is not None else None,
            step=ustrip(unit, step) if step is not None else None,
            **kwargs,
        ),
        unit=unit,
    )


# -----------------------------------------------


@dispatch
def empty_like(
    x: AbstractQuantity, /, *, device: Any = None, **kwargs: Any
) -> AbstractQuantity:
    """Return a new array with the same shape and type as a given array.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from unxt import Quantity

    >>> jnp.empty_like(Quantity(5, "m"))
    Quantity(Array(0, dtype=int32, ...), unit='m')

    """
    out = type_np(x)(jax_xp.empty_like(x.value, **kwargs), unit=x.unit)
    return jax.device_put(out, device=device)


# -----------------------------------------------


@dispatch
def full(
    shape: Any, fill_value: AbstractQuantity, /, **kwargs: Any
) -> AbstractQuantity:
    """Return a new array of given shape and type, filled with `fill_value`.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from unxt import Quantity

    >>> jnp.full((2, 2), Quantity(5, "m"))
    Quantity(Array([[5, 5], [5, 5]], dtype=int32, ...), unit='m')

    """
    fill_val = ustrip(fill_value.unit, fill_value)
    return Quantity(jax_xp.full(shape, fill_val, **kwargs), unit=fill_value.unit)


# -----------------------------------------------


@dispatch
def full_like(
    x: AbstractQuantity, /, *, fill_value: Any, **kwargs: Any
) -> AbstractQuantity:
    """Return a new array with the same shape and type as a given array.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from unxt import Quantity

    >>> jnp.full_like(Quantity(5, "m"), fill_value=Quantity(10, "m"))
    Quantity(Array(10, dtype=int32, ...), unit='m')

    """
    # re-dispatch to the correct implementation
    return full_like(x, fill_value, **kwargs)


@dispatch
def full_like(
    x: AbstractQuantity, fill_value: ArrayLike, /, **kwargs: Any
) -> AbstractQuantity:
    """Return a new array with the same shape and type as a given array.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from unxt import Quantity

    >>> jnp.full_like(Quantity(5, "m"), 100.0)
    Quantity(Array(100, dtype=int32, ...), unit='m')

    """
    return type_np(x)(jax_xp.full_like(x.value, fill_value, **kwargs), unit=x.unit)


@dispatch
def full_like(
    x: AbstractQuantity, fill_value: AbstractQuantity, /, **kwargs: Any
) -> AbstractQuantity:
    """Return a new array with the same shape and type as a given array.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from unxt import Quantity

    >>> jnp.full_like(Quantity(5, "m"), Quantity(10, "m"))
    Quantity(Array(10, dtype=int32, ...), unit='m')

    """
    fill_val = ustrip(x.unit, fill_value)
    return type_np(x)(jax_xp.full_like(x.value, fill_val, **kwargs), unit=x.unit)


# -----------------------------------------------


@dispatch
def linspace(
    start: AbstractQuantity, stop: AbstractQuantity, num: Any, /, **kwargs: Any
) -> AbstractQuantity:
    """Return evenly spaced values within a given interval.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from unxt import Quantity

    >>> jnp.linspace(Quantity(0, "m"), Quantity(10, "m"), 5)
    Quantity(Array([ 0. ,  2.5,  5. ,  7.5, 10. ], dtype=float32), unit='m')

    """
    unit = start.unit
    return type_np(start)(
        jax_xp.linspace(ustrip(unit, start), ustrip(unit, stop), num, **kwargs),
        unit=unit,
    )


@dispatch
def linspace(
    start: AbstractQuantity, stop: AbstractQuantity, /, **kwargs: Any
) -> AbstractQuantity:
    """Return evenly spaced values within a given interval.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from unxt import Quantity

    >>> jnp.linspace(Quantity(0, "m"), Quantity(10, "m"), num=5)
    Quantity(Array([ 0. ,  2.5,  5. ,  7.5, 10. ], dtype=float32), unit='m')

    """
    unit = start.unit
    return type_np(start)(
        jax_xp.linspace(ustrip(unit, start), ustrip(unit, stop), **kwargs), unit=unit
    )


# -----------------------------------------------


@dispatch
def ones_like(
    x: AbstractQuantity, /, *, device: Any = None, **kwargs: Any
) -> AbstractQuantity:
    """Return a new array with the same shape and type as a given array.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from unxt import Quantity

    >>> jnp.ones_like(Quantity(5, "m"))
    Quantity(Array(1, dtype=int32, ...), unit='m')

    """
    out = type_np(x)(jax_xp.ones_like(x.value, **kwargs), unit=x.unit)
    return jax.device_put(out, device=device)


# -----------------------------------------------


@dispatch
def zeros_like(
    x: AbstractQuantity, /, *, device: Any = None, **kwargs: Any
) -> AbstractQuantity:
    """Return a new array with the same shape and type as a given array.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from unxt import Quantity

    >>> jnp.zeros_like(Quantity(5, "m"))
    Quantity(Array(0, dtype=int32, ...), unit='m')

    """
    out = type_np(x)(jax_xp.zeros_like(x.value, **kwargs), unit=x.unit)
    return jax.device_put(out, device=device)
