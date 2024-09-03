# pylint: disable=import-error

from collections.abc import Callable
from typing import Any

import jax
import jax.core
import jax.experimental.array_api as jax_xp
from jaxtyping import ArrayLike
from plum import Dispatcher, Function
from plum.parametric import type_unparametrized as type_np

from quaxed.array_api._dispatch import dispatcher as xp_dispatcher
from quaxed.numpy._dispatch import dispatcher as np_dispatcher

from .base import AbstractQuantity
from .core import Quantity
from .functional import ustrip


def chain_dispatchers(*dispatchers: Dispatcher) -> Callable[[Any], Function]:
    """Apply many dispatchers to a function."""

    def decorator(method: Any) -> Function:
        for dispatcher in dispatchers:
            f = dispatcher(method)
        return f

    return decorator


# -----------------------------------------------


@chain_dispatchers(np_dispatcher, xp_dispatcher)
def arange(
    start: AbstractQuantity,
    stop: AbstractQuantity | None = None,
    step: AbstractQuantity | None = None,
    **kwargs: Any,
) -> AbstractQuantity:
    """Return evenly spaced values within a given interval.

    This method is registered in the `np` and `xp` dispatchers.

    Returns
    -------
    `unxt.AbstractQuantity`
        Of the same type as `start`.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> from unxt import Quantity

    >>> xp.arange(Quantity(5, "m"))
    Quantity['length'](Array([0, 1, 2, 3, 4], dtype=int32), unit='m')

    >>> xp.arange(Quantity(5, "m"), Quantity(10, "m"))
    Quantity['length'](Array([5, 6, 7, 8, 9], dtype=int32), unit='m')

    >>> xp.arange(Quantity(5, "m"), Quantity(10, "m"), Quantity(2, "m"))
    Quantity['length'](Array([5, 7, 9], dtype=int32), unit='m')

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


@chain_dispatchers(np_dispatcher, xp_dispatcher)
def empty_like(
    x: AbstractQuantity, /, *, device: Any = None, **kwargs: Any
) -> AbstractQuantity:
    """Return a new array with the same shape and type as a given array.

    This method is registered in the `np` and `xp` dispatchers.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> from unxt import Quantity

    >>> xp.empty_like(Quantity(5, "m"))
    Quantity['length'](Array(0, dtype=int32, ...), unit='m')

    """
    out = type_np(x)(jax_xp.empty_like(x.value, **kwargs), unit=x.unit)
    return jax.device_put(out, device=device)


# -----------------------------------------------


@chain_dispatchers(np_dispatcher, xp_dispatcher)
def full_like(
    x: AbstractQuantity, /, *, fill_value: Any, **kwargs: Any
) -> AbstractQuantity:
    """Return a new array with the same shape and type as a given array.

    This method is registered in the `np` and `xp` dispatchers.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> from unxt import Quantity

    >>> xp.full_like(Quantity(5, "m"), fill_value=Quantity(10, "m"))
    Quantity['length'](Array(10, dtype=int32, ...), unit='m')

    """
    # re-dispatch to the correct implementation
    return full_like(x, fill_value, **kwargs)


@chain_dispatchers(np_dispatcher, xp_dispatcher)  # type: ignore[no-redef]
def full_like(
    x: AbstractQuantity, fill_value: ArrayLike, /, **kwargs: Any
) -> AbstractQuantity:
    """Return a new array with the same shape and type as a given array.

    This method is registered in the `np` and `xp` dispatchers.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> from unxt import Quantity

    >>> xp.full_like(Quantity(5, "m"), 100.0)
    Quantity['length'](Array(100, dtype=int32, ...), unit='m')

    """
    return type_np(x)(jax_xp.full_like(x.value, fill_value, **kwargs), unit=x.unit)


@chain_dispatchers(np_dispatcher, xp_dispatcher)  # type: ignore[no-redef]
def full_like(
    x: AbstractQuantity, fill_value: AbstractQuantity, /, **kwargs: Any
) -> AbstractQuantity:
    """Return a new array with the same shape and type as a given array.

    This method is registered in the `np` and `xp` dispatchers.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> from unxt import Quantity

    >>> xp.full_like(Quantity(5, "m"), Quantity(10, "m"))
    Quantity['length'](Array(10, dtype=int32, ...), unit='m')

    """
    fill_val = ustrip(x.unit, fill_value)
    return type_np(x)(jax_xp.full_like(x.value, fill_val, **kwargs), unit=x.unit)


# -----------------------------------------------


@chain_dispatchers(np_dispatcher, xp_dispatcher)
def linspace(start: Quantity, stop: Quantity, num: Any, /, **kwargs: Any) -> Quantity:
    """Return evenly spaced values within a given interval.

    This method is registered in the `np` and `xp` dispatchers.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> from unxt import Quantity

    >>> xp.linspace(Quantity(0, "m"), Quantity(10, "m"), 5)
    Quantity['length'](Array([ 0. ,  2.5,  5. ,  7.5, 10. ], dtype=float32), unit='m')

    """
    unit = start.unit
    return Quantity(
        jax_xp.linspace(ustrip(unit, start), ustrip(unit, stop), num, **kwargs),
        unit=unit,
    )


@chain_dispatchers(np_dispatcher, xp_dispatcher)
def linspace(start: Quantity, stop: Quantity, /, **kwargs: Any) -> Quantity:
    """Return evenly spaced values within a given interval.

    This method is registered in the `np` and `xp` dispatchers.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> from unxt import Quantity

    >>> xp.linspace(Quantity(0, "m"), Quantity(10, "m"), num=5)
    Quantity['length'](Array([ 0. ,  2.5,  5. ,  7.5, 10. ], dtype=float32), unit='m')

    """
    unit = start.unit
    return Quantity(
        jax_xp.linspace(ustrip(unit, start), ustrip(unit, stop), **kwargs), unit=unit
    )


# -----------------------------------------------


@chain_dispatchers(np_dispatcher, xp_dispatcher)
def ones_like(
    x: AbstractQuantity, /, *, device: Any = None, **kwargs: Any
) -> AbstractQuantity:
    """Return a new array with the same shape and type as a given array.

    This method is registered in the `np` and `xp` dispatchers.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> from unxt import Quantity

    >>> xp.ones_like(Quantity(5, "m"))
    Quantity['length'](Array(1, dtype=int32, ...), unit='m')

    """
    out = type_np(x)(jax_xp.ones_like(x.value, **kwargs), unit=x.unit)
    return jax.device_put(out, device=device)


# -----------------------------------------------


@chain_dispatchers(np_dispatcher, xp_dispatcher)
def zeros_like(
    x: AbstractQuantity, /, *, device: Any = None, **kwargs: Any
) -> AbstractQuantity:
    """Return a new array with the same shape and type as a given array.

    This method is registered in the `np` and `xp` dispatchers.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> from unxt import Quantity

    >>> xp.zeros_like(Quantity(5, "m"))
    Quantity['length'](Array(0, dtype=int32, ...), unit='m')

    """
    out = type_np(x)(jax_xp.zeros_like(x.value, **kwargs), unit=x.unit)
    return jax.device_put(out, device=device)
