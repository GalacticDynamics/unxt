# pylint: disable=import-error

from collections.abc import Callable
from typing import Any, TypeVar

import jax
import jax.core
import jax.experimental.array_api as jax_xp
import numpy as np
from jax import Device
from plum import Dispatcher, Function
from plum.parametric import type_unparametrized as type_np

from quaxed._types import DType
from quaxed.array_api._dispatch import dispatcher as xp_dispatcher
from quaxed.numpy._dispatch import dispatcher as np_dispatcher

from .base import AbstractQuantity
from .core import Quantity

T = TypeVar("T")


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
    start: Quantity,
    stop: Quantity | None = None,
    step: Quantity | None = None,
    *,
    dtype: Any = None,
    device: Any = None,
) -> Quantity:
    unit = start.unit
    return Quantity(
        jax_xp.arange(
            start.value,
            stop=stop.to_units_value(unit) if stop is not None else None,
            step=step.to_units_value(unit) if step is not None else None,
            dtype=dtype,
            device=device,
        ),
        unit=unit,
    )


# -----------------------------------------------


@chain_dispatchers(np_dispatcher, xp_dispatcher)
def empty_like(
    x: AbstractQuantity, /, *, dtype: Any = None, device: Any = None
) -> AbstractQuantity:
    out = type_np(x)(jax_xp.empty_like(x.value, dtype=dtype), unit=x.unit)
    return jax.device_put(out, device=device)


# -----------------------------------------------


@chain_dispatchers(np_dispatcher, xp_dispatcher)
def full_like(
    x: AbstractQuantity,
    /,
    *,
    fill_value: Any,
    dtype: Any = None,
    device: Any = None,
) -> AbstractQuantity:
    return full_like(x, fill_value, dtype=dtype, device=device)


@chain_dispatchers(np_dispatcher, xp_dispatcher)  # type: ignore[no-redef]
def full_like(
    x: AbstractQuantity,
    fill_value: AbstractQuantity,
    /,
    *,
    dtype: Any = None,
    device: Any = None,
) -> AbstractQuantity:
    fill_val = fill_value.to_units_value(x.unit)
    return type_np(x)(
        jax_xp.full_like(x.value, fill_val, dtype=dtype, device=device), unit=x.unit
    )


@chain_dispatchers(np_dispatcher, xp_dispatcher)  # type: ignore[no-redef]
def full_like(
    x: AbstractQuantity,
    fill_value: bool | int | float | complex,
    /,
    *,
    dtype: Any = None,
    device: Any = None,
) -> AbstractQuantity:
    return type_np(x)(
        jax_xp.full_like(x.value, fill_value, dtype=dtype, device=device), unit=x.unit
    )


# -----------------------------------------------


@chain_dispatchers(np_dispatcher, xp_dispatcher)
def linspace(
    start: Quantity,
    stop: Quantity,
    num: int | np.integer,
    /,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
    endpoint: bool = True,
) -> Quantity:
    unit = start.unit
    return Quantity(
        jax_xp.linspace(
            start.to_units_value(unit),
            stop.to_units_value(unit),
            num=num,
            dtype=dtype,
            device=device,
            endpoint=endpoint,
        ),
        unit=unit,
    )


@chain_dispatchers(np_dispatcher, xp_dispatcher)
def ones_like(
    x: AbstractQuantity, /, *, dtype: Any = None, device: Any = None
) -> AbstractQuantity:
    out = type_np(x)(jax_xp.ones_like(x.value, dtype=dtype), unit=x.unit)
    return jax.device_put(out, device=device)


@chain_dispatchers(np_dispatcher, xp_dispatcher)
def zeros_like(
    x: AbstractQuantity, /, *, dtype: Any = None, device: Any = None
) -> AbstractQuantity:
    out = type_np(x)(jax_xp.zeros_like(x.value, dtype=dtype), unit=x.unit)
    return jax.device_put(out, device=device)
