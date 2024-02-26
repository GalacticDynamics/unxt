# pylint: disable=import-error

from typing import Any, TypeVar

import jax
import jax.core
import jax.experimental.array_api as jax_xp
from jax import Device

from array_api_jax_compat._dispatch import dispatcher as dispatcher_
from array_api_jax_compat._types import DType

from ._core import Quantity

T = TypeVar("T")


def dispatcher(f: T) -> T:  # TODO: figure out mypy stub issue.
    """Dispatcher that makes mypy happy."""
    return dispatcher_(f)


# -----------------------------------------------


@dispatcher
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
            stop=stop.to_value(unit) if stop is not None else None,
            step=step.to_value(unit) if step is not None else None,
            dtype=dtype,
            device=device,
        ),
        unit=unit,
    )


# -----------------------------------------------


@dispatcher
def empty_like(x: Quantity, /, *, dtype: Any = None, device: Any = None) -> Quantity:
    out = Quantity(jax_xp.empty_like(x.value, dtype=dtype), unit=x.unit)
    return jax.device_put(out, device=device)


# -----------------------------------------------


@dispatcher
def full_like(
    x: Quantity,
    *,
    fill_value: Any,
    dtype: Any = None,
    device: Any = None,
) -> Quantity:
    return Quantity(
        jax_xp.full_like(x.value, fill_value, dtype=dtype, device=device), unit=x.unit
    )


@dispatcher  # type: ignore[no-redef]
def full_like(
    x: Quantity,
    /,
    fill_value: bool | int | float | complex,
    *,
    dtype: Any = None,
    device: Any = None,
) -> Quantity:
    return Quantity(
        jax_xp.full_like(x.value, fill_value, dtype=dtype, device=device), unit=x.unit
    )


# -----------------------------------------------


@dispatcher
def linspace(
    start: Quantity,
    stop: Quantity,
    num: int,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
    endpoint: bool = True,
) -> Quantity:
    unit = start.unit
    return Quantity(
        jax_xp.linspace(
            start.to_value(unit),
            stop.to_value(unit),
            num=num,
            dtype=dtype,
            device=device,
            endpoint=endpoint,
        ),
        unit=unit,
    )


@dispatcher
def ones_like(x: Quantity, /, *, dtype: Any = None, device: Any = None) -> Quantity:
    out = Quantity(jax_xp.ones_like(x.value, dtype=dtype), unit=x.unit)
    return jax.device_put(out, device=device)


@dispatcher
def zeros_like(x: Quantity, /, *, dtype: Any = None, device: Any = None) -> Quantity:
    out = Quantity(jax_xp.zeros_like(x.value, dtype=dtype), unit=x.unit)
    return jax.device_put(out, device=device)
