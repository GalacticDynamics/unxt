from typing import Any, TypeVar

import jax
import jax.core
import jax.numpy as jnp
from array_api_jax_compat._dispatch import dispatcher as dispatcher_

from ._core import Quantity

T = TypeVar("T")


def dispatcher(f: T) -> T:  # TODO: figure out mypy stub issue.
    """Dispatcher that makes mypy happy."""
    return dispatcher_(f)


@dispatcher
def empty_like(x: Quantity, /, *, dtype: Any = None, device: Any = None) -> Quantity:
    out = Quantity(jnp.empty_like(x.value, dtype=dtype), unit=x.unit)
    return jax.device_put(out, device=device)


@dispatcher
def full_like(
    x: Quantity,
    /,
    fill_value: bool | int | float | complex,
    *,
    dtype: Any = None,
    device: Any = None,
) -> Quantity:
    out = Quantity(jnp.full_like(x.value, fill_value, dtype=dtype), unit=x.unit)
    return jax.device_put(out, device=device)


@dispatcher
def ones_like(x: Quantity, /, *, dtype: Any = None, device: Any = None) -> Quantity:
    out = Quantity(jnp.ones_like(x.value, dtype=dtype), unit=x.unit)
    return jax.device_put(out, device=device)


@dispatcher
def zeros_like(x: Quantity, /, *, dtype: Any = None, device: Any = None) -> Quantity:
    out = Quantity(jnp.zeros_like(x.value, dtype=dtype), unit=x.unit)
    return jax.device_put(out, device=device)
