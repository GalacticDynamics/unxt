"""Copyright (c) 2023 Nathaniel Starkman. All rights reserved.

jax-quantity: Quantities in JAX
"""

from collections.abc import Callable, Sequence
from typing import Any, TypeAlias, TypeGuard, overload

import array_api_jax_compat
import jax
from astropy.units import UnitBase
from jax._src.util import wraps
from jax.tree_util import tree_map

from ._core import Quantity


def __getattr__(name: str) -> Any:  # TODO: fuller annotation
    """Forward all other attribute accesses to Quaxified JAX."""
    return getattr(array_api_jax_compat, name)


# =============================================================================
# Grad
# Lightly modified from dfm/jpu

Aux: TypeAlias = Any


def is_quantity(obj: Any) -> TypeGuard[Quantity]:
    return hasattr(obj, "unit") and hasattr(obj, "value")


# -----------------


@overload
def grad(
    fun: Callable[..., Quantity],
    argnums: int | Sequence[int],
    *,
    has_aux: bool = False,
    holomorphic: bool,
    allow_int: bool,
    reduce_axes: Sequence[int],
) -> Callable[..., Quantity]:
    ...


@overload
def grad(
    fun: Callable[..., tuple[Quantity, Aux]],
    argnums: int | Sequence[int],
    *,
    has_aux: bool = True,
    holomorphic: bool,
    allow_int: bool,
    reduce_axes: Sequence[int],
) -> Callable[..., tuple[Quantity, Aux]]:
    ...


def grad(
    fun: Callable[..., Quantity] | Callable[..., tuple[Quantity, Aux]],
    argnums: int | Sequence[int] = 0,
    *,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: Sequence[int] = (),
) -> Callable[..., Quantity] | Callable[..., tuple[Quantity, Aux]]:
    value_and_grad_f = value_and_grad(
        fun,
        argnums,
        has_aux=has_aux,
        holomorphic=holomorphic,
        allow_int=allow_int,
        reduce_axes=reduce_axes,
    )

    docstr = (
        "Gradient of {fun} with respect to positional argument(s) "
        "{argnums}. Takes the same arguments as {fun} but returns the "
        "gradient, which has the same shape as the arguments at "
        "positions {argnums}."
    )

    @wraps(fun, docstr=docstr, argnums=argnums)  # type: ignore[misc]  # untyped decorator
    def grad_f(*args: Any, **kwargs: Any) -> Quantity:
        _, g = value_and_grad_f(*args, **kwargs)
        return g

    @wraps(fun, docstr=docstr, argnums=argnums)  # type: ignore[misc]  # untyped decorator
    def grad_f_aux(*args: Any, **kwargs: Any) -> tuple[Quantity, Aux]:
        (_, aux), g = value_and_grad_f(*args, **kwargs)
        return g, aux

    return grad_f_aux if has_aux else grad_f


# -----------------


@overload
def value_and_grad(
    fun: Callable[..., Quantity],
    argnums: int | Sequence[int],
    *,
    has_aux: bool = False,
    holomorphic: bool,
    allow_int: bool,
    reduce_axes: Sequence[int],
) -> Callable[..., tuple[Quantity, Quantity]]:
    ...


@overload
def value_and_grad(
    fun: Callable[..., tuple[Quantity, Aux]],
    argnums: int | Sequence[int],
    *,
    has_aux: bool = True,
    holomorphic: bool,
    allow_int: bool,
    reduce_axes: Sequence[int],
) -> Callable[..., tuple[tuple[Quantity, Aux], Quantity]]:
    ...


def value_and_grad(
    fun: Callable[..., Quantity] | Callable[..., tuple[Quantity, Aux]],
    argnums: int | Sequence[int] = 0,
    *,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: Sequence[int] = (),
) -> Callable[..., tuple[Quantity, Quantity] | tuple[tuple[Quantity, Aux], Quantity]]:
    # inspired by: https://twitter.com/shoyer/status/1531703890512490499
    docstr = (
        "Value and gradient of {fun} with respect to positional "
        "argument(s) {argnums}. Takes the same arguments as {fun} but "
        "returns a two-element tuple where the first element is the value "
        "of {fun} and the second element is the gradient, which has the "
        "same shape as the arguments at positions {argnums}."
    )

    def fun_wo_units(
        *args: Any, **kwargs: Any
    ) -> tuple[jax.Array, tuple[UnitBase, Aux]]:
        if has_aux:
            result, aux = fun(*args, **kwargs)
        else:
            result = fun(*args, **kwargs)
            aux = None

        if is_quantity(result):
            value = result.value
            unit = result.unit
        else:
            value = result
            unit = None

        return value, (unit, aux)

    value_and_grad_fun = jax.value_and_grad(
        fun_wo_units,
        argnums=argnums,
        has_aux=True,
        holomorphic=holomorphic,
        allow_int=allow_int,
        reduce_axes=reduce_axes,
    )

    @wraps(fun, docstr=docstr, argnums=argnums)  # type: ignore[misc]  # untyped decorator
    def wrapped(
        *args: Any, **kwargs: Any
    ) -> tuple[Quantity, Quantity] | tuple[tuple[Quantity, Aux], Quantity]:
        (result_wo_units, (result_units, aux)), grad = value_and_grad_fun(
            *args, **kwargs
        )

        if result_units is None:
            result = result_wo_units
            grad = tree_map(
                lambda g: (g.value * (1 / g.unit) if is_quantity(g) else g),
                grad,
                is_leaf=is_quantity,
            )

        else:
            result = result_wo_units * result_units
            grad = tree_map(
                lambda g: (
                    g.value * (result_units / g.unit)
                    if is_quantity(g)
                    else g * result_units
                ),
                grad,
                is_leaf=is_quantity,
            )

        if has_aux:
            return (result, aux), grad
        return result, grad

    return wrapped


# =============================================================================
