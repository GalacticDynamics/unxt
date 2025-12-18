"""Benchmark tests for quaxed functions on quantities."""

from collections.abc import Callable
from typing import Any, TypeAlias, TypedDict

import jax
import pytest
from jax._src.stages import Compiled

import quaxed.numpy as jnp

import unxt as u

Args: TypeAlias = tuple[Any, ...]

x = jnp.linspace(0, 1, 1000)
x_nodim = u.Quantity(x, "")
x_length = u.Quantity(x, "m")
x_angle = u.Quantity(x, "rad")


def process_func(func: Callable[..., Any], args: Args) -> tuple[Compiled, Args]:
    """JIT and compile the function."""
    return jax.jit(func), args


class ParameterizationKWArgs(TypedDict):
    """Keyword arguments for a pytest parameterization."""

    argvalues: list[tuple[Callable[..., Any], Args]]
    ids: list[str]


def process_pytest_argvalues(
    process_fn: Callable[[Callable[..., Any], Args], tuple[Callable[..., Any], Args]],
    argvalues: list[tuple[Callable[..., Any], *tuple[Args, ...]]],
) -> ParameterizationKWArgs:
    """Process the argvalues."""
    # Get the ID for each parameterization
    get_dims = lambda args: tuple(str(u.dimension_of(a)) for a in args)
    ids: list[str] = []
    processed_argvalues: list[tuple[Compiled, Args]] = []

    for func, *many_args in argvalues:
        for args in many_args:
            ids.append(f"{func.__name__}-{get_dims(args)}")
            processed_argvalues.append(process_fn(func, args))

    # Process the argvalues and return the parameterization, with IDs
    return {"argvalues": processed_argvalues, "ids": ids}


# TODO: also benchmark BareQuantity
funcs_and_args: list[tuple[Callable[..., Any], *tuple[Args, ...]]] = [
    (jnp.abs, (x_nodim,), (x_length,)),
    (jnp.acos, (x_nodim,)),
    (jnp.acosh, (x_nodim,)),
    (jnp.add, (x_nodim, x_nodim), (x_length, x_length)),
    (jnp.asin, (x_nodim,)),
    (jnp.asinh, (x_nodim,)),
    (jnp.atan, (x_nodim,)),
    (jnp.atan2, (x_nodim, x_nodim)),
    (jnp.atanh, (x_nodim,)),
    # bitwise_and
    # bitwise_left_shift
    # bitwise_invert
    # bitwise_or
    # bitwise_right_shift
    # bitwise_xor
    (jnp.ceil, (x_nodim,), (x_length,)),
    (jnp.conj, (x_nodim,), (x_length,)),
    (jnp.cos, (x_nodim,), (x_angle,)),
    (jnp.cosh, (x_nodim,), (x_angle,)),
    (jnp.divide, (x_nodim, x_nodim), (x_length, x_length)),
    (jnp.equal, (x_nodim, x_nodim), (x_length, x_length)),
    (jnp.exp, (x_nodim,)),
    (jnp.expm1, (x_nodim,)),
    (jnp.floor, (x_nodim,), (x_length,)),
    (jnp.floor_divide, (x_nodim, x_nodim), (x_length, x_length)),
    (jnp.greater, (x_nodim, x_nodim), (x_length, x_length)),
    (jnp.greater_equal, (x_nodim, x_nodim), (x_length, x_length)),
    (jnp.imag, (x_nodim,), (x_length,)),
    (jnp.isfinite, (x_nodim,), (x_length,)),
    (jnp.isinf, (x_nodim,), (x_length,)),
    (jnp.isnan, (x_nodim,), (x_length,)),
    (jnp.less, (x_nodim, x_nodim), (x_length, x_length)),
    (jnp.less_equal, (x_nodim, x_nodim), (x_length, x_length)),
    (jnp.log, (x_nodim,)),
    (jnp.log1p, (x_nodim,)),
    (jnp.log2, (x_nodim,)),
    (jnp.log10, (x_nodim,)),
    (jnp.logaddexp, (x_nodim, x_nodim)),
    # (jnp.logical_and, (x_nodim, x_nodim)),
    # (jnp.logical_not, (x_nodim,)),
    # (jnp.logical_or, (x_nodim, x_nodim)),
    # (jnp.logical_xor, (x_nodim, x_nodim)),
    (jnp.multiply, (x_nodim, x_nodim), (x_length, x_length)),
    (jnp.negative, (x_nodim,), (x_length,)),
    (jnp.not_equal, (x_nodim, x_nodim), (x_length, x_length)),
    (jnp.positive, (x_nodim,), (x_length,)),
    # (jnp.power, (x_nodim, 2.0), (x_length, 2.0)),
    (jnp.real, (x_nodim,), (x_length,)),
    (jnp.remainder, (x_nodim, x_nodim), (x_length, x_length)),
    (jnp.round, (x_nodim,), (x_length,)),
    (jnp.sign, (x_nodim,), (x_length,)),
    (jnp.sin, (x_nodim,), (x_angle,)),
    (jnp.sinh, (x_nodim,), (x_angle,)),
    (jnp.square, (x_nodim,), (x_length,)),
    (jnp.sqrt, (x_nodim,), (x_length,)),
    (jnp.subtract, (x_nodim, x_nodim), (x_length, x_length)),
    (jnp.tan, (x_nodim,), (x_angle,)),
    (jnp.tanh, (x_nodim,), (x_angle,)),
    (jnp.trunc, (x_nodim,), (x_length,)),
]


# =============================================================================


@pytest.mark.parametrize(
    ("func", "args"), **process_pytest_argvalues(process_func, funcs_and_args)
)
@pytest.mark.benchmark(group="quaxed", max_time=1.0)
def test_jit_compile(func, args):
    """Test the speed of jitting a function."""
    _ = func.lower(*args).compile()


@pytest.mark.parametrize(
    ("func", "args"), **process_pytest_argvalues(process_func, funcs_and_args)
)
@pytest.mark.benchmark(group="quaxed", max_time=1.0)
def test_execute(func, args):
    """Test the speed of calling the function."""
    _ = jax.block_until_ready(func(*args))
