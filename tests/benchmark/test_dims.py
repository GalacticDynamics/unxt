"""Benchmark tests for quaxified jax."""

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import pytest

import unxt as u

if TYPE_CHECKING:
    import jaxlib


@pytest.fixture
def func_dimension() -> "jaxlib._jax.PjitFunction":
    return eqx.filter_jit(u.dimension)


@pytest.fixture
def func_dimension_of() -> "jaxlib._jax.PjitFunction":
    return eqx.filter_jit(u.dimension_of)


#####################################################################
# `dimension`

args = [(u.dimension("length"),), ("length",)]


@pytest.mark.parametrize("args", args, ids=str)
@pytest.mark.benchmark(group="dimensions", max_time=1.0)
def test_dimension(args):
    """Test calling `unxt.dimension`."""
    _ = u.dimension(*args)


@pytest.mark.parametrize("args", args, ids=str)
@pytest.mark.benchmark(group="dimensions", max_time=1.0)
def test_dimension_jit_compile(func_dimension, args):
    """Test the speed of jitting."""
    _ = func_dimension.lower(*args).compile()


@pytest.mark.parametrize("args", args, ids=str)
@pytest.mark.benchmark(group="dimensions", max_time=1.0)
def test_dimension_execute(func_dimension, args):
    """Test the speed of calling the function."""
    _ = jax.block_until_ready(func_dimension(*args))


#####################################################################
# `dimension_of`


args = [
    (u.dimension("length"),),  # -> Dimension('length')
    (u.unit("m"),),  # -> Dimension('length')
    (u.Quantity(1, "m"),),  # -> Dimension('length')
    (2,),  # -> None
]


@pytest.mark.parametrize("args", args, ids=str)
@pytest.mark.benchmark(group="dimensions", max_time=1.0)
def test_dimension_of(args):
    """Test calling `unxt.dimension_of`."""
    _ = u.dimension_of(*args)


@pytest.mark.parametrize("args", args, ids=str)
@pytest.mark.benchmark(group="dimensions", max_time=1.0)
def test_dimension_of_jit_compile(func_dimension_of, args):
    """Test the speed of jitting."""
    _ = func_dimension_of.lower(*args).compile()


@pytest.mark.parametrize("args", args, ids=str)
@pytest.mark.benchmark(group="dimensions", max_time=1.0)
def test_dimension_of_execute(func_dimension_of, args):
    """Test the speed of calling the function."""
    _ = jax.block_until_ready(func_dimension_of(*args))
