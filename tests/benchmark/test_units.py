"""Benchmark tests for `unxt.units`."""

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import pytest

import unxt as u

if TYPE_CHECKING:
    import jaxlib


@pytest.fixture
def func_unit() -> "jaxlib._jax.PjitFunction":
    return eqx.filter_jit(u.unit)


@pytest.fixture
def func_unit_of() -> "jaxlib._jax.PjitFunction":
    # need to filter_jit because arg can be a array or other object
    return eqx.filter_jit(u.unit_of)


#####################################################################
# `unit`

args = [(u.unit("meter"),), ("meter",)]


@pytest.mark.parametrize("args", args, ids=str)
@pytest.mark.benchmark(group="units", max_time=1.0)
def test_unit(args):
    """Test calling `unxt.unit`."""
    _ = u.unit(*args)


@pytest.mark.parametrize("args", args, ids=str)
@pytest.mark.benchmark(group="units", max_time=1.0)
def test_unit_jit_compile(func_unit, args):
    """Test the speed of calling the function."""
    _ = func_unit.lower(*args).compile()


@pytest.mark.parametrize("args", args, ids=str)
@pytest.mark.benchmark(group="units", max_time=1.0)
def test_unit_execute(func_unit, args):
    """Test the speed of calling the function."""
    _ = jax.block_until_ready(func_unit(*args))


#####################################################################
# `unit_of`

args = [(u.unit("meter"),), (u.Quantity(1, "m"),), (2,)]


@pytest.mark.parametrize("args", args, ids=str)
@pytest.mark.benchmark(group="units", max_time=1.0)
def test_unit_of(args):
    """Test calling `unxt.unit_of`."""
    _ = u.unit_of(*args)


@pytest.mark.parametrize("args", args, ids=str)
@pytest.mark.benchmark(group="units", max_time=1.0)
def test_unit_of_jit_compile(func_unit_of, args):
    """Test the speed of jitting a function."""
    _ = func_unit_of.lower(*args).compile()


@pytest.mark.parametrize("args", args, ids=str)
@pytest.mark.benchmark(group="units", max_time=1.0)
def test_unit_of_execute(func_unit_of, args):
    """Test the speed of calling the function."""
    _ = jax.block_until_ready(func_unit_of(*args))
