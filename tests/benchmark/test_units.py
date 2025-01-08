"""Benchmark tests for `unxt.units`."""

import jax
import pytest

import unxt as u

METER = u.unit("m")


@pytest.fixture
def func_unit_is_length():
    return lambda x: u.unit(x) == METER


@pytest.fixture
def func_unit_of_length():
    return lambda x: u.unit_of(x) == METER


#####################################################################
# `unit`


@pytest.mark.parametrize(
    "args",
    [
        (u.unit("meter"),),  # -> Unit('meter')
        ("meter",),  # -> Unit('meter')
    ],
)
@pytest.mark.benchmark(group="units", warmup=False)
def test_unit(args):
    """Test calling `unxt.unit`."""
    _ = u.unit(*args)


@pytest.mark.parametrize(
    "args",
    [
        (u.unit("meter"),),  # -> Unit('meter')
        ("meter",),  # -> Unit('meter')
    ],
)
@pytest.mark.benchmark(group="units", warmup=True)
def test_unit_execute(func_unit_is_length, args):
    """Test the speed of calling the function."""
    _ = jax.block_until_ready(func_unit_is_length(*args))


#####################################################################
# `unit_of`


@pytest.mark.parametrize(
    "args",
    [
        (u.unit("meter"),),  # -> Unit('meter')
        (u.Quantity(1, "m"),),  # -> Unit('meter')
        (2,),
    ],
)
@pytest.mark.benchmark(group="units", warmup=False)
def test_unit_of(args):
    """Test calling `unxt.unit_of`."""
    _ = u.unit_of(*args)


@pytest.mark.parametrize(
    "args",
    [
        (u.Quantity(1, "m"),),  # -> Unit('meter')
    ],
)
@pytest.mark.benchmark(group="units", warmup=False)
def test_unit_of_jit_compile(func_unit_of_length, args):
    """Test the speed of jitting a function."""
    _ = jax.jit(func_unit_of_length).lower(*args).compile()
