"""Benchmark tests for quaxified jax."""

import jax
import pytest

import unxt as u

LENGTH = u.dimension("length")


@pytest.fixture
def func_dimension_is_length():
    return lambda x: u.dimension(x) == LENGTH


@pytest.fixture
def func_dimension_of_length():
    return lambda x: u.dimension_of(x) == LENGTH


#####################################################################
# `dimension`


@pytest.mark.parametrize(
    "args",
    [
        (u.dimension("length"),),  # -> Dimension('length')
        ("length",),  # -> Dimension('length')
    ],
)
@pytest.mark.benchmark(group="dimensions", warmup=False)
def test_dimension(args):
    """Test calling `unxt.dimension`."""
    _ = u.dimension(*args)


@pytest.mark.parametrize(
    "args",
    [
        (u.dimension("length"),),  # -> Dimension('length')
        ("length",),  # -> Dimension('length')
    ],
)
@pytest.mark.benchmark(group="dimensions", warmup=True)
def test_dimension_execute(func_dimension_is_length, args):
    """Test the speed of calling the function."""
    _ = jax.block_until_ready(func_dimension_is_length(*args))


#####################################################################
# `dimension_of`


@pytest.mark.parametrize(
    "args",
    [
        (u.dimension("length"),),  # -> Dimension('length')
        (u.unit("m"),),  # -> Dimension('length')
        (u.Quantity(1, "m"),),  # -> Dimension('length')
        (2,),  # -> None
    ],
)
@pytest.mark.benchmark(group="dimensions", warmup=False)
def test_dimension_of(args):
    """Test calling `unxt.dimension_of`."""
    _ = u.dimension_of(*args)


@pytest.mark.parametrize(
    "args",
    [
        (u.Quantity(1, "m"),),  # -> Dimension('length')
    ],
)
@pytest.mark.benchmark(group="dimensions", warmup=False)
def test_dimension_of_jit_compile(func_dimension_of_length, args):
    """Test the speed of jitting."""
    _ = jax.jit(func_dimension_of_length).lower(*args).compile()


@pytest.mark.parametrize(
    "args",
    [
        (u.dimension("length"),),  # -> Dimension('length')
        (u.unit("m"),),  # -> Dimension('length')
        (u.Quantity(1, "m"),),  # -> Dimension('length')
        (2,),  # -> None
    ],
)
@pytest.mark.benchmark(group="dimensions", warmup=True)
def test_dimension_of_execute(func_dimension_of_length, args):
    """Test the speed of calling the function."""
    _ = jax.block_until_ready(func_dimension_of_length(*args))
