"""Benchmark tests for `unxt.unitsystems`."""

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import pytest

import unxt as u

if TYPE_CHECKING:
    import jaxlib


@pytest.fixture
def func_unitsystem() -> "jaxlib._jax.PjitFunction":
    # The lambda function is necessary because JIT doesn't understand how to
    # introspect the signature of a multiple-dispatch function.
    return eqx.filter_jit(lambda *args: u.unitsystem(*args))


@pytest.fixture
def func_unitsystem_of() -> "jaxlib._jax.PjitFunction":
    return eqx.filter_jit(u.unitsystem_of)


@pytest.fixture
def func_equivalent() -> "jaxlib._jax.PjitFunction":
    return eqx.filter_jit(u.unitsystems.equivalent)


#####################################################################
# `dimension`

args = [
    (u.unitsystem("kpc", "Myr", "Msun", "radian"),),
    (("kpc", "Myr", "Msun", "radian"),),
    ("kpc", "Myr", "Msun", "radian"),
    (),  # -> dimensionless
    (None,),  # -> dimensionless
    ("galactic",),
    (u.unitsystem("galactic"), "candela"),
    (u.unitsystems.StandardUSysFlag, "galactic"),
    (u.unitsystems.DynamicalSimUSysFlag, "m", "kg"),
]


@pytest.mark.parametrize("args", args, ids=str)
@pytest.mark.benchmark(group="unitsystems", max_time=1.0)
def test_unitsystem(args):
    """Test calling `unxt.unitsystem`."""
    _ = u.unitsystem(*args)


@pytest.mark.parametrize("args", args, ids=str)
@pytest.mark.benchmark(group="unitsystems", max_time=1.0)
def test_unitsystem_jit_compile(func_unitsystem, args):
    """Test the speed of jitting."""
    _ = func_unitsystem.lower(*args).compile()


@pytest.mark.parametrize("args", args, ids=str)
@pytest.mark.benchmark(group="unitsystems", max_time=1.0)
def test_unitsystem_execute(func_unitsystem, args):
    """Test the speed of calling the function."""
    _ = jax.block_until_ready(func_unitsystem(*args))


#####################################################################
# `unitsystem_of`


args = [
    (u.unitsystem("kpc", "Myr", "Msun", "radian"),),
    (2,),  # -> None
]


@pytest.mark.parametrize("args", args, ids=str)
@pytest.mark.benchmark(group="unitsystems", max_time=1.0)
def test_unitsystem_of(args):
    """Test calling `unxt.unitsystem_of`."""
    _ = u.unitsystem_of(*args)


@pytest.mark.parametrize("args", args, ids=str)
@pytest.mark.benchmark(group="unitsystems", max_time=1.0)
def test_unitsystem_of_jit_compile(func_unitsystem_of, args):
    """Test the speed of jitting."""
    _ = func_unitsystem_of.lower(*args).compile()


@pytest.mark.parametrize("args", args, ids=str)
@pytest.mark.benchmark(group="unitsystems", max_time=1.0)
def test_unitsystem_of_execute(func_unitsystem_of, args):
    """Test the speed of calling the function."""
    _ = jax.block_until_ready(func_unitsystem_of(*args))


#####################################################################
# `equivalent`

args = [
    (u.unitsystem("kpc", "Myr", "Msun", "radian"), u.unitsystems.galactic),
]


@pytest.mark.parametrize("args", args, ids=str)
@pytest.mark.benchmark(group="unitsystems", max_time=1.0)
def test_equivalent(args):
    """Test calling `unxt.equivalent`."""
    _ = u.unitsystems.equivalent(*args)


@pytest.mark.parametrize("args", args, ids=str)
@pytest.mark.benchmark(group="unitsystems", max_time=1.0)
def test_equivalent_jit_compile(func_equivalent, args):
    """Test the speed of jitting."""
    _ = func_equivalent.lower(*args).compile()


@pytest.mark.parametrize("args", args, ids=str)
@pytest.mark.benchmark(group="unitsystems", max_time=1.0)
def test_equivalent_execute(func_equivalent, args):
    """Test the speed of calling the function."""
    _ = jax.block_until_ready(func_equivalent(*args))
