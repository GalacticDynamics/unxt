"""Benchmark tests for `unxt.uconvert_value` with JAX transformations.

These benchmarks test uconvert_value performance with jit and vmap
using pytest-codspeed for precise measurements.
"""

import jax
import jax.numpy as jnp
import pytest

import unxt as u


@pytest.fixture
def xarray():
    return jnp.linspace(1.0, 1000.0, num=int(1e4), dtype=jnp.float32)


#####################################################################
# Benchmark: uconvert_value with scalar values


@pytest.mark.benchmark(group="uconvert_value", max_time=1.0)
def test_uconvert_value_scalar_string_units(benchmark):
    """Benchmark uconvert_value with scalar value and string units."""
    benchmark(u.uconvert_value, "m", "km", 5.0)


@pytest.mark.benchmark(group="uconvert_value", max_time=1.0)
def test_uconvert_value_scalar_unit_objects(benchmark):
    """Benchmark uconvert_value with scalar value and unit objects."""
    uto = u.unit("m")
    ufrom = u.unit("km")
    benchmark(u.uconvert_value, uto, ufrom, 5.0)


#####################################################################
# Benchmark: uconvert_value with arrays (no JIT)


@pytest.mark.benchmark(group="uconvert_value", max_time=1.0)
def test_uconvert_value_array_no_jit(benchmark, xarray):
    """Benchmark uconvert_value with array (no JIT compilation)."""
    benchmark(u.uconvert_value, u.unit("m"), u.unit("km"), xarray)


#####################################################################
# Benchmark: uconvert_value with vmap only


@pytest.fixture
def func_vmap_uconvert():
    """Create vmap version of uconvert_value."""
    return jax.vmap(u.uconvert_value, in_axes=(None, None, 0))


@pytest.mark.benchmark(group="uconvert_value_vmap", max_time=1.0)
def test_uconvert_value_vmap_no_jit(benchmark, func_vmap_uconvert, xarray):
    """Benchmark vmap(uconvert_value) without JIT."""
    benchmark(func_vmap_uconvert, u.unit("m"), u.unit("km"), xarray)


@pytest.mark.benchmark(group="uconvert_value_vmap", max_time=1.0)
def test_uconvert_value_vmap_jit_compile(xarray):
    """Benchmark vmap(uconvert_value) JIT compilation."""
    vmap_jit_uconvert = jax.vmap(
        jax.jit(u.uconvert_value, static_argnums=(0, 1)),
        in_axes=(None, None, 0),
    )
    _ = vmap_jit_uconvert.__wrapped__.lower(u.unit("m"), u.unit("km"), xarray).compile()


@pytest.mark.benchmark(group="uconvert_value_vmap", max_time=1.0)
def test_uconvert_value_vmap_jit_execute(benchmark, func_vmap_uconvert, xarray):
    """Benchmark vmap(uconvert_value) after JIT compilation."""
    jit_fn = jax.jit(func_vmap_uconvert)
    # Warm up
    _ = jit_fn(u.unit("m"), u.unit("km"), xarray)
    # Benchmark
    benchmark(lambda: jax.block_until_ready(jit_fn(u.unit("m"), u.unit("km"), xarray)))


#####################################################################
# Benchmark: uconvert_value with vmap + jit (static args)


@pytest.fixture
def func_vmap_jit_uconvert():
    """Create vmap(jit(uconvert_value)) with static args."""
    return jax.vmap(
        jax.jit(u.uconvert_value, static_argnums=(0, 1)),
        in_axes=(None, None, 0),
    )


@pytest.mark.benchmark(group="uconvert_value_vmap_jit", max_time=1.0)
def test_uconvert_value_vmap_jit_execute(benchmark, func_vmap_jit_uconvert, xarray):
    """Benchmark vmap(jit(uconvert_value)) with static unit arguments."""
    uto = u.unit("m")
    ufrom = u.unit("km")
    # Warm up
    _ = func_vmap_jit_uconvert(uto, ufrom, xarray)
    # Benchmark
    benchmark(lambda: jax.block_until_ready(func_vmap_jit_uconvert(uto, ufrom, xarray)))


#####################################################################
# Benchmark: Larger array performance


@pytest.mark.benchmark(group="uconvert_value_large", max_time=1.0)
def test_uconvert_value_large_array_no_jit(benchmark, xarray):
    """Benchmark uconvert_value with large array (no JIT)."""
    benchmark(u.uconvert_value, u.unit("m"), u.unit("km"), xarray)


@pytest.mark.benchmark(group="uconvert_value_large", max_time=1.0)
def test_uconvert_value_large_array_vmap_jit(benchmark, func_vmap_jit_uconvert, xarray):
    """Benchmark vmap(jit(uconvert_value)) with large array."""
    uto = u.unit("m")
    ufrom = u.unit("km")
    # Warm up
    _ = func_vmap_jit_uconvert(uto, ufrom, xarray)
    # Benchmark
    benchmark(lambda: jax.block_until_ready(func_vmap_jit_uconvert(uto, ufrom, xarray)))
