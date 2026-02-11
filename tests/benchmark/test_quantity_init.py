"""Benchmark tests for Quantity initialization methods."""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import unxt as u
from unxt._src.quantity.value import convert_to_quantity_value
from unxt.units import unit as parse_unit

# ==================== Test ID Generation and Parameters ====================


def _get_id_for_value(value: Any) -> str:
    """Generate test ID from input value's dtype and shape.

    Parameters
    ----------
    value : Any
        The input value (scalar, list, or array)

    Returns
    -------
    str
        Test ID like "int32_scalar", "float32_1d_3", etc.

    """
    # Convert to JAX array to inspect dtype and shape
    arr = jnp.asarray(value)

    # Get dtype name (abbreviated for readability)
    dtype_str = str(arr.dtype)

    # Format shape part
    if arr.shape == ():  # scalar
        shape_str = "scalar"
    elif arr.ndim == 1:  # 1D array
        shape_str = f"1d_{arr.shape[0]}"
    else:  # multi-dimensional
        shape_str = "x".join(map(str, arr.shape))

    return f"{dtype_str}_{shape_str}"


# Pre-generate test parameter sets with IDs
_init_test_params = [
    (1, "m"),
    (1.0, "m"),
    ([1, 2, 3], "m"),
]
_init_test_ids = [_get_id_for_value(value) for value, _ in _init_test_params]

_jit_compile_test_params = [
    (1.0, "m"),
    ([1.0, 2.0, 3.0], "m"),
]
_jit_compile_test_ids = [
    _get_id_for_value(value) for value, _ in _jit_compile_test_params
]

_jit_exec_test_params = [
    (1.0, "m"),
    ([1.0, 2.0, 3.0], "m"),
]
_jit_exec_test_ids = [_get_id_for_value(value) for value, _ in _jit_exec_test_params]


# ==================== Converter-based implementation ====================
# This is the baseline implementation on main (without explicit __init__)


class QuantityWithFieldConverters(u.AbstractQuantity):
    """Quantity using converter-based approach (baseline)."""

    value: "jax.Array | u.quantity.StaticValue" = eqx.field(
        converter=convert_to_quantity_value
    )
    unit: "u.AbstractUnit" = eqx.field(converter=parse_unit)


# ==================== Non-JIT Initialization Benchmarks ====================


@pytest.mark.parametrize(("value", "unit"), _init_test_params, ids=_init_test_ids)
@pytest.mark.benchmark(group="quantity_init", min_rounds=10)
def test_quantity_explicit_init(value, unit):
    """Benchmark Quantity with explicit __init__ method (PR implementation)."""
    u.Quantity(value, unit)


@pytest.mark.parametrize(("value", "unit"), _init_test_params, ids=_init_test_ids)
@pytest.mark.benchmark(group="quantity_init", min_rounds=10)
def test_quantity_converter_based(value, unit):
    """Benchmark Quantity with converter-based approach (baseline)."""
    QuantityWithFieldConverters(value, unit)


# ==================== JIT Compilation Benchmarks ====================


@pytest.fixture
def jitted_quantity_explicit():
    """JIT-compiled Quantity creation with explicit __init__."""
    return eqx.filter_jit(lambda v, unit_str: u.Quantity(v, unit_str))


@pytest.fixture
def jitted_quantity_converter():
    """JIT-compiled Quantity creation with converters."""
    return eqx.filter_jit(lambda v, unit_str: QuantityWithFieldConverters(v, unit_str))


@pytest.mark.parametrize(
    ("value", "unit"), _jit_compile_test_params, ids=_jit_compile_test_ids
)
@pytest.mark.benchmark(group="quantity_init_jit_compile", min_rounds=3)
def test_quantity_explicit_init_jit_compile(value, unit, jitted_quantity_explicit):
    """Benchmark JIT compilation of explicit __init__ Quantity."""
    _ = jitted_quantity_explicit.lower(value, unit).compile()


@pytest.mark.parametrize(
    ("value", "unit"), _jit_compile_test_params, ids=_jit_compile_test_ids
)
@pytest.mark.benchmark(group="quantity_init_jit_compile", min_rounds=3)
def test_quantity_converter_jit_compile(value, unit, jitted_quantity_converter):
    """Benchmark JIT compilation of converter-based Quantity."""
    _ = jitted_quantity_converter.lower(value, unit).compile()


# ==================== JIT Execution Benchmarks ====================


@pytest.mark.parametrize(
    ("value", "unit"), _jit_exec_test_params, ids=_jit_exec_test_ids
)
@pytest.mark.benchmark(group="quantity_init_jit_exec", min_rounds=10)
def test_quantity_explicit_init_jit_execute(value, unit, jitted_quantity_explicit):
    """Benchmark execution of JIT-compiled explicit __init__ Quantity."""
    result = jitted_quantity_explicit(value, unit)
    _ = jax.block_until_ready(result)


@pytest.mark.parametrize(
    ("value", "unit"), _jit_exec_test_params, ids=_jit_exec_test_ids
)
@pytest.mark.benchmark(group="quantity_init_jit_exec", min_rounds=10)
def test_quantity_converter_jit_execute(value, unit, jitted_quantity_converter):
    """Benchmark execution of JIT-compiled converter-based Quantity."""
    result = jitted_quantity_converter(value, unit)
    _ = jax.block_until_ready(result)
