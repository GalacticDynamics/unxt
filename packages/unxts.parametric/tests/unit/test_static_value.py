"""``ParametricQuantity(StaticValue)`` as a JAX static argument.

A ``StaticValue`` must be held by the parametric class (``PQ``), whose ``value``
field is static; the lightweight ``Quantity`` treats its value as a pytree leaf,
so an array-backed ``StaticValue`` there would be rejected as pytree metadata.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from unxts.parametric import PQ

import unxt as u


def test_quantity_static_value_as_jit_static_arg() -> None:
    """ParametricQuantity(StaticValue) works as a static arg to jitted funcs."""
    sv = u.quantity.StaticValue(np.array([2.0, 3.0]))
    q_static = PQ(sv, "m")

    # Define a function that uses the static quantity
    @partial(jax.jit, static_argnames=("q_static",))
    def compute(x, q_static):
        # Access the value from the static quantity
        return x * jnp.asarray(q_static.value)

    # First call - should compile
    x = jnp.array([1.0, 1.0])
    result = compute(x, q_static)
    assert np.allclose(result, np.array([2.0, 3.0]))

    # Second call with same static value - should use cached compilation
    result2 = compute(jnp.array([2.0, 2.0]), q_static)
    assert np.allclose(result2, np.array([4.0, 6.0]))

    # Third call with different static value - should trigger recompilation
    sv_new = u.quantity.StaticValue(np.array([5.0, 7.0]))
    q_static_new = PQ(sv_new, "m")
    result3 = compute(jnp.array([1.0, 1.0]), q_static_new)
    assert np.allclose(result3, np.array([5.0, 7.0]))

    # Verify the quantity is actually being used as static (must be hashable)
    assert isinstance(hash(q_static), int)
    assert isinstance(hash(q_static_new), int)


def test_quantity_static_value_jit_with_operations() -> None:
    """ParametricQuantity(StaticValue) works correctly in jitted operations."""
    sv = u.quantity.StaticValue(np.array([2.0, 3.0]))
    q_static = PQ(sv, "m")

    @partial(jax.jit, static_argnames=("scale",))
    def scale_and_add(x, scale):
        # Convert scale to array for computation
        scale_array = jnp.asarray(scale.value)
        return x + scale_array

    x = jnp.array([1.0, 2.0])
    result = scale_and_add(x, q_static)
    assert np.allclose(result, np.array([3.0, 5.0]))

    # Test with different units to ensure unit is also part of static args
    sv2 = u.quantity.StaticValue(np.array([2.0, 3.0]))
    q_static2 = PQ(sv2, "km")

    result2 = scale_and_add(x, q_static2)
    assert np.allclose(result2, np.array([3.0, 5.0]))

    # Verify they have different hashes (different units)
    assert hash(q_static) != hash(q_static2)
