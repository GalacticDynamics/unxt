"""Integration tests for uconvert_value JAX transformations and performance.

This module tests uconvert_value with JAX transformations (jit, vmap)
and demonstrates the performance improvement from JIT compilation.
"""

import jax
import jax.numpy as jnp
import numpy as np

import unxt as u


class TestUconvertValueVmapJit:
    """Test uconvert_value with vmap and jit transformations."""

    def test_uconvert_value_vmap_jit_basic(self) -> None:
        """Test basic vmap(jit(uconvert_value)) functionality.

        With vmap(jit(uconvert_value)), we can convert multiple values
        efficiently by:
        - JIT compiling the function once
        - Using vmap to apply it over multiple input values
        - Keeping unit arguments static (not vmapped)
        """
        # Define vmapped and jitted version
        # vmap maps over x (last argument), leaving uto and ufrom unmapped
        # Use static_argnums=(0, 1) since parameters are positional-only
        vmap_jit_uconvert = jax.vmap(
            jax.jit(u.uconvert_value, static_argnums=(0, 1)),
            in_axes=(None, None, 0),  # (uto, ufrom, x)
        )

        # Input values to convert
        values = jnp.array([1.0, 2.0, 5.0, 10.0])

        # Convert km to m
        result = vmap_jit_uconvert(u.unit("m"), u.unit("km"), values)

        # Check results
        expected = jnp.array([1000.0, 2000.0, 5000.0, 10000.0])
        assert np.allclose(result, expected)

    def test_uconvert_value_vmap_jit_with_string_units(self) -> None:
        """Test vmap(jit(uconvert_value)) with string units."""
        vmap_jit_uconvert = jax.vmap(
            jax.jit(u.uconvert_value, static_argnums=(0, 1)),
            in_axes=(None, None, 0),
        )

        values = jnp.array([1000.0, 2000.0, 3000.0])
        result = vmap_jit_uconvert("km", "m", values)

        expected = jnp.array([1.0, 2.0, 3.0])
        assert np.allclose(result, expected)


class TestUconvertValueJitOnly:
    """Test uconvert_value with JIT alone (without vmap)."""

    def test_uconvert_value_jit_single_value(self) -> None:
        """Test JIT compilation of uconvert_value for single value."""
        jit_uconvert = jax.jit(u.uconvert_value, static_argnums=(0, 1))

        result = jit_uconvert(u.unit("m"), u.unit("km"), 5.0)
        assert jnp.isclose(result, 5000.0)

    def test_uconvert_value_jit_with_array(self) -> None:
        """Test JIT compilation of uconvert_value for array values."""
        jit_uconvert = jax.jit(u.uconvert_value, static_argnums=(0, 1))

        values = jnp.array([1.0, 2.0, 3.0])
        result = jit_uconvert(u.unit("m"), u.unit("km"), values)

        expected = jnp.array([1000.0, 2000.0, 3000.0])
        assert np.allclose(result, expected)


class TestUconvertValueVmapOnly:
    """Test uconvert_value with vmap alone (without JIT)."""

    def test_uconvert_value_vmap_vectorization(self) -> None:
        """Test vmap vectorization of uconvert_value."""
        vmap_uconvert = jax.vmap(u.uconvert_value, in_axes=(None, None, 0))

        values = jnp.array([1.0, 2.0, 5.0])
        result = vmap_uconvert(u.unit("m"), u.unit("km"), values)

        expected = jnp.array([1000.0, 2000.0, 5000.0])
        assert np.allclose(result, expected)

    def test_uconvert_value_nested_vmap(self) -> None:
        """Test nested vmap over multiple dimensions."""
        # Nested vmap: outer maps over axis 0, inner maps over axis 0
        # For 2D input, this vmaps over rows, then elements within rows
        vmap_uconvert = jax.vmap(
            jax.vmap(u.uconvert_value, in_axes=(None, None, 0)),
            in_axes=(None, None, 0),
        )

        # 2D array: (batches, values_per_batch)
        values = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        result = vmap_uconvert(u.unit("m"), u.unit("km"), values)

        expected = jnp.array([[1000.0, 2000.0], [3000.0, 4000.0]])
        assert np.allclose(result, expected)
