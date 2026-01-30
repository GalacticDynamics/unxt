"""Tests for Quantity printing with wadler-lindig."""

import jax
import jax.numpy as jnp
import wadler_lindig as wl

import unxt as u


class TestShortName:
    """Test the short_name feature for wadler-lindig printing."""

    def test_quantity_has_short_name(self):
        """Test that Quantity has a short_name class variable."""
        assert hasattr(u.Quantity, "short_name")
        assert u.Quantity.short_name == "Q"

    def test_barequantity_no_short_name(self):
        """Test that BareQuantity doesn't have a short_name or it's None."""
        # It should either not have the attribute or have it as None
        short_name = getattr(u.quantity.BareQuantity, "short_name", None)
        assert short_name is None

    def test_use_short_name_default_false(self):
        """Test that use_short_name defaults to False."""
        q = u.Q([1, 2, 3], "m")
        result = wl.pformat(q)
        assert result.startswith("Quantity")
        assert not result.startswith("Q(")

    def test_use_short_name_true(self):
        """Test that use_short_name=True uses the short name."""
        q = u.Q([1, 2, 3], "m")
        result = wl.pformat(q, use_short_name=True)
        assert result.startswith("Q(")
        assert "unit='m'" in result

    def test_use_short_name_with_include_params(self):
        """Test that use_short_name works with include_params."""
        q = u.Q([1, 2, 3], "m")
        result = wl.pformat(q, use_short_name=True, include_params=True)
        assert result.startswith("Q['length']")

    def test_use_short_name_with_named_unit_false(self):
        """Test that use_short_name works with named_unit=False."""
        q = u.Q([1, 2, 3], "m")
        result = wl.pformat(q, use_short_name=True, named_unit=False)
        assert result.startswith("Q(")
        # Should have unit as positional arg not named
        assert "'m')" in result or ", 'm')" in result

    def test_use_short_name_with_short_arrays(self):
        """Test that use_short_name works with short_arrays."""
        q = u.Q([1, 2, 3], "m")

        # Default short_arrays=True
        result = wl.pformat(q, use_short_name=True, short_arrays=True)
        assert result.startswith("Q(")
        assert "i32[3]" in result

        # short_arrays=False
        result = wl.pformat(q, use_short_name=True, short_arrays=False)
        assert result.startswith("Q(")
        assert "Array(" in result

    def test_use_short_name_with_short_arrays_compact(self):
        """Test that use_short_name works with short_arrays='compact'."""
        q = u.Q([1, 2, 3], "m")
        result = wl.pformat(q, use_short_name=True, short_arrays="compact")
        assert result.startswith("Q(")
        assert "[1, 2, 3]" in result

    def test_bare_quantity_use_short_name_none(self):
        """Test that BareQuantity with use_short_name=True still uses full name."""
        q = u.quantity.BareQuantity([1, 2, 3], "m")
        result = wl.pformat(q, use_short_name=True)
        # Should still use full name since short_name is None
        assert result.startswith("BareQuantity")

    def test_pprint(self):
        """Test that pprint works with use_short_name."""
        q = u.Q([1, 2, 3], "m")
        # This should not raise an error
        wl.pprint(q, use_short_name=True)

    def test_pdoc_method_directly(self):
        """Test calling __pdoc__ directly with use_short_name."""
        q = u.Q([1, 2, 3], "m")

        doc = q.__pdoc__(use_short_name=False)
        formatted = wl.pformat(doc)
        assert formatted.startswith("Quantity")

        doc = q.__pdoc__(use_short_name=True)
        formatted = wl.pformat(doc)


class TestStringConversionWithJIT:
    """Test that str() works on Quantity and Angle inside JAX JIT with tracers."""

    def test_str_quantity_in_jit(self):
        """Test that str(Quantity) works inside jax.jit with tracers.

        When values are tracers inside JIT, str() should work without raising an
        error.  We verify this by calling str() during JIT tracing and returning
        a derived value.
        """

        @jax.jit
        def process_with_str(q: u.Quantity) -> u.Quantity:
            # Call str() on the tracer to verify it doesn't raise
            _ = str(q)
            # Return the quantity multiplied by 2
            return q * 2

        q = u.Q([1.0, 2.0, 3.0], "m")
        result = process_with_str(q)
        assert result.unit == q.unit
        assert jnp.allclose(result.value, q.value * 2)

    def test_str_angle_in_jit(self):
        """Test that str(Angle) works inside jax.jit with tracers.

        When values are tracers inside JIT, str() should work without raising an error.
        """

        @jax.jit
        def process_with_str(angle: u.Angle) -> u.Angle:
            # Call str() on the tracer to verify it doesn't raise
            _ = str(angle)
            # Return the angle multiplied by 2
            return angle * 2

        angle = u.Angle([0.5, 1.0, 1.5], "rad")
        result = process_with_str(angle)
        assert result.unit == angle.unit
        assert jnp.allclose(result.value, angle.value * 2)

    def test_str_quantity_multiple_calls_in_jit(self):
        """Test that str(Quantity) works reliably in multiple JIT calls."""

        @jax.jit
        def process_and_stringify(q: u.Quantity) -> u.Quantity:
            # Multiple str() calls shouldn't affect the computation
            _ = str(q)
            q_doubled = q * 2
            _ = str(q_doubled)
            return q_doubled

        q = u.Q(5.0, "kg")
        result = process_and_stringify(q)
        assert result.unit == q.unit

        assert jnp.allclose(result.value, q.value * 2)
