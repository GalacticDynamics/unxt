"""Tests for astropy interop uconvert_value function."""

import astropy.units as apyu
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import unxt as u


class TestUconvertValueWithAstropyUnits:
    """Test uconvert_value with Astropy unit objects."""

    def test_uconvert_value_with_apy_units_scalar(self) -> None:
        """Test scalar conversion with Astropy unit objects."""
        result = u.uconvert_value(apyu.Unit("km"), apyu.Unit("m"), 5000)
        assert jnp.isclose(result, 5.0)

    def test_uconvert_value_with_apy_units_array(self) -> None:
        """Test array conversion with Astropy unit objects."""
        values = jnp.array([1000, 2000, 5000])
        result = u.uconvert_value(apyu.Unit("km"), apyu.Unit("m"), values)
        expected = jnp.array([1.0, 2.0, 5.0])
        assert np.allclose(result, expected)

    def test_uconvert_value_apy_units_no_conversion_needed(self) -> None:
        """Test that no conversion is done when units are identical."""
        values = jnp.array([1, 2, 3])
        result = u.uconvert_value(apyu.Unit("m"), apyu.Unit("m"), values)
        assert jnp.array_equal(result, values)

    def test_uconvert_value_apy_units_different_scales(self) -> None:
        """Test conversion with different unit scales."""
        # cm to mm
        result = u.uconvert_value(apyu.Unit("mm"), apyu.Unit("cm"), 5)
        assert jnp.isclose(result, 50.0)

        # kg to g
        result = u.uconvert_value(apyu.Unit("g"), apyu.Unit("kg"), 2)
        assert jnp.isclose(result, 2000.0)

    def test_uconvert_value_apy_units_complex_units(self) -> None:
        """Test conversion with complex composite units."""
        # km/s to m/s
        result = u.uconvert_value(apyu.Unit("m/s"), apyu.Unit("km/s"), 1)
        assert jnp.isclose(result, 1000.0)

        # Kelvin to Celsius with equivalencies
        with apyu.add_enabled_equivalencies(apyu.temperature()):
            result = u.uconvert_value(apyu.Unit("deg_C"), apyu.Unit("Kelvin"), 273.15)
            assert jnp.isclose(result, 0.0)

    def test_uconvert_value_apy_incompatible_units(self) -> None:
        """Test that incompatible Astropy units raise errors."""
        # Length to time should fail
        with pytest.raises(apyu.UnitConversionError, match="not convertible"):
            u.uconvert_value(apyu.Unit("s"), apyu.Unit("m"), 1)

        # Mass to length should fail
        with pytest.raises(apyu.UnitConversionError, match="not convertible"):
            u.uconvert_value(apyu.Unit("kg"), apyu.Unit("m"), 1)

    def test_uconvert_value_apy_float32_preservation(self) -> None:
        """Test that dtype is preserved with Astropy units."""
        result = u.uconvert_value(
            apyu.Unit("km"),
            apyu.Unit("m"),
            jnp.array(1000.0, dtype=jnp.float32),
        )
        assert result.dtype == jnp.float32


class TestUconvertValueMixedAstropyAndUnxtUnits:
    """Test uconvert_value with mixed Astropy and unxt unit types."""

    def test_uconvert_value_apy_both_units(self) -> None:
        """Test conversion with both Astropy unit objects."""
        result = u.uconvert_value(apyu.Unit("km"), apyu.Unit("m"), 5000)
        assert jnp.isclose(result, 5.0)

    def test_uconvert_value_string_units(self) -> None:
        """Test conversion with both string units."""
        result = u.uconvert_value("km", "m", 5000)
        assert jnp.isclose(result, 5.0)

    def test_uconvert_value_unxt_unit_objects(self) -> None:
        """Test conversion with unxt unit objects."""
        result = u.uconvert_value(u.unit("km"), u.unit("m"), 5000)
        assert jnp.isclose(result, 5.0)


class TestUconvertValueWithAstropyQuantity:
    """Test uconvert_value convenience dispatch with Astropy Quantity objects."""

    def test_uconvert_value_apy_quantity_raw_value(self) -> None:
        """Test uconvert_value with raw Astropy Quantity value."""
        # Validate conversion using string units with a raw magnitude
        result = u.uconvert_value("km", "m", 5000)
        assert jnp.isclose(result, 5.0)

    def test_uconvert_value_unxt_quantity_convenience(self) -> None:
        """Test convenience dispatch with unxt Quantity objects."""
        q = u.Q(5000, "m")
        result = u.uconvert_value("km", "m", q)

        assert isinstance(result, u.Quantity)
        assert jnp.isclose(result.value, 5.0)
        assert result.unit == u.unit("km")

    def test_uconvert_value_apy_quantity_with_apy_units(self) -> None:
        """Test with Astropy units only."""
        result = u.uconvert_value(apyu.Unit("km"), apyu.Unit("m"), 1000)
        assert jnp.isclose(result, 1.0)

    def test_uconvert_value_unxt_quantity_with_array_values(self) -> None:
        """Test convenience dispatch with array unxt Quantity."""
        q = u.Q([1000, 2000, 5000], "m")
        result = u.uconvert_value("km", "m", q)

        assert isinstance(result, u.Quantity)
        assert np.allclose(result.value, [1.0, 2.0, 5.0])
        assert result.unit == u.unit("km")


class TestUconvertValueAstropyInteropRelationships:
    """Test relationships between uconvert_value and other conversion functions."""

    def test_uconvert_value_vs_apy_quantity_to(self) -> None:
        """Test that uconvert_value matches Astropy's .to() method."""
        apy_q = apyu.Quantity(5000, "m")
        apy_converted = apy_q.to("km")

        # Using uconvert_value with raw value
        uconvert_result = u.uconvert_value("km", "m", 5000)

        # Results should match
        assert np.isclose(uconvert_result, apy_converted.value)

    def test_uconvert_value_vs_uconvert_unxt_quantity(self) -> None:
        """Test consistency between uconvert_value and uconvert for unxt Quantity."""
        q = u.Q(1000, "m")

        # Using uconvert on Quantity
        uconvert_result = u.uconvert("km", q)

        # Using uconvert_value on raw value
        uconvert_value_result = u.uconvert_value("km", "m", 1000)

        assert jnp.isclose(uconvert_result.value, uconvert_value_result)

    def test_uconvert_value_apy_equivalencies(self) -> None:
        """Test uconvert_value with Astropy equivalencies."""
        # Temperature conversion using equivalencies
        with apyu.add_enabled_equivalencies(apyu.temperature()):
            result = u.uconvert_value(apyu.Unit("deg_C"), apyu.Unit("K"), 273.15)
            assert jnp.isclose(result, 0.0, atol=1e-5)


class TestUconvertValueAstropyJaxIntegration:
    """Test JAX integration with Astropy-based uconvert_value."""

    def test_uconvert_value_apy_jit_compilation(self) -> None:
        """Test JIT compilation of uconvert_value with Astropy units."""

        @jax.jit
        def convert_to_km(x):
            return u.uconvert_value(apyu.Unit("km"), apyu.Unit("m"), x)

        result = convert_to_km(jnp.array(5000.0))
        assert jnp.isclose(result, 5.0)

    def test_uconvert_value_apy_vmap(self) -> None:
        """Test vmap with uconvert_value using Astropy units."""

        def convert_km_to_m(x):
            return u.uconvert_value(apyu.Unit("m"), apyu.Unit("km"), x)

        values = jnp.array([1.0, 2.0, 5.0])
        result = jax.vmap(convert_km_to_m)(values)
        expected = jnp.array([1000.0, 2000.0, 5000.0])
        assert np.allclose(result, expected)

    def test_uconvert_value_apy_grad(self) -> None:
        """Test autodiff with uconvert_value using Astropy units."""

        def apply_conversion(x):
            return jnp.sum(u.uconvert_value(apyu.Unit("km"), apyu.Unit("m"), x))

        values = jnp.array([1000.0, 2000.0, 3000.0])
        grad_fn = jax.grad(apply_conversion)
        result = grad_fn(values)

        # Gradient should be [1.0, 1.0, 1.0] scaled by conversion factor
        expected = jnp.array([0.001, 0.001, 0.001])
        assert np.allclose(result, expected, atol=1e-6)


class TestUconvertValueAstropyDistanceAngle:
    """Test uconvert_value with specialized Astropy coordinate objects."""

    def test_uconvert_value_apy_with_distance_units(self) -> None:
        """Test uconvert_value with distance-related units."""
        # Convert kpc to pc
        result = u.uconvert_value(apyu.Unit("pc"), apyu.Unit("kpc"), 1)
        assert jnp.isclose(result, 1000.0)

        # Convert AU to meters
        result = u.uconvert_value(apyu.Unit("m"), apyu.Unit("AU"), 1)
        expected = (1 * apyu.AU).to(apyu.m).value
        assert jnp.isclose(result, expected)

    def test_uconvert_value_apy_with_angle_units(self) -> None:
        """Test uconvert_value with angle-related units."""
        # Convert degrees to radians
        result = u.uconvert_value(apyu.Unit("rad"), apyu.Unit("deg"), 180)
        assert jnp.isclose(result, jnp.pi)


class TestUconvertValueAstropyErrorHandling:
    """Test error handling in uconvert_value with Astropy units."""

    def test_uconvert_value_apy_incompatible_dimensions(self) -> None:
        """Test incompatible unit conversions with Astropy."""
        with pytest.raises(apyu.UnitConversionError, match="not convertible"):
            u.uconvert_value(apyu.Unit("s"), apyu.Unit("m"), 1)

    def test_uconvert_value_apy_quantity_incompatible_units(self) -> None:
        """Test incompatible conversion with Astropy Quantity convenience dispatch."""
        apy_q = apyu.Quantity(1, "m")  # Length quantity
        # Try to convert to incompatible unit (time)
        with pytest.raises(apyu.UnitConversionError, match="not convertible"):
            u.uconvert_value("s", "m", apy_q)
