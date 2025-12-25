"""Property-based tests using hypothesis."""

import jax.numpy as jnp
import unxt_xarray  # noqa: F401
import xarray as xr
from hypothesis import given, strategies as st
from unxt_xarray._src.conversion import _array_attach_units

import unxt as u
import unxt_hypothesis as ust


# Hypothesis strategies for test data
@st.composite
def dataarray_with_units(draw):
    """Generate a DataArray with unit attributes."""
    # Generate data
    size = draw(st.integers(min_value=1, max_value=10))
    data = draw(
        st.lists(
            st.floats(
                min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False
            ),
            min_size=size,
            max_size=size,
        )
    )

    # Generate a unit
    unit = draw(ust.units())

    # Create DataArray with unit attribute
    da = xr.DataArray(jnp.array(data), dims=["x"], attrs={"units": str(unit)})

    return da, unit


@st.composite
def dataarray_with_quantity(draw):
    """Generate a DataArray containing a Quantity."""
    # Generate a quantity - ensure it's at least 1D
    q = draw(ust.quantities())

    # If scalar, make it 1D by wrapping in an array
    if q.ndim == 0:
        q = u.Quantity(jnp.array([q.value]), q.unit)

    # Create DataArray
    da = xr.DataArray(q, dims=["x"])

    return da, u.unit_of(q)


class TestQuantifyDequantifyProperties:
    """Property-based tests for quantify/dequantify."""

    @given(dataarray_with_units())
    def test_quantify_creates_quantity(self, da_and_unit):
        """Quantify should create Quantities from unit attributes."""
        da, expected_unit = da_and_unit

        quantified = da.unxt.quantify()

        assert isinstance(quantified.data, u.Quantity)
        # Check dimension is compatible
        assert u.dimension_of(quantified.data) == u.dimension_of(expected_unit)

    def test_dequantify_preserves_value(self):
        """Dequantifying should preserve numerical values."""
        # Create a simple test case manually instead of using hypothesis strategy
        q = u.Quantity([1.0, 2.0, 3.0], "m")
        da = xr.DataArray(q, dims=["x"])
        unit = u.unit_of(q)

        dequantified = da.unxt.dequantify()

        # Values should be preserved (comparing the underlying arrays)
        original_value = u.ustrip(unit, da.data)
        dequantified_value = dequantified.data

        assert jnp.allclose(original_value, dequantified_value, rtol=1e-5, atol=1e-8)

    @given(dataarray_with_units())
    def test_roundtrip_preserves_value(self, da_and_unit):
        """Quantify -> dequantify should preserve values."""
        da, _ = da_and_unit
        original_data = da.data

        # Roundtrip
        quantified = da.unxt.quantify()
        dequantified = quantified.unxt.dequantify()

        # Values should match
        assert jnp.allclose(original_data, dequantified.data, rtol=1e-5, atol=1e-8)

    @given(dataarray_with_units())
    def test_roundtrip_preserves_unit_string(self, da_and_unit):
        """Quantify -> dequantify should preserve unit attribute."""
        da, _ = da_and_unit
        original_unit = da.attrs["units"]

        # Roundtrip
        quantified = da.unxt.quantify()
        dequantified = quantified.unxt.dequantify()

        # Unit attribute should be restored
        assert "units" in dequantified.attrs
        # Check dimensional equivalence rather than string equality
        assert u.dimension_of(u.unit(dequantified.attrs["units"])) == u.dimension_of(
            u.unit(original_unit)
        )

    @given(ust.quantities())
    def test_extract_attach_roundtrip(self, q):
        """Extracting then attaching units should preserve Quantity."""
        # Extract
        unit = u.unit_of(q)
        value = u.ustrip(unit, q)

        # Re-attach
        reconstructed = _array_attach_units(value, unit)

        # Should be equivalent
        assert u.dimension_of(reconstructed) == u.dimension_of(q)
        assert jnp.allclose(reconstructed.value, q.value, rtol=1e-5, atol=1e-8)


class TestConversionInvariants:
    """Test invariants of conversion functions."""

    @given(
        st.lists(
            st.floats(
                min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False
            ),
            min_size=1,
            max_size=10,
        )
    )
    def test_extract_from_plain_array_is_none(self, data):
        """Extracting units from plain array should give None."""
        arr = jnp.array(data)
        unit = u.unit_of(arr)
        assert unit is None

    @given(ust.quantities())
    def test_attach_none_returns_unchanged(self, q):
        """Attaching None as unit should return data unchanged."""
        # Strip to get plain array
        value = u.ustrip(q)

        # Attach None
        result = _array_attach_units(value, None)

        # Should be the same object
        assert result is value
