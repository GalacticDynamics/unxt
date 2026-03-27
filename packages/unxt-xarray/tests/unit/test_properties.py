"""Property-based tests using hypothesis."""

import jax.numpy as jnp
import xarray as xr
from hypothesis import given, strategies as st
from unxt_xarray import attach_units, extract_units, strip_units

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

    @given(dataarray_with_units())
    def test_dequantify_preserves_value(self, da_and_unit):
        """Dequantifying should preserve numerical values."""
        da, _ = da_and_unit
        original_data = da.data

        quantified = da.unxt.quantify()
        dequantified = quantified.unxt.dequantify()

        assert jnp.allclose(original_data, dequantified.data, rtol=1e-5, atol=1e-8)

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
        """Extracting then attaching units through xarray should preserve Quantity."""
        if q.ndim == 0:
            q = u.Quantity(jnp.array([q.value]), q.unit)
        da = xr.DataArray(q, dims=["x"])
        units = extract_units(da)
        stripped = strip_units(da)
        reattached = attach_units(stripped, units)
        assert isinstance(reattached.data, u.Quantity)
        assert jnp.allclose(reattached.data.value, q.value, rtol=1e-5, atol=1e-8)


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
        """Attaching None as unit should return data unchanged via public API."""
        if q.ndim == 0:
            q = u.Quantity(jnp.array([q.value]), q.unit)
        # Strip to get plain array
        value = u.ustrip(q)
        da = xr.DataArray(value, dims=["x"])
        # Attach with no units (empty mapping) -> no-op
        result = attach_units(da, {})
        assert not isinstance(result.data, u.Quantity)
