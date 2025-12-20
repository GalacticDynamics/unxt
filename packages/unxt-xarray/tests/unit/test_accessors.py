"""Tests for xarray accessors."""

import jax.numpy as jnp
import unxt_xarray  # noqa: F401
import xarray as xr

import unxt as u


class TestDataArrayAccessor:
    """Test the .unxt accessor for DataArray."""

    def test_accessor_exists(self):
        """Test that .unxt accessor is registered."""
        da = xr.DataArray([1.0, 2.0], dims=["x"])
        assert hasattr(da, "unxt")

    def test_quantify_from_attribute(self):
        """Test quantify using units from attribute."""
        da = xr.DataArray([1.0, 2.0, 3.0], dims=["x"], attrs={"units": "m"})
        quantified = da.unxt.quantify()

        assert isinstance(quantified.data, u.Quantity)
        assert quantified.data.unit == u.unit("m")
        assert jnp.allclose(quantified.data.value, jnp.array([1.0, 2.0, 3.0]))

    def test_quantify_explicit_unit(self):
        """Test quantify with explicit unit."""
        da = xr.DataArray([1.0, 2.0, 3.0], dims=["x"])
        quantified = da.unxt.quantify("km")

        assert isinstance(quantified.data, u.Quantity)
        assert quantified.data.unit == u.unit("km")

    def test_quantify_with_coords(self):
        """Test quantify with coordinate units."""
        # Use non-dimension coordinate to preserve attributes
        da = xr.DataArray(
            [1.0, 2.0],
            dims=["i"],
            coords={"i": [0, 1], "x": ("i", [0.0, 1.0], {"units": "s"})},
            attrs={"units": "m"},
        )

        quantified = da.unxt.quantify()

        assert isinstance(quantified.data, u.Quantity)
        assert quantified.data.unit == u.unit("m")
        assert isinstance(quantified.coords["x"].data, u.Quantity)
        assert quantified.coords["x"].data.unit == u.unit("s")

    def test_quantify_kwargs(self):
        """Test quantify with keyword arguments for coordinates."""
        # Use non-dimension coordinate to preserve Quantity
        da = xr.DataArray(
            [1.0, 2.0],
            dims=["i"],
            coords={"i": [0, 1], "x": ("i", [0.0, 1.0])},
            attrs={"units": "m"},
        )

        quantified = da.unxt.quantify(x="s")

        assert isinstance(quantified.coords["x"].data, u.Quantity)
        assert quantified.coords["x"].data.unit == u.unit("s")

    def test_quantify_explicit_overrides_attribute(self):
        """Test that explicit units override attribute units."""
        da = xr.DataArray([1.0, 2.0], dims=["x"], attrs={"units": "m"})
        quantified = da.unxt.quantify("km")

        assert quantified.data.unit == u.unit("km")

    def test_dequantify(self):
        """Test dequantify converts Quantities back to plain arrays."""
        q = u.Quantity([1.0, 2.0, 3.0], "m")
        da = xr.DataArray(q, dims=["x"])

        dequantified = da.unxt.dequantify()

        assert not isinstance(dequantified.data, u.Quantity)
        assert dequantified.attrs["units"] == "m"
        assert jnp.allclose(dequantified.data, jnp.array([1.0, 2.0, 3.0]))

    def test_dequantify_with_coords(self):
        """Test dequantify with coordinate units."""
        q_data = u.Quantity([1.0, 2.0], "m")
        q_coord = u.Quantity([0.0, 1.0], "s")
        # Use non-dimension coordinate to preserve Quantity
        da = xr.DataArray(q_data, dims=["i"], coords={"i": [0, 1], "x": ("i", q_coord)})

        dequantified = da.unxt.dequantify()

        assert dequantified.attrs["units"] == "m"
        assert dequantified.coords["x"].attrs["units"] == "s"

    def test_dequantify_custom_attribute(self):
        """Test dequantify with custom attribute name."""
        q = u.Quantity([1.0, 2.0], "m")
        da = xr.DataArray(q, dims=["x"])

        dequantified = da.unxt.dequantify(unit_attribute="unit")

        assert "unit" in dequantified.attrs
        assert dequantified.attrs["unit"] == "m"
        assert "units" not in dequantified.attrs

    def test_roundtrip(self):
        """Test quantify -> dequantify roundtrip preserves data."""
        da = xr.DataArray(
            [1.0, 2.0, 3.0],
            dims=["x"],
            coords={"x": [0.0, 1.0, 2.0]},
            attrs={"units": "m", "description": "distance"},
        )
        da.coords["x"].attrs["units"] = "s"

        # Quantify
        quantified = da.unxt.quantify()

        # Dequantify
        dequantified = quantified.unxt.dequantify()

        # Check data is preserved
        assert jnp.allclose(dequantified.data, da.data)
        assert dequantified.attrs["units"] == "m"
        assert dequantified.attrs["description"] == "distance"
        assert dequantified.coords["x"].attrs["units"] == "s"


class TestDatasetAccessor:
    """Test the .unxt accessor for Dataset."""

    def test_accessor_exists(self):
        """Test that .unxt accessor is registered."""
        ds = xr.Dataset({"a": ("x", [1.0, 2.0])})
        assert hasattr(ds, "unxt")

    def test_quantify_from_attributes(self):
        """Test quantify using units from attributes."""
        ds = xr.Dataset(
            {
                "a": ("x", [1.0, 2.0], {"units": "m"}),
                "b": ("y", [3.0, 4.0], {"units": "s"}),
            }
        )

        quantified = ds.unxt.quantify()

        assert isinstance(quantified["a"].data, u.Quantity)
        assert quantified["a"].data.unit == u.unit("m")
        assert isinstance(quantified["b"].data, u.Quantity)
        assert quantified["b"].data.unit == u.unit("s")

    def test_quantify_explicit_units(self):
        """Test quantify with explicit units."""
        ds = xr.Dataset({"a": ("x", [1.0, 2.0]), "b": ("y", [3.0, 4.0])})

        quantified = ds.unxt.quantify(a="m", b="s")

        assert isinstance(quantified["a"].data, u.Quantity)
        assert quantified["a"].data.unit == u.unit("m")
        assert isinstance(quantified["b"].data, u.Quantity)
        assert quantified["b"].data.unit == u.unit("s")

    def test_quantify_mixed(self):
        """Test quantify with some variables having units."""
        ds = xr.Dataset(
            {
                "a": ("x", [1.0, 2.0], {"units": "m"}),
                "b": ("y", [3.0, 4.0]),  # No units
            }
        )

        quantified = ds.unxt.quantify()

        assert isinstance(quantified["a"].data, u.Quantity)
        assert not isinstance(quantified["b"].data, u.Quantity)

    def test_quantify_with_coords(self):
        """Test quantify with coordinate units."""
        # Use non-dimension coordinates to preserve attributes
        ds = xr.Dataset(
            {"a": ("i", [1.0, 2.0], {"units": "m"})},
            coords={"i": [0, 1], "x": ("i", [0.0, 1.0], {"units": "s"})},
        )

        quantified = ds.unxt.quantify()

        assert isinstance(quantified["a"].data, u.Quantity)
        assert isinstance(quantified.coords["x"].data, u.Quantity)
        assert quantified.coords["x"].data.unit == u.unit("s")

    def test_dequantify(self):
        """Test dequantify converts Quantities back."""
        q1 = u.Quantity([1.0, 2.0], "m")
        q2 = u.Quantity([3.0, 4.0], "s")
        ds = xr.Dataset({"a": ("x", q1), "b": ("y", q2)})

        dequantified = ds.unxt.dequantify()

        assert not isinstance(dequantified["a"].data, u.Quantity)
        assert not isinstance(dequantified["b"].data, u.Quantity)
        assert dequantified["a"].attrs["units"] == "m"
        assert dequantified["b"].attrs["units"] == "s"

    def test_dequantify_with_coords(self):
        """Test dequantify preserves coordinate units."""
        q_data = u.Quantity([1.0, 2.0], "m")
        q_coord = u.Quantity([0.0, 1.0], "s")
        # Use non-dimension coordinate to preserve Quantity
        ds = xr.Dataset(
            {"a": ("i", q_data)},
            coords={"i": [0, 1], "x": ("i", q_coord)},
        )

        dequantified = ds.unxt.dequantify()

        assert dequantified["a"].attrs["units"] == "m"
        assert dequantified.coords["x"].attrs["units"] == "s"

    def test_roundtrip(self):
        """Test quantify -> dequantify roundtrip."""
        # Use non-dimension coordinate to preserve attributes
        ds = xr.Dataset(
            {
                "temperature": ("i", [20.0, 25.0, 30.0], {"units": "K"}),
                "pressure": ("i", [1.0, 1.1, 1.2], {"units": "Pa"}),
            },
            coords={"i": [0, 1, 2], "x": ("i", [0.0, 1.0, 2.0], {"units": "m"})},
        )

        # Quantify
        quantified = ds.unxt.quantify()

        # Dequantify
        dequantified = quantified.unxt.dequantify()

        # Check preservation
        assert dequantified["temperature"].attrs["units"] == "K"
        assert dequantified["pressure"].attrs["units"] == "Pa"
        assert dequantified.coords["x"].attrs["units"] == "m"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_quantify_no_units_raises(self):
        """Test that quantify without units and no attributes works (no-op)."""
        da = xr.DataArray([1.0, 2.0], dims=["x"])
        # Should work but not quantify anything
        result = da.unxt.quantify()
        assert not isinstance(result.data, u.Quantity)

    def test_dequantify_plain_array(self):
        """Test dequantify on plain array (no quantities)."""
        da = xr.DataArray([1.0, 2.0], dims=["x"])
        dequantified = da.unxt.dequantify()

        assert not isinstance(dequantified.data, u.Quantity)
        assert "units" not in dequantified.attrs

    def test_preserves_other_attributes(self):
        """Test that other attributes are preserved."""
        da = xr.DataArray(
            [1.0, 2.0],
            dims=["x"],
            attrs={"units": "m", "long_name": "distance", "valid_range": [0, 100]},
        )

        quantified = da.unxt.quantify()
        dequantified = quantified.unxt.dequantify()

        assert dequantified.attrs["long_name"] == "distance"
        assert dequantified.attrs["valid_range"] == [0, 100]

    def test_preserves_name(self):
        """Test that DataArray name is preserved."""
        da = xr.DataArray(
            [1.0, 2.0], dims=["x"], name="velocity", attrs={"units": "m/s"}
        )

        quantified = da.unxt.quantify()
        dequantified = quantified.unxt.dequantify()

        assert quantified.name == "velocity"
        assert dequantified.name == "velocity"
