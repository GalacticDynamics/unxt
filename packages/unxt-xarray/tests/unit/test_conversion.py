"""Tests for conversion functions."""

import jax.numpy as jnp
import pytest
import xarray as xr
from unxt_xarray import (
    attach_units,
    extract_unit_attributes,
    extract_units,
    strip_units,
)

import unxt as u


class TestSentinelIsNone:
    """Verify the data-array key is None, not a magic string."""

    def test_extract_unit_attributes_uses_none_key(self):
        """Test that extract_unit_attributes uses None key for data array."""
        da = xr.DataArray([1.0], dims=["x"], attrs={"units": "m"})
        units = extract_unit_attributes(da)
        assert None in units
        assert units[None] == u.unit("m")

    def test_extract_unit_attributes_no_sentinel_string(self):
        """Test that extract_unit_attributes does not use a magic string key."""
        da = xr.DataArray([1.0], dims=["x"], attrs={"units": "m"})
        units = extract_unit_attributes(da)
        assert "<this-array>" not in units

    def test_extract_units_uses_none_key(self):
        """Test that extract_units uses None key for data array."""
        q = u.Quantity([1.0], "m")
        da = xr.DataArray(q, dims=["x"])
        units = extract_units(da)
        assert None in units
        assert units[None] == u.unit("m")

    def test_extract_units_no_sentinel_string(self):
        """Test that extract_units does not use a magic string key."""
        q = u.Quantity([1.0], "m")
        da = xr.DataArray(q, dims=["x"])
        units = extract_units(da)
        assert "<this-array>" not in units

    def test_attach_units_accepts_none_key(self):
        """Test that attach_units accepts None key for data array."""
        da = xr.DataArray([1.0, 2.0], dims=["x"])
        quantified = attach_units(da, {None: "m"})
        assert isinstance(quantified.data, u.Quantity)
        assert quantified.data.unit == u.unit("m")


class TestExtractUnitAttributes:
    """Test extracting unit attributes from xarray objects."""

    def test_dataarray_with_units_attr(self):
        """Test extracting units attribute from DataArray."""
        da = xr.DataArray([1.0, 2.0], dims=["x"], attrs={"units": "m"})
        units = extract_unit_attributes(da)

        assert units[None] == u.unit("m")

    def test_dataarray_with_coord_units(self):
        """Test extracting units from DataArray coordinates."""
        da = xr.DataArray(
            [1.0, 2.0],
            dims=["i"],
            coords={"i": [0, 1], "x": ("i", [0.0, 1.0], {"units": "s"})},
            attrs={"units": "m"},
        )
        units = extract_unit_attributes(da)

        assert units[None] == u.unit("m")
        assert units["x"] == u.unit("s")

    def test_dataarray_no_units(self):
        """Test DataArray without units returns empty dict."""
        da = xr.DataArray([1.0, 2.0], dims=["x"])
        units = extract_unit_attributes(da)

        assert len(units) == 0

    def test_dataset_with_units(self):
        """Test extracting units from Dataset."""
        ds = xr.Dataset(
            {
                "a": ("x", [1.0, 2.0], {"units": "m"}),
                "b": ("y", [3.0, 4.0], {"units": "s"}),
            }
        )
        units = extract_unit_attributes(ds)

        assert units["a"] == u.unit("m")
        assert units["b"] == u.unit("s")

    def test_dataset_mixed_units(self):
        """Test Dataset with some variables having units."""
        ds = xr.Dataset(
            {
                "a": ("x", [1.0, 2.0], {"units": "m"}),
                "b": ("y", [3.0, 4.0]),  # No units
            }
        )
        units = extract_unit_attributes(ds)

        assert units["a"] == u.unit("m")
        assert "b" not in units

    def test_invalid_type(self):
        """Test that invalid type raises TypeError."""
        with pytest.raises(TypeError, match="Cannot extract unit attributes"):
            extract_unit_attributes([1, 2, 3])  # type: ignore[arg-type]


class TestExtractUnits:
    """Test extracting units from Quantities in xarray objects."""

    def test_dataarray_with_quantity(self):
        """Test extracting units from Quantity in DataArray."""
        q = u.Quantity([1.0, 2.0], "m")
        da = xr.DataArray(q, dims=["x"])
        units = extract_units(da)

        assert units[None] == u.unit("m")

    def test_dataarray_with_quantity_coords(self):
        """Test extracting units from Quantity coordinates."""
        q_data = u.Quantity([1.0, 2.0], "m")
        q_coord = u.Quantity([0.0, 1.0], "s")
        # Use non-dimension coordinate to preserve Quantity
        da = xr.DataArray(q_data, dims=["i"], coords={"i": [0, 1], "x": ("i", q_coord)})
        units = extract_units(da)

        assert units[None] == u.unit("m")
        assert units["x"] == u.unit("s")

    def test_dataarray_no_quantities(self):
        """Test DataArray without Quantities returns empty dict."""
        da = xr.DataArray([1.0, 2.0], dims=["x"])
        units = extract_units(da)

        assert len(units) == 0

    def test_dataset_with_quantities(self):
        """Test extracting units from Dataset with Quantities."""
        q1 = u.Quantity([1.0, 2.0], "m")
        q2 = u.Quantity([3.0, 4.0], "s")
        ds = xr.Dataset({"a": ("x", q1), "b": ("y", q2)})
        units = extract_units(ds)

        assert units["a"] == u.unit("m")
        assert units["b"] == u.unit("s")

    def test_invalid_type(self):
        """Test that invalid type raises TypeError."""
        with pytest.raises(TypeError, match="Cannot extract units"):
            extract_units([1, 2, 3])  # type: ignore[arg-type]


class TestAttachUnits:
    """Test attaching units to xarray objects."""

    def test_attach_to_dataarray(self):
        """Test attaching units to DataArray."""
        da = xr.DataArray([1.0, 2.0], dims=["x"])
        quantified = attach_units(da, {None: "m"})

        assert isinstance(quantified.data, u.Quantity)
        assert quantified.data.unit == u.unit("m")

    def test_attach_to_dataarray_coords(self):
        """Test attaching units to DataArray and coordinates."""
        da = xr.DataArray(
            [1.0, 2.0], dims=["i"], coords={"i": [0, 1], "x": ("i", [0.0, 1.0])}
        )
        quantified = attach_units(da, {None: "m", "x": "s"})

        assert isinstance(quantified.data, u.Quantity)
        assert quantified.data.unit == u.unit("m")
        assert isinstance(quantified.coords["x"].data, u.Quantity)
        assert quantified.coords["x"].data.unit == u.unit("s")

    def test_attach_to_dataset(self):
        """Test attaching units to Dataset."""
        ds = xr.Dataset({"a": ("x", [1.0, 2.0]), "b": ("y", [3.0, 4.0])})
        quantified = attach_units(ds, {"a": "m", "b": "s"})

        assert isinstance(quantified["a"].data, u.Quantity)
        assert quantified["a"].data.unit == u.unit("m")
        assert isinstance(quantified["b"].data, u.Quantity)
        assert quantified["b"].data.unit == u.unit("s")

    def test_attach_partial_units(self):
        """Test attaching units to only some variables."""
        ds = xr.Dataset({"a": ("x", [1.0, 2.0]), "b": ("y", [3.0, 4.0])})
        quantified = attach_units(ds, {"a": "m"})

        assert isinstance(quantified["a"].data, u.Quantity)
        assert not isinstance(quantified["b"].data, u.Quantity)

    def test_invalid_type_raises(self):
        """Invalid type raises TypeError."""
        with pytest.raises(TypeError):
            attach_units("not xarray", {})  # type: ignore[arg-type]


class TestStripUnits:
    """Test stripping units from xarray objects."""

    def test_strip_from_dataarray(self):
        """Test stripping units from DataArray."""
        q = u.Quantity([1.0, 2.0], "m")
        da = xr.DataArray(q, dims=["x"])
        stripped = strip_units(da)

        assert not isinstance(stripped.data, u.Quantity)
        assert jnp.allclose(stripped.data, jnp.array([1.0, 2.0]))

    def test_strip_from_dataarray_with_coords(self):
        """Test stripping units from DataArray and coordinates."""
        q_data = u.Quantity([1.0, 2.0], "m")
        q_coord = u.Quantity([0.0, 1.0], "s")
        da = xr.DataArray(q_data, dims=["x"], coords={"x": q_coord})
        stripped = strip_units(da)

        assert not isinstance(stripped.data, u.Quantity)
        assert not isinstance(stripped.coords["x"].data, u.Quantity)

    def test_strip_from_dataset(self):
        """Test stripping units from Dataset."""
        q1 = u.Quantity([1.0, 2.0], "m")
        q2 = u.Quantity([3.0, 4.0], "s")
        ds = xr.Dataset({"a": ("x", q1), "b": ("y", q2)})
        stripped = strip_units(ds)

        assert not isinstance(stripped["a"].data, u.Quantity)
        assert not isinstance(stripped["b"].data, u.Quantity)

    def test_invalid_type_raises(self):
        """Invalid type raises TypeError."""
        with pytest.raises(TypeError):
            strip_units("not xarray")  # type: ignore[arg-type]
