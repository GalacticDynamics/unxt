"""Edge cases of the unxts.interop.xarray accessors and conversion helpers."""

import pytest
import unxts.interop.xarray as ux
import xarray as xr

import unxt as u


@pytest.mark.parametrize(
    ("fn", "match"),
    [
        (ux.extract_unit_attributes, "Cannot extract unit attributes from type"),
        (ux.extract_units, "Cannot extract units from type"),
    ],
)
def test_extract_from_an_unsupported_type(fn, match):
    with pytest.raises(TypeError, match=match):
        fn(object())


def test_attach_units_to_an_unsupported_type():
    with pytest.raises(TypeError, match="Cannot attach units to type"):
        ux.attach_units(object(), {None: "m"})


def test_strip_units_from_an_unsupported_type():
    with pytest.raises(TypeError, match="Cannot strip units from type"):
        ux.strip_units(object())


def test_quantify_rejects_a_bad_units_argument():
    da = xr.DataArray([1.0, 2.0], dims=["x"])
    with pytest.raises(TypeError, match="units must be a string"):
        da.unxt.quantify(1.0)


def test_extract_units_of_an_unquantified_dataarray():
    """Nothing carries a unit, so nothing is extracted."""
    da = xr.DataArray([1.0, 2.0], dims=["i"], coords={"i": [0, 1]})
    assert ux.extract_units(da) == {}
    assert ux.extract_units(xr.Dataset({"a": ("x", [1.0, 2.0])})) == {}


def _dataarray_with_coord_units() -> xr.DataArray:
    return xr.DataArray(
        [1.0, 2.0],
        dims=["i"],
        coords={"i": [0, 1], "x": ("i", [0.0, 1.0], {"units": "s"})},
        attrs={"units": "m"},
    ).unxt.quantify()


def test_dataarray_round_trip_with_coordinate_units():
    """Coordinate units survive quantify -> dequantify on a DataArray."""
    q = _dataarray_with_coord_units()
    assert ux.extract_units(q)["x"] == u.unit("s")

    back = q.unxt.dequantify()
    assert back.attrs["units"] == "m"
    assert back.coords["x"].attrs["units"] == "s"


def test_dequantify_honours_the_format_spec():
    """``format`` is forwarded to ``builtins.format`` for the unit string."""
    q = _dataarray_with_coord_units()
    back = q.unxt.dequantify(format="latex")
    assert back.attrs["units"] == format(u.unit("m"), "latex")


def test_dataset_quantify_with_explicit_units_mapping():
    """An explicit mapping overrides (and supplements) the ``units`` attributes."""
    ds = xr.Dataset({"a": ("x", [1.0, 2.0], {"units": "m"}), "b": ("x", [3.0, 4.0])})
    q = ds.unxt.quantify({"b": "s"})
    assert u.unit_of(q["a"].data) == u.unit("m")
    assert u.unit_of(q["b"].data) == u.unit("s")


def test_dataset_round_trip_with_coordinate_units():
    """Coordinate units survive quantify -> dequantify on a Dataset."""
    ds = xr.Dataset(
        {"a": ("i", [1.0, 2.0], {"units": "m"})},
        coords={"i": [0, 1], "x": ("i", [0.0, 1.0], {"units": "s"})},
    )
    q = ds.unxt.quantify()
    back = q.unxt.dequantify(format="latex")
    assert back["a"].attrs["units"] == format(u.unit("m"), "latex")
    assert back.coords["x"].attrs["units"] == format(u.unit("s"), "latex")


def test_dequantify_a_dataarray_whose_data_has_no_unit():
    """Only the coordinate carries a unit, so no data-level ``units`` attribute."""
    da = xr.DataArray(
        [1.0, 2.0],
        dims=["i"],
        coords={"i": [0, 1], "x": ("i", [0.0, 1.0], {"units": "s"})},
    ).unxt.quantify()
    back = da.unxt.dequantify()
    assert "units" not in back.attrs
    assert back.coords["x"].attrs["units"] == "s"
