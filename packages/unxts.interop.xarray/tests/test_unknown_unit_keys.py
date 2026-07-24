"""quantify()/attach_units reject unit specs for names that don't exist.

Regression: a typo in a `**unit_kwargs` name was silently dropped, leaving the
data unquantified with no error.
"""

import pytest
import unxts.interop.xarray  # noqa: F401  # registers the .unxt accessor
import xarray as xr

import unxt as u


def test_dataset_typo_key_raises():
    ds = xr.Dataset({"temperature": ("x", [1.0, 2.0])})
    with pytest.raises(ValueError, match=r"not in the Dataset.*temperatrue"):
        ds.unxt.quantify(temperatrue="K")
    # the correct name still works
    q = ds.unxt.quantify(temperature="K")
    assert u.unit_of(q["temperature"].data) == u.unit("K")


def test_dataarray_typo_coord_key_raises():
    # ``y`` is a *non-dimension* coordinate on dim ``x``. A dimension coordinate
    # would be wrapped in a pandas index by xarray, which drops the unit (a
    # separate, known limitation), so a non-dimension coord is used here to
    # verify that a valid coord name actually quantifies.
    da = xr.DataArray([1.0, 2.0], dims=["x"], coords={"y": ("x", [10.0, 20.0])})
    with pytest.raises(ValueError, match=r"not in the DataArray.*'yy'"):
        da.unxt.quantify(yy="s")
    # valid data + coord names still quantify
    q = da.unxt.quantify("m", y="s")
    assert u.unit_of(q.data) == u.unit("m")
    assert u.unit_of(q.coords["y"].data) == u.unit("s")
