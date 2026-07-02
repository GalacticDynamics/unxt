"""Smoke tests for the unxts.interop.xarray public API."""

import unxts.interop.xarray as ux
import xarray as xr

import unxt as u


def test_all_symbols_present():
    for name in ux.__all__:
        assert hasattr(ux, name), f"unxts.interop.xarray missing: {name}"


def test_accessor_registered_on_import():
    da = xr.DataArray([1.0, 2.0])
    assert hasattr(da, "unxt")


def test_attach_and_strip_round_trip():
    da = xr.DataArray([1.0, 2.0])
    attached = ux.attach_units(da, {None: "m"})
    assert isinstance(attached.data, u.AbstractQuantity)
    stripped = ux.strip_units(attached)
    assert not isinstance(stripped.data, u.AbstractQuantity)
