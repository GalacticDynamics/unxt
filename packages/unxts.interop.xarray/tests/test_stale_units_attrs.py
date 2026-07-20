"""Regression tests: ``quantify`` must consume the ``units`` attribute.

Before the fix, ``attach_units`` copied ``obj.attrs`` verbatim into the result,
so the ``units`` attribute it had just consumed survived onto the quantified
object. Two symptoms followed:

- silent: after a unit conversion the stale attribute contradicted the data,
  so anything trusting ``.attrs["units"]`` (plot labels, CF serialization)
  reported the pre-conversion unit;
- loud: calling ``quantify()`` twice raised an opaque ``TypeError`` because the
  stale attribute made the second call try to re-quantify a `Quantity`.
"""

import pytest
import unxts.interop.xarray  # noqa: F401  # registers the .unxt accessor
import xarray as xr
from unxts.interop.xarray._src.conversion import UNIT_ATTR

import unxt as u


def test_dataarray_quantify_consumes_units_attr():
    """`quantify` removes the attribute it consumed from the DataArray."""
    da = xr.DataArray([1000.0, 2000.0], dims=["x"], attrs={UNIT_ATTR: "m"})

    q = da.unxt.quantify()

    assert u.unit_of(q.data) == u.unit("m")
    assert UNIT_ATTR not in q.attrs
    # The source object must not be mutated.
    assert da.attrs[UNIT_ATTR] == "m"


def test_dataarray_attrs_do_not_contradict_data_after_uconvert():
    """After a conversion no attribute claims the pre-conversion unit."""
    da = xr.DataArray([1000.0, 2000.0], dims=["x"], attrs={UNIT_ATTR: "m"})

    q = da.unxt.quantify()
    converted = q.copy(data=u.uconvert("km", q.data))

    assert u.unit_of(converted.data) == u.unit("km")
    assert converted.attrs.get(UNIT_ATTR) != "m"


def test_dataarray_quantify_consumes_coord_units_attr():
    """Coordinate attributes are consumed too, not just the data variable."""
    da = xr.DataArray(
        [1.0, 2.0],
        dims=["i"],
        coords={"i": [0, 1], "x": ("i", [0.0, 1.0], {UNIT_ATTR: "s"})},
        attrs={UNIT_ATTR: "m"},
    )

    q = da.unxt.quantify()

    assert u.unit_of(q.coords["x"].data) == u.unit("s")
    assert UNIT_ATTR not in q.coords["x"].attrs


def test_dataarray_quantify_twice_is_a_no_op():
    """A second `quantify` is a no-op rather than an opaque TypeError."""
    da = xr.DataArray([1000.0, 2000.0], dims=["x"], attrs={UNIT_ATTR: "m"})

    once = da.unxt.quantify()
    twice = once.unxt.quantify()

    assert u.unit_of(twice.data) == u.unit("m")
    assert UNIT_ATTR not in twice.attrs


def test_dataarray_quantify_already_quantified_raises_clear_error():
    """Explicitly re-quantifying quantified data names the real cause."""
    da = xr.DataArray([1000.0, 2000.0], dims=["x"], attrs={UNIT_ATTR: "m"})
    once = da.unxt.quantify()

    with pytest.raises(ValueError, match="already a Quantity"):
        once.unxt.quantify("km")


def test_dataset_quantify_consumes_units_attrs():
    """Dataset data variables and coordinates both drop the consumed attr."""
    ds = xr.Dataset(
        {"a": ("i", [1000.0, 2000.0], {UNIT_ATTR: "m"})},
        coords={"x": ("i", [0.0, 1.0], {UNIT_ATTR: "s"})},
    )

    q = ds.unxt.quantify()

    assert u.unit_of(q["a"].data) == u.unit("m")
    assert UNIT_ATTR not in q["a"].attrs
    assert u.unit_of(q.coords["x"].data) == u.unit("s")
    assert UNIT_ATTR not in q.coords["x"].attrs


def test_dataset_quantify_twice_is_a_no_op():
    ds = xr.Dataset({"a": ("i", [1000.0, 2000.0], {UNIT_ATTR: "m"})})

    twice = ds.unxt.quantify().unxt.quantify()

    assert u.unit_of(twice["a"].data) == u.unit("m")
    assert UNIT_ATTR not in twice["a"].attrs


def test_dataarray_round_trip_through_conversion_is_correct():
    """Quantify -> uconvert -> dequantify still reports the live unit."""
    da = xr.DataArray([1000.0, 2000.0], dims=["x"], attrs={UNIT_ATTR: "m"})

    q = da.unxt.quantify()
    converted = q.copy(data=u.uconvert("km", q.data))
    back = converted.unxt.dequantify()

    assert back.attrs[UNIT_ATTR] == "km"


def test_dataarray_quantify_preserves_unrelated_attrs():
    """Only the unit attribute is consumed; everything else survives."""
    da = xr.DataArray(
        [1.0, 2.0], dims=["x"], attrs={UNIT_ATTR: "m", "long_name": "distance"}
    )

    q = da.unxt.quantify()

    assert q.attrs["long_name"] == "distance"
    assert UNIT_ATTR not in q.attrs


def test_dimension_coord_keeps_units_attr():
    """A dimension coordinate keeps its attr: xarray drops the Quantity there.

    xarray coerces dimension coordinates to plain index arrays, discarding the
    `Quantity`. The attribute is then the only surviving record of the unit, so
    consuming it would lose the unit outright.
    """
    da = xr.DataArray(
        [1.0, 2.0],
        dims=["x"],
        coords={"x": ("x", [0.0, 1.0], {UNIT_ATTR: "s"})},
        attrs={UNIT_ATTR: "m"},
    )

    q = da.unxt.quantify()

    # Precondition: xarray really did drop the Quantity on the dim coordinate.
    assert u.unit_of(q.coords["x"].data) is None
    # So the attribute must be preserved.
    assert q.coords["x"].attrs[UNIT_ATTR] == "s"
    # The data variable, which did keep its Quantity, still drops the attr.
    assert UNIT_ATTR not in q.attrs


def test_dataarray_attrs_untouched_when_no_unit_attached():
    """An unrelated ``units`` attr on a variable with no unit is left alone."""
    da = xr.DataArray([1.0, 2.0], dims=["x"])
    attached = xr.DataArray(da.data, dims=["x"], attrs={"long_name": "count"})

    q = attached.unxt.quantify()

    assert q.attrs == {"long_name": "count"}
