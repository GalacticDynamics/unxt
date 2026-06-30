"""Tests for unit propagation through xarray operations.

A ``unxt.Quantity`` reports ``quaxed.numpy`` as its Array API namespace, and
(as of the Array-API-conformant boolean ops) so do the masks/predicates xarray
builds internally. Operations on a DataArray therefore propagate units natively,
with no monkeypatching of xarray or its namespace machinery.
"""

import jax.numpy as jnp

# Importing unxt_xarray registers the .unxt accessor.
import unxt_xarray  # noqa: F401
import xarray as xr
from xarray.compat import array_api_compat

import unxt as u


def _quantity_da():
    """Build a 2-D DataArray whose data is a length Quantity."""
    q = u.Quantity(jnp.arange(6.0).reshape(2, 3), "m")
    return xr.DataArray(q, dims=["t", "x"], coords={"t": [0, 1], "x": [10, 20, 30]})


class TestReductionsPropagateUnits:
    """Reductions keep the Quantity (and its unit)."""

    def test_sum_preserves_unit(self):
        """``sum`` keeps the length unit."""
        result = _quantity_da().sum()
        assert isinstance(result.data, u.AbstractQuantity)
        assert u.unit_of(result.data) == u.unit("m")

    def test_prod_changes_unit_dimensionally(self):
        """``prod`` over the length-3 ``x`` axis gives m * m * m -> m**3."""
        result = _quantity_da().prod("x")
        assert u.unit_of(result.data) == u.unit("m3")

    def test_median_preserves_unit(self):
        """``median`` (nan-aware quantile) keeps the length unit."""
        result = _quantity_da().median()
        assert u.unit_of(result.data) == u.unit("m")

    def test_quantile_preserves_unit(self):
        """``quantile`` keeps the length unit."""
        result = _quantity_da().quantile(0.5)
        assert u.unit_of(result.data) == u.unit("m")


class TestMaskingPropagatesUnits:
    """where/fillna are unit-aware."""

    def test_where_preserves_unit(self):
        """``where`` keeps the length unit."""
        da = _quantity_da()
        result = da.where(da > u.Quantity(2.0, "m"))
        assert u.unit_of(result.data) == u.unit("m")

    def test_fillna_preserves_unit(self):
        """``fillna`` keeps the length unit."""
        da = _quantity_da()
        result = da.where(da > u.Quantity(2.0, "m")).fillna(u.Quantity(0.0, "m"))
        assert u.unit_of(result.data) == u.unit("m")


class TestProductsPropagateUnits:
    """Inner products combine units dimensionally."""

    def test_dot_squares_unit(self):
        """``dot`` of two length arrays gives an area."""
        da = _quantity_da()
        result = da.dot(da)
        assert u.unit_of(result.data) == u.unit("m2")


class TestNoNamespacePatching:
    """The integration must not monkeypatch xarray's namespace resolver."""

    def test_get_array_namespace_is_unpatched(self):
        """``unxt_xarray`` leaves xarray's get_array_namespace untouched."""
        assert array_api_compat.get_array_namespace.__module__.startswith("xarray")
