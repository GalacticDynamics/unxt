"""Tests for Quantity printing with wadler-lindig."""

import wadler_lindig as wl

import unxt as u


class TestShortName:
    """Test the short_name feature for wadler-lindig printing."""

    def test_quantity_has_short_name(self):
        """Test that Quantity has a short_name class variable."""
        assert hasattr(u.Quantity, "short_name")
        assert u.Quantity.short_name == "Q"

    def test_barequantity_no_short_name(self):
        """Test that BareQuantity doesn't have a short_name or it's None."""
        # It should either not have the attribute or have it as None
        short_name = getattr(u.quantity.BareQuantity, "short_name", None)
        assert short_name is None

    def test_use_short_name_default_false(self):
        """Test that use_short_name defaults to False."""
        q = u.Quantity([1, 2, 3], "m")
        result = wl.pformat(q)
        assert result.startswith("Quantity")
        assert not result.startswith("Q(")

    def test_use_short_name_true(self):
        """Test that use_short_name=True uses the short name."""
        q = u.Quantity([1, 2, 3], "m")
        result = wl.pformat(q, use_short_name=True)
        assert result.startswith("Q(")
        assert "unit='m'" in result

    def test_use_short_name_with_include_params(self):
        """Test that use_short_name works with include_params."""
        q = u.Quantity([1, 2, 3], "m")
        result = wl.pformat(q, use_short_name=True, include_params=True)
        assert result.startswith("Q['length']")

    def test_use_short_name_with_named_unit_false(self):
        """Test that use_short_name works with named_unit=False."""
        q = u.Quantity([1, 2, 3], "m")
        result = wl.pformat(q, use_short_name=True, named_unit=False)
        assert result.startswith("Q(")
        # Should have unit as positional arg not named
        assert "'m')" in result or ", 'm')" in result

    def test_use_short_name_with_short_arrays(self):
        """Test that use_short_name works with short_arrays."""
        q = u.Quantity([1, 2, 3], "m")

        # Default short_arrays=True
        result = wl.pformat(q, use_short_name=True, short_arrays=True)
        assert result.startswith("Q(")
        assert "i32[3]" in result

        # short_arrays=False
        result = wl.pformat(q, use_short_name=True, short_arrays=False)
        assert result.startswith("Q(")
        assert "Array(" in result

    def test_use_short_name_with_short_arrays_compact(self):
        """Test that use_short_name works with short_arrays='compact'."""
        q = u.Quantity([1, 2, 3], "m")
        result = wl.pformat(q, use_short_name=True, short_arrays="compact")
        assert result.startswith("Q(")
        assert "[1, 2, 3]" in result

    def test_bare_quantity_use_short_name_none(self):
        """Test that BareQuantity with use_short_name=True still uses full name."""
        q = u.quantity.BareQuantity([1, 2, 3], "m")
        result = wl.pformat(q, use_short_name=True)
        # Should still use full name since short_name is None
        assert result.startswith("BareQuantity")

    def test_pprint(self):
        """Test that pprint works with use_short_name."""
        q = u.Quantity([1, 2, 3], "m")
        # This should not raise an error
        wl.pprint(q, use_short_name=True)

    def test_pdoc_method_directly(self):
        """Test calling __pdoc__ directly with use_short_name."""
        q = u.Quantity([1, 2, 3], "m")

        doc = q.__pdoc__(use_short_name=False)
        formatted = wl.pformat(doc)
        assert formatted.startswith("Quantity")

        doc = q.__pdoc__(use_short_name=True)
        formatted = wl.pformat(doc)
