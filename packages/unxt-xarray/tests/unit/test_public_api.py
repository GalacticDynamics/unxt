"""Verify public API is accessible from the top-level package."""

import unxt_xarray


class TestPublicImports:
    """The four conversion functions must be importable from unxt_xarray."""

    def test_attach_units_importable(self):
        """Test that `attach_units` is importable."""
        assert hasattr(unxt_xarray, "attach_units")

    def test_extract_units_importable(self):
        """Test that `extract_units` is importable."""
        assert hasattr(unxt_xarray, "extract_units")

    def test_extract_unit_attributes_importable(self):
        """Test that `extract_unit_attributes` is importable."""
        assert hasattr(unxt_xarray, "extract_unit_attributes")

    def test_strip_units_importable(self):
        """Test that `strip_units` is importable."""
        assert hasattr(unxt_xarray, "strip_units")

    def test_all_lists_public_functions(self):
        """Test that __all__ lists all public functions."""
        for name in (
            "attach_units",
            "extract_units",
            "extract_unit_attributes",
            "strip_units",
        ):
            assert name in unxt_xarray.__all__, f"{name!r} missing from __all__"

    def test_no_private_names_in_all(self):
        """Test that __all__ does not contain private names."""
        for name in unxt_xarray.__all__:
            assert not name.startswith("_"), f"{name!r} is private but in __all__"
