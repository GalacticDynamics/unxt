"""Tests for the unxt_api package-level API."""

import types

import unxt_api as api

# ==============================================================================
# Package structure tests
# ==============================================================================


def test_version_available() -> None:
    """Test that __version__ is available."""
    assert hasattr(api, "__version__")
    assert isinstance(api.__version__, str)


def test_all_exports() -> None:
    """Test that __all__ contains all expected exports."""
    expected = {
        "__version__",
        "dimension",
        "dimension_of",
        "unit",
        "unit_of",
        "uconvert",
        "ustrip",
        "is_unit_convertible",
        "wrap_to",
        "unitsystem_of",
    }
    assert set(api.__all__) == expected


def test_all_exports_importable() -> None:
    """Test that all items in __all__ are importable."""
    for name in api.__all__:
        assert hasattr(api, name), f"{name} not found in unxt_api"


def test_no_private_in_all() -> None:
    """Test that __all__ doesn't contain private names."""
    for name in api.__all__:
        assert not name.startswith("_") or name == "__version__"


# ==============================================================================
# Dimension functions tests
# ==============================================================================


def test_dimension_exists() -> None:
    """Test that dimension function exists."""
    assert hasattr(api, "dimension")
    assert callable(api.dimension)


def test_dimension_of_exists() -> None:
    """Test that dimension_of function exists."""
    assert hasattr(api, "dimension_of")
    assert callable(api.dimension_of)


# ==============================================================================
# Unit functions tests
# ==============================================================================


def test_unit_exists() -> None:
    """Test that unit function exists."""
    assert hasattr(api, "unit")
    assert callable(api.unit)


def test_unit_of_exists() -> None:
    """Test that unit_of function exists."""
    assert hasattr(api, "unit_of")
    assert callable(api.unit_of)


# ==============================================================================
# Quantity functions tests
# ==============================================================================


def test_uconvert_exists() -> None:
    """Test that uconvert function exists."""
    assert hasattr(api, "uconvert")
    assert callable(api.uconvert)


def test_ustrip_exists() -> None:
    """Test that ustrip function exists."""
    assert hasattr(api, "ustrip")
    assert callable(api.ustrip)


def test_is_unit_convertible_exists() -> None:
    """Test that is_unit_convertible function exists."""
    assert hasattr(api, "is_unit_convertible")
    assert callable(api.is_unit_convertible)


def test_wrap_to_exists() -> None:
    """Test that wrap_to function exists."""
    assert hasattr(api, "wrap_to")
    assert callable(api.wrap_to)


# ==============================================================================
# Unit system functions tests
# ==============================================================================


def test_unitsystem_of_exists() -> None:
    """Test that unitsystem_of function exists."""
    assert hasattr(api, "unitsystem_of")
    assert callable(api.unitsystem_of)


# ==============================================================================
# Package documentation tests
# ==============================================================================


def test_package_has_docstring() -> None:
    """Test that the package has a docstring."""
    assert api.__doc__ is not None
    assert len(api.__doc__) > 0


def test_functions_have_docstrings() -> None:
    """Test that all public functions have docstrings."""
    functions = [
        api.dimension,
        api.dimension_of,
        api.unit,
        api.unit_of,
        api.uconvert,
        api.ustrip,
        api.is_unit_convertible,
        api.wrap_to,
        api.unitsystem_of,
    ]

    for func in functions:
        assert func.__doc__ is not None, f"{func.__name__} missing docstring"
        assert len(func.__doc__) > 0, f"{func.__name__} has empty docstring"


# ==============================================================================
# Module attributes tests
# ==============================================================================


def test_module_name() -> None:
    """Test that module name is correct."""
    assert api.__name__ == "unxt_api"


def test_no_unintended_exports() -> None:
    """Test that we're not accidentally exporting implementation details."""
    # Check that we're not exporting things we shouldn't
    for name in dir(api):
        if not name.startswith("_") and name not in api.__all__:
            # Only modules should be present outside __all__
            obj = getattr(api, name)
            assert isinstance(obj, types.ModuleType), f"Unexpected export: {name}"
