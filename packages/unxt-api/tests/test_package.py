"""Tests for the unxt_api package-level API."""

import types

import unxt_api as uapi

# ==============================================================================
# Package structure tests
# ==============================================================================


def test_version_available() -> None:
    """Test that __version__ is available."""
    assert hasattr(uapi, "__version__")
    assert isinstance(uapi.__version__, str)


def test_all_exports() -> None:
    """Test that __all__ contains all expected exports."""
    expected = {
        "__version__",
        "dimension",
        "dimension_of",
        "unit",
        "unit_of",
        "uconvert_value",
        "uconvert",
        "ustrip",
        "is_unit_convertible",
        "wrap_to",
        "unitsystem_of",
    }
    assert set(uapi.__all__) == expected


def test_all_exports_importable() -> None:
    """Test that all items in __all__ are importable."""
    for name in uapi.__all__:
        assert hasattr(uapi, name), f"{name} not found in unxt_api"


def test_no_private_in_all() -> None:
    """Test that __all__ doesn't contain private names."""
    for name in uapi.__all__:
        assert not name.startswith("_") or name == "__version__"


# ==============================================================================
# Dimension functions tests
# ==============================================================================


def test_dimension_exists() -> None:
    """Test that dimension function exists."""
    assert hasattr(uapi, "dimension")
    assert callable(uapi.dimension)


def test_dimension_of_exists() -> None:
    """Test that dimension_of function exists."""
    assert hasattr(uapi, "dimension_of")
    assert callable(uapi.dimension_of)


# ==============================================================================
# Unit functions tests
# ==============================================================================


def test_unit_exists() -> None:
    """Test that unit function exists."""
    assert hasattr(uapi, "unit")
    assert callable(uapi.unit)


def test_unit_of_exists() -> None:
    """Test that unit_of function exists."""
    assert hasattr(uapi, "unit_of")
    assert callable(uapi.unit_of)


# ==============================================================================
# Quantity functions tests
# ==============================================================================


def test_uconvert_exists() -> None:
    """Test that uconvert function exists."""
    assert hasattr(uapi, "uconvert")
    assert callable(uapi.uconvert)


def test_ustrip_exists() -> None:
    """Test that ustrip function exists."""
    assert hasattr(uapi, "ustrip")
    assert callable(uapi.ustrip)


def test_is_unit_convertible_exists() -> None:
    """Test that is_unit_convertible function exists."""
    assert hasattr(uapi, "is_unit_convertible")
    assert callable(uapi.is_unit_convertible)


def test_wrap_to_exists() -> None:
    """Test that wrap_to function exists."""
    assert hasattr(uapi, "wrap_to")
    assert callable(uapi.wrap_to)


# ==============================================================================
# Unit system functions tests
# ==============================================================================


def test_unitsystem_of_exists() -> None:
    """Test that unitsystem_of function exists."""
    assert hasattr(uapi, "unitsystem_of")
    assert callable(uapi.unitsystem_of)


# ==============================================================================
# Package documentation tests
# ==============================================================================


def test_package_has_docstring() -> None:
    """Test that the package has a docstring."""
    assert uapi.__doc__ is not None
    assert len(uapi.__doc__) > 0


def test_functions_have_docstrings() -> None:
    """Test that all public functions have docstrings."""
    functions = [
        uapi.dimension,
        uapi.dimension_of,
        uapi.unit,
        uapi.unit_of,
        uapi.uconvert,
        uapi.ustrip,
        uapi.is_unit_convertible,
        uapi.wrap_to,
        uapi.unitsystem_of,
    ]

    for func in functions:
        assert func.__doc__ is not None, f"{func.__name__} missing docstring"
        assert len(func.__doc__) > 0, f"{func.__name__} has empty docstring"


# ==============================================================================
# Module attributes tests
# ==============================================================================


def test_module_name() -> None:
    """Test that module name is correct."""
    assert uapi.__name__ == "unxt_api"


def test_no_unintended_exports() -> None:
    """Test that we're not accidentally exporting implementation details."""
    # Check that we're not exporting things we shouldn't
    for name in dir(uapi):
        if not name.startswith("_") and name not in uapi.__all__:
            # Only modules should be present outside __all__
            obj = getattr(uapi, name)
            assert isinstance(obj, types.ModuleType), f"Unexpected export: {name}"
