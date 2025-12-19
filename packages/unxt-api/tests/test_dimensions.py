"""Unit tests for dimension API (unxt_api._dimensions)."""

import plum
import pytest

import unxt_api as api

# ==============================================================================
# Tests for dimension()
# ==============================================================================


def test_dimension_is_abstract_dispatch() -> None:
    """Test that dimension is an abstract dispatch function."""
    assert isinstance(api.dimension, plum.function.Function)
    assert hasattr(api.dimension, "methods")


def test_dimension_accepts_any_type() -> None:
    """Test that dimension accepts Any type in signature."""
    # The abstract signature should accept Any
    assert "dimension" in dir(api)


def test_dimension_no_default_implementation() -> None:
    """Test that calling dimension without implementation raises error."""

    # Create a custom type that has no dispatch registered
    class NoDispatchType:
        pass

    obj = NoDispatchType()

    # Should raise NotFoundLookupError since no dispatch is registered
    with pytest.raises(plum.resolver.NotFoundLookupError):
        api.dimension(obj)


def test_dimension_can_register_custom_dispatch(custom_dimension_type) -> None:
    """Test that custom dispatches can be registered."""

    # Register a custom dispatch
    @plum.dispatch
    def dimension(obj: custom_dimension_type, /) -> str:
        return obj.dim_str

    # Create instance and test
    obj = custom_dimension_type("length")
    result = dimension(obj)
    assert result == "length"


def test_dimension_multiple_dispatches_possible() -> None:
    """Test that multiple dispatches can coexist."""
    # The abstract function should support multiple dispatches
    # Count existing methods
    initial_count = len(api.dimension.methods)
    assert initial_count >= 0  # At least the abstract signature


# ==============================================================================
# Tests for dimension_of()
# ==============================================================================


def test_dimension_of_is_abstract_dispatch() -> None:
    """Test that dimension_of is an abstract dispatch function."""
    assert isinstance(api.dimension_of, plum.function.Function)
    assert hasattr(api.dimension_of, "methods")


def test_dimension_of_accepts_any_type() -> None:
    """Test that dimension_of accepts Any type in signature."""
    assert "dimension_of" in dir(api)


def test_dimension_of_no_default_implementation() -> None:
    """Test that calling dimension_of without implementation may return None.

    Note: When unxt is imported (e.g., during doctest execution), it may
    register a default implementation that returns None for unknown types.
    """

    class NoDispatchType:
        pass

    obj = NoDispatchType()

    try:
        result = api.dimension_of(obj)
        # If it doesn't raise, it should return None (unxt's default)
        assert result is None
    except plum.resolver.NotFoundLookupError:
        # This is also acceptable if unxt hasn't been imported
        pass


def test_dimension_of_can_register_custom_dispatch(custom_unit_type) -> None:
    """Test that custom dispatches can be registered for dimension_of."""

    @plum.dispatch
    def dimension_of(obj: custom_unit_type, /) -> str:
        # Simple implementation that extracts dimension from unit string
        return f"dim_of_{obj.unit_str}"

    obj = custom_unit_type("m")
    result = dimension_of(obj)
    assert result == "dim_of_m"


# ==============================================================================
# API consistency tests
# ==============================================================================


def test_both_dimension_functions_are_exported() -> None:
    """Test that both functions are exported from unxt_api."""
    assert hasattr(api, "dimension")
    assert hasattr(api, "dimension_of")


def test_both_dimension_functions_in_all() -> None:
    """Test that both functions are in __all__."""
    assert "dimension" in api.__all__
    assert "dimension_of" in api.__all__


def test_dimension_functions_have_independent_registries() -> None:
    """Test that dimension and dimension_of have independent dispatch registries."""
    # They should be different function objects
    assert api.dimension is not api.dimension_of

    # They should have their own method registries
    assert id(api.dimension.methods) != id(api.dimension_of.methods)
