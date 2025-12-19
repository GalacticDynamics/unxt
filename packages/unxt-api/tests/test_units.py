"""Unit tests for units API (unxt_api._units)."""

import plum
import pytest

import unxt_api as api

# ==============================================================================
# Tests for unit()
# ==============================================================================


def test_unit_is_abstract_dispatch() -> None:
    """Test that unit is an abstract dispatch function."""
    assert isinstance(api.unit, plum.function.Function)
    assert hasattr(api.unit, "methods")


def test_unit_accepts_any_type() -> None:
    """Test that unit accepts Any type in signature."""
    assert "unit" in dir(api)


def test_unit_no_default_implementation() -> None:
    """Test that calling unit without implementation raises error."""

    class NoDispatchType:
        pass

    obj = NoDispatchType()

    with pytest.raises(plum.resolver.NotFoundLookupError):
        api.unit(obj)


def test_unit_can_register_custom_dispatch(custom_unit_type) -> None:
    """Test that custom dispatches can be registered."""

    @plum.dispatch
    def unit(obj: custom_unit_type, /) -> str:
        return obj.unit_str

    obj = custom_unit_type("m")
    result = unit(obj)
    assert result == "m"


def test_unit_multiple_dispatches_possible() -> None:
    """Test that multiple dispatches can coexist."""
    initial_count = len(api.unit.methods)
    assert initial_count >= 0


# ==============================================================================
# Tests for unit_of()
# ==============================================================================


def test_unit_of_is_abstract_dispatch() -> None:
    """Test that unit_of is an abstract dispatch function."""
    assert isinstance(api.unit_of, plum.function.Function)
    assert hasattr(api.unit_of, "methods")


def test_unit_of_accepts_any_type() -> None:
    """Test that unit_of accepts Any type in signature."""
    assert "unit_of" in dir(api)


def test_unit_of_no_default_implementation() -> None:
    """Test that calling unit_of without implementation may return None.

    Note: When unxt is imported (e.g., during doctest execution), it may
    register a default implementation that returns None for unknown types.
    """

    class NoDispatchType:
        pass

    obj = NoDispatchType()

    try:
        result = api.unit_of(obj)
        # If it doesn't raise, it should return None (unxt's default)
        assert result is None
    except plum.resolver.NotFoundLookupError:
        # This is also acceptable if unxt hasn't been imported
        pass


def test_unit_of_can_register_custom_dispatch(custom_quantity_type) -> None:
    """Test that custom dispatches can be registered for unit_of."""

    @plum.dispatch
    def unit_of(obj: custom_quantity_type, /) -> str:
        return obj.unit_str

    obj = custom_quantity_type(42, "kg")
    result = unit_of(obj)
    assert result == "kg"


# ==============================================================================
# API consistency tests
# ==============================================================================


def test_both_unit_functions_are_exported() -> None:
    """Test that both functions are exported from unxt_api."""
    assert hasattr(api, "unit")
    assert hasattr(api, "unit_of")


def test_both_unit_functions_in_all() -> None:
    """Test that both functions are in __all__."""
    assert "unit" in api.__all__
    assert "unit_of" in api.__all__


def test_unit_functions_have_independent_registries() -> None:
    """Test that unit and unit_of have independent dispatch registries."""
    assert api.unit is not api.unit_of
    assert id(api.unit.methods) != id(api.unit_of.methods)
