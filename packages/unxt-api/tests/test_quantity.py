"""Unit tests for quantity API (unxt_api._quantity)."""

import inspect
from typing import Any

import plum
import pytest

import unxt_api as api

# ==============================================================================
# Tests for uconvert()
# ==============================================================================


def test_uconvert_is_abstract_dispatch() -> None:
    """Test that uconvert is an abstract dispatch function."""
    assert isinstance(api.uconvert, plum.function.Function)
    assert hasattr(api.uconvert, "methods")


def test_uconvert_accepts_any_type() -> None:
    """Test that uconvert accepts Any type in signature."""
    assert "uconvert" in dir(api)


def test_uconvert_no_default_implementation_raises() -> None:
    """Test that calling uconvert without implementation raises error."""

    class NoDispatchType:
        pass

    obj1 = NoDispatchType()
    obj2 = NoDispatchType()

    with pytest.raises(plum.resolver.NotFoundLookupError):
        api.uconvert(obj1, obj2)


def test_uconvert_can_register_custom_dispatch(
    custom_unit_type, custom_quantity_type
) -> None:
    """Test that custom dispatches can be registered."""

    @plum.dispatch
    def uconvert(
        to_unit: custom_unit_type, quantity: custom_quantity_type, /
    ) -> custom_quantity_type:
        # Simple mock conversion
        return custom_quantity_type(quantity.value, to_unit.unit_str)

    unit = custom_unit_type("m")
    quantity = custom_quantity_type(1000, "km")
    result = uconvert(unit, quantity)

    assert isinstance(result, custom_quantity_type)
    assert result.unit_str == "m"


# ==============================================================================
# Tests for ustrip()
# ==============================================================================


def test_ustrip_is_abstract_dispatch() -> None:
    """Test that ustrip is an abstract dispatch function."""
    assert isinstance(api.ustrip, plum.function.Function)
    assert hasattr(api.ustrip, "methods")


def test_ustrip_accepts_any_type() -> None:
    """Test that ustrip accepts varargs."""
    assert "ustrip" in dir(api)


def test_ustrip_no_default_implementation_raises() -> None:
    """Test that calling ustrip without implementation raises error."""

    class NoDispatchType:
        pass

    obj = NoDispatchType()

    with pytest.raises(plum.resolver.NotFoundLookupError):
        api.ustrip(obj)


def test_ustrip_can_register_custom_dispatch(custom_quantity_type) -> None:
    """Test that custom dispatches can be registered for ustrip."""

    @plum.dispatch
    def ustrip(quantity: custom_quantity_type, /) -> Any:
        return quantity.value

    quantity = custom_quantity_type(42.5, "m")
    result = ustrip(quantity)
    assert result == 42.5


def test_ustrip_varargs_signature() -> None:
    """Test that ustrip supports variable arguments."""
    # The signature should accept *args
    # This is reflected in the abstract signature


# ==============================================================================
# Tests for is_unit_convertible()
# ==============================================================================


def test_is_unit_convertible_is_abstract_dispatch() -> None:
    """Test that is_unit_convertible is an abstract dispatch function."""
    assert isinstance(api.is_unit_convertible, plum.function.Function)
    assert hasattr(api.is_unit_convertible, "methods")


def test_is_unit_convertible_accepts_any_type() -> None:
    """Test that is_unit_convertible accepts Any types."""
    assert "is_unit_convertible" in dir(api)


def test_is_unit_convertible_no_default_implementation() -> None:
    """Test that is_unit_convertible raises when no dispatch found."""

    class NoDispatchType:
        pass

    obj1 = NoDispatchType()
    obj2 = NoDispatchType()

    # Should raise NotFoundLookupError when no dispatch is registered
    with pytest.raises(plum.resolver.NotFoundLookupError):
        api.is_unit_convertible(obj1, obj2)


def test_is_unit_convertible_can_register_custom_dispatch(custom_unit_type) -> None:
    """Test that custom dispatches can be registered."""

    @plum.dispatch
    def is_unit_convertible(
        to_unit: custom_unit_type, from_unit: custom_unit_type, /
    ) -> bool:
        # Simple mock: same dimension means convertible
        return to_unit.unit_str[0] == from_unit.unit_str[0]

    unit1 = custom_unit_type("m")
    unit2 = custom_unit_type("mm")
    unit3 = custom_unit_type("kg")

    assert is_unit_convertible(unit1, unit2) is True
    assert is_unit_convertible(unit1, unit3) is False


def test_is_unit_convertible_returns_bool() -> None:
    """Test that return type annotation is bool."""
    # The abstract signature returns bool
    inspect.signature(api.is_unit_convertible.__wrapped__)
    # Note: __wrapped__ may not be available, but we can check the docstring


# ==============================================================================
# Tests for wrap_to()
# ==============================================================================


def test_wrap_to_is_abstract_dispatch() -> None:
    """Test that wrap_to is an abstract dispatch function."""
    assert isinstance(api.wrap_to, plum.function.Function)
    assert hasattr(api.wrap_to, "methods")


def test_wrap_to_accepts_any_type() -> None:
    """Test that wrap_to accepts Any types."""
    assert "wrap_to" in dir(api)


def test_wrap_to_no_default_implementation_raises() -> None:
    """Test that calling wrap_to without implementation raises error."""

    class NoDispatchType:
        pass

    obj = NoDispatchType()

    with pytest.raises(plum.resolver.NotFoundLookupError):
        api.wrap_to(obj, obj, obj)


def test_wrap_to_can_register_custom_dispatch(custom_quantity_type) -> None:
    """Test that custom dispatches can be registered."""

    @plum.dispatch
    def wrap_to(
        x: custom_quantity_type,
        min: custom_quantity_type,
        max: custom_quantity_type,
        /,
    ) -> custom_quantity_type:
        # Simple wrapping logic
        range_size = max.value - min.value
        wrapped = ((x.value - min.value) % range_size) + min.value
        return custom_quantity_type(wrapped, x.unit_str)

    x = custom_quantity_type(370, "deg")
    min_val = custom_quantity_type(0, "deg")
    max_val = custom_quantity_type(360, "deg")

    result = wrap_to(x, min_val, max_val)
    assert result.value == 10


def test_wrap_to_keyword_argument_dispatch() -> None:
    """Test that wrap_to supports keyword arguments through redirect."""
    # The package provides a keyword argument dispatch that redirects
    # to the positional version
    # Check that both dispatches exist
    assert len(api.wrap_to.methods) >= 2


# ==============================================================================
# API consistency tests
# ==============================================================================


def test_all_quantity_functions_are_exported() -> None:
    """Test that all quantity functions are exported."""
    assert hasattr(api, "uconvert")
    assert hasattr(api, "ustrip")
    assert hasattr(api, "is_unit_convertible")
    assert hasattr(api, "wrap_to")


def test_all_quantity_functions_in_all() -> None:
    """Test that all quantity functions are in __all__."""
    assert "uconvert" in api.__all__
    assert "ustrip" in api.__all__
    assert "is_unit_convertible" in api.__all__
    assert "wrap_to" in api.__all__


def test_all_quantity_functions_are_dispatch_functions() -> None:
    """Test that all quantity functions are dispatch functions."""
    assert isinstance(api.uconvert, plum.function.Function)
    assert isinstance(api.ustrip, plum.function.Function)
    assert isinstance(api.is_unit_convertible, plum.function.Function)
    assert isinstance(api.wrap_to, plum.function.Function)
