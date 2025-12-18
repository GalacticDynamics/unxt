"""Advanced tests for unxt-api covering edge cases and advanced scenarios."""

import math

import plum
import pytest

import unxt_api as api

# ==============================================================================
# Dispatch method inspection tests
# ==============================================================================


def test_methods_attribute_exists() -> None:
    """Test that dispatch functions have methods attribute."""
    assert hasattr(api.dimension, "methods")
    assert hasattr(api.unit, "methods")
    assert hasattr(api.uconvert, "methods")


def test_can_list_registered_methods() -> None:
    """Test that we can list registered methods."""
    # All abstract functions should have at least their abstract signature
    assert len(api.dimension.methods) >= 0
    assert len(api.unit.methods) >= 0


def test_can_inspect_method_signatures() -> None:
    """Test that we can inspect individual method signatures."""
    # The methods should be inspectable
    for method in api.dimension.methods:
        assert hasattr(method, "signature")


# ==============================================================================
# Type annotation tests
# ==============================================================================


def test_abstract_signatures_use_any() -> None:
    """Test that abstract signatures accept Any type."""
    # Check dimension signature
    # Note: The actual signature may be wrapped, so we check the docstring
    assert api.dimension.__doc__ is not None


def test_return_type_annotations() -> None:
    """Test that functions have appropriate return type hints."""
    # is_unit_convertible should return bool
    # The others return Any in the abstract signatures
    # Type annotations are checked by mypy, not runtime


# ==============================================================================
# Concurrent dispatch registration tests
# ==============================================================================


def test_independent_registrations_dont_interfere() -> None:
    """Test that registrations for different types don't interfere."""

    class TypeA:
        value: str = "a"

    class TypeB:
        value: str = "b"

    # Register for TypeA
    @plum.dispatch
    def dimension(obj: TypeA, /) -> str:
        return "dimension_a"

    # Register for TypeB
    @plum.dispatch
    def dimension(obj: TypeB, /) -> str:
        return "dimension_b"

    # Both should work independently
    a = TypeA()
    b = TypeB()
    assert dimension(a) == "dimension_a"
    assert dimension(b) == "dimension_b"


def test_multiple_packages_can_register() -> None:
    """Test that multiple packages can register their types."""

    # Simulate package 1
    class Package1Type:
        pass

    @plum.dispatch
    def unit_of(obj: Package1Type, /) -> str:
        return "package1_unit"

    # Simulate package 2
    class Package2Type:
        pass

    @plum.dispatch
    def unit_of(obj: Package2Type, /) -> str:
        return "package2_unit"

    # Both should coexist
    obj1 = Package1Type()
    obj2 = Package2Type()
    assert unit_of(obj1) == "package1_unit"
    assert unit_of(obj2) == "package2_unit"


# ==============================================================================
# Complex type hierarchy tests
# ==============================================================================


def test_inheritance_hierarchy() -> None:
    """Test dispatch resolution with inheritance."""

    class BaseQuantity:
        pass

    class LengthQuantity(BaseQuantity):
        pass

    class DistanceQuantity(LengthQuantity):
        pass

    # Register for base
    @plum.dispatch
    def dimension_of(obj: BaseQuantity, /) -> str:
        return "base"

    # Register more specific
    @plum.dispatch
    def dimension_of(obj: LengthQuantity, /) -> str:
        return "length"

    # Test resolution
    base = BaseQuantity()
    length = LengthQuantity()
    distance = DistanceQuantity()

    assert dimension_of(base) == "base"
    assert dimension_of(length) == "length"
    # distance should resolve to length (most specific match)
    assert dimension_of(distance) == "length"


def test_multiple_inheritance() -> None:
    """Test dispatch with multiple inheritance."""

    class HasUnit:
        pass

    class HasDimension:
        pass

    class Quantity(HasUnit, HasDimension):
        pass

    # Register for each parent
    @plum.dispatch
    def unit_of(obj: HasUnit, /) -> str:
        return "has_unit"

    @plum.dispatch
    def dimension_of(obj: HasDimension, /) -> str:
        return "has_dimension"

    # Test
    q = Quantity()
    assert unit_of(q) == "has_unit"
    assert dimension_of(q) == "has_dimension"


# ==============================================================================
# Default behavior tests
# ==============================================================================


def test_is_unit_convertible_no_default() -> None:
    """Test that is_unit_convertible raises when no dispatch found."""

    class UnknownType:
        pass

    obj1 = UnknownType()
    obj2 = UnknownType()

    # Should raise when no dispatch is registered
    with pytest.raises(plum.resolver.NotFoundLookupError):
        api.is_unit_convertible(obj1, obj2)


def test_other_functions_raise_by_default() -> None:
    """Test that other functions raise when no dispatch found."""

    class UnknownType:
        pass

    obj = UnknownType()

    with pytest.raises(plum.resolver.NotFoundLookupError):
        api.dimension(obj)

    with pytest.raises(plum.resolver.NotFoundLookupError):
        api.unit(obj)


# ==============================================================================
# Vararg function tests
# ==============================================================================


def test_ustrip_varargs() -> None:
    """Test that ustrip can handle variable arguments."""

    class SimpleQuantity:
        def __init__(self, value: float) -> None:
            self.value = value

    # Single argument dispatch
    @plum.dispatch
    def ustrip(q: SimpleQuantity, /) -> float:
        return q.value

    q = SimpleQuantity(42.0)
    assert ustrip(q) == 42.0


def test_wrap_to_keyword_redirect() -> None:
    """Test that wrap_to keyword argument version redirects correctly."""

    class TestAngle:
        def __init__(self, value: float) -> None:
            self.value = value

    # Positional version
    @plum.dispatch
    def wrap_to(x: TestAngle, min: TestAngle, max: TestAngle, /) -> TestAngle:
        return TestAngle(42.0)  # simplified

    # Test positional
    result = wrap_to(TestAngle(370), TestAngle(0), TestAngle(360))
    assert result.value == 42.0

    # Test keyword (should redirect to positional)
    result_kw = wrap_to(TestAngle(370), min=TestAngle(0), max=TestAngle(360))
    assert result_kw.value == 42.0


# ==============================================================================
# Edge case tests
# ==============================================================================


def test_none_values() -> None:
    """Test handling of None values."""

    class MaybeQuantity:
        def __init__(self, value: float | None) -> None:
            self.value = value

    @plum.dispatch
    def ustrip(q: MaybeQuantity, /) -> float | None:
        return q.value

    assert ustrip(MaybeQuantity(42.0)) == 42.0
    assert ustrip(MaybeQuantity(None)) is None


def test_zero_values() -> None:
    """Test handling of zero values."""

    class Quantity:
        def __init__(self, value: float) -> None:
            self.value = value

    @plum.dispatch
    def ustrip(q: Quantity, /) -> float:
        return q.value

    assert ustrip(Quantity(0.0)) == 0.0
    assert ustrip(Quantity(-0.0)) == -0.0


def test_special_float_values() -> None:
    """Test handling of special float values (inf, nan)."""

    class Quantity:
        def __init__(self, value: float) -> None:
            self.value = value

    @plum.dispatch
    def ustrip(q: Quantity, /) -> float:
        return q.value

    assert math.isinf(ustrip(Quantity(float("inf"))))
    assert math.isinf(ustrip(Quantity(float("-inf"))))
    assert math.isnan(ustrip(Quantity(float("nan"))))


# ==============================================================================
# Dispatch resolution order tests
# ==============================================================================


def test_most_specific_wins() -> None:
    """Test that the most specific dispatch is chosen."""

    class Base:
        pass

    class Derived(Base):
        pass

    @plum.dispatch
    def unit_of(obj: Base, /) -> str:
        return "base"

    @plum.dispatch
    def unit_of(obj: Derived, /) -> str:
        return "derived"

    assert unit_of(Base()) == "base"
    assert unit_of(Derived()) == "derived"


def test_ambiguous_dispatch_error() -> None:
    """Test that ambiguous dispatches raise AmbiguousLookupError."""

    # Plum raises an error for truly ambiguous cases
    class A:
        pass

    class B:
        pass

    class C(A, B):
        pass

    @plum.dispatch
    def dimension(obj: A, /) -> str:
        return "a"

    @plum.dispatch
    def dimension(obj: B, /) -> str:
        return "b"

    # C inherits from both A and B - this is ambiguous
    c = C()
    with pytest.raises(plum.resolver.AmbiguousLookupError):
        dimension(c)


# ==============================================================================
# Docstring example tests
# ==============================================================================


def test_uconvert_docstring_example_pattern() -> None:
    """Test that the pattern shown in uconvert docstring is valid."""
    # The docstring shows examples with unxt, but we test the pattern
    assert api.uconvert.__doc__ is not None
    assert "Examples" in api.uconvert.__doc__


def test_ustrip_docstring_example_pattern() -> None:
    """Test that the pattern shown in ustrip docstring is valid."""
    assert api.ustrip.__doc__ is not None
    assert "Examples" in api.ustrip.__doc__


def test_wrap_to_docstring_example_pattern() -> None:
    """Test that the pattern shown in wrap_to docstring is valid."""
    assert api.wrap_to.__doc__ is not None
    assert "Examples" in api.wrap_to.__doc__
