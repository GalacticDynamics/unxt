"""Usage tests demonstrating how to use unxt-api to create custom implementations.

These tests show realistic usage patterns for third-party packages that want
to integrate with the unxt dispatch system.
"""

from typing import Any, ClassVar

import plum
import pytest

import unxt_api as api

# ==============================================================================
# Custom type integration tests
# ==============================================================================


def test_simple_quantity_type() -> None:
    """Test creating a simple quantity type with unit support."""

    # Define a custom quantity type
    class SimpleQuantity:
        def __init__(self, value: float, unit: str) -> None:
            self.value = value
            self.unit = unit

        def __repr__(self) -> str:
            return f"SimpleQuantity({self.value}, '{self.unit}')"

    # Register dispatch for unit_of
    @plum.dispatch
    def unit_of(obj: SimpleQuantity, /) -> str:
        return obj.unit

    # Create instance and test
    q = SimpleQuantity(5.0, "m")
    assert unit_of(q) == "m"

    q2 = SimpleQuantity(10.0, "kg")
    assert unit_of(q2) == "kg"


def test_dimension_aware_type() -> None:
    """Test creating a type that supports both unit and dimension."""

    class DimensionalQuantity:
        _dimension_map: ClassVar = {
            "m": "length",
            "km": "length",
            "s": "time",
            "kg": "mass",
        }

        def __init__(self, value: float, unit: str) -> None:
            self.value = value
            self.unit = unit

        @property
        def dimension(self) -> str:
            return self._dimension_map.get(self.unit, "unknown")

    # Register dispatches
    @plum.dispatch
    def unit_of(obj: DimensionalQuantity, /) -> str:
        return obj.unit

    @plum.dispatch
    def dimension_of(obj: DimensionalQuantity, /) -> str:
        return obj.dimension

    # Test
    q = DimensionalQuantity(100.0, "km")
    assert unit_of(q) == "km"
    assert dimension_of(q) == "length"


def test_unit_conversion_implementation() -> None:
    """Test implementing unit conversion for a custom type."""

    class ConvertibleQuantity:
        _conversion_factors: ClassVar = {
            ("km", "m"): 1000.0,
            ("m", "km"): 0.001,
            ("m", "m"): 1.0,
            ("km", "km"): 1.0,
        }

        def __init__(self, value: float, unit: str) -> None:
            self.value = value
            self.unit = unit

        def convert_to(self, target_unit: str) -> "ConvertibleQuantity":
            factor = self._conversion_factors.get((self.unit, target_unit))
            if factor is None:
                msg = f"Cannot convert from {self.unit} to {target_unit}"
                raise ValueError(msg)
            return ConvertibleQuantity(self.value * factor, target_unit)

    # Register dispatch
    @plum.dispatch
    def uconvert(to_unit: str, quantity: ConvertibleQuantity, /) -> ConvertibleQuantity:
        return quantity.convert_to(to_unit)

    @plum.dispatch
    def is_unit_convertible(
        to_unit: str, from_quantity: ConvertibleQuantity, /
    ) -> bool:
        key = (from_quantity.unit, to_unit)
        return key in ConvertibleQuantity._conversion_factors

    # Test conversion
    q = ConvertibleQuantity(5.0, "km")
    result = uconvert("m", q)
    assert result.value == 5000.0
    assert result.unit == "m"

    # Test convertibility check
    assert is_unit_convertible("m", q) is True
    assert is_unit_convertible("kg", q) is False


def test_strip_units_implementation() -> None:
    """Test implementing unit stripping for a custom type."""

    class StrippableQuantity:
        def __init__(self, value: float, unit: str) -> None:
            self.value = value
            self.unit = unit

    # Register dispatch for basic ustrip
    @plum.dispatch
    def ustrip(quantity: StrippableQuantity, /) -> float:
        return quantity.value

    # Test
    q = StrippableQuantity(42.5, "m")
    assert ustrip(q) == 42.5


def test_angle_wrapping_implementation() -> None:
    """Test implementing angle wrapping for a custom type."""

    class Angle:
        def __init__(self, value: float, unit: str = "deg") -> None:
            self.value = value
            self.unit = unit

        def wrap(self, min_val: float, max_val: float) -> "Angle":
            range_size = max_val - min_val
            wrapped = ((self.value - min_val) % range_size) + min_val
            return Angle(wrapped, self.unit)

    # Register dispatch
    @plum.dispatch
    def wrap_to(x: Angle, min: Angle, max: Angle, /) -> Angle:
        return x.wrap(min.value, max.value)

    # Test
    angle = Angle(370.0, "deg")
    result = wrap_to(angle, Angle(0.0), Angle(360.0))
    assert result.value == 10.0


# ==============================================================================
# Unit system integration tests
# ==============================================================================


def test_custom_unit_system() -> None:
    """Test creating a custom unit system type."""

    class CustomUnitSystem:
        def __init__(self, length: str, time: str, mass: str) -> None:
            self.length = length
            self.time = time
            self.mass = mass

        def get_unit(self, dimension: str) -> str:
            return getattr(self, dimension, None)

    class CustomQuantity:
        def __init__(self, value: float, unit: str) -> None:
            self.value = value
            self.unit = unit
            # Infer system (simplified)
            if unit in ("m", "s", "kg"):
                self._system = CustomUnitSystem("m", "s", "kg")
            elif unit in ("km", "s", "g"):
                self._system = CustomUnitSystem("km", "s", "g")
            else:
                self._system = None

    # Register dispatch
    @plum.dispatch
    def unitsystem_of(obj: CustomQuantity, /) -> CustomUnitSystem | None:
        return obj._system

    # Test
    q = CustomQuantity(100.0, "m")
    system = unitsystem_of(q)
    assert system is not None
    assert system.length == "m"
    assert system.time == "s"
    assert system.mass == "kg"


# ==============================================================================
# Complete package integration tests
# ==============================================================================


def test_minimal_physics_package() -> None:
    """Test a minimal physics package that integrates with unxt-api."""

    # Define the types
    class Unit:
        def __init__(self, symbol: str, dimension: str) -> None:
            self.symbol = symbol
            self.dimension = dimension

    class Quantity:
        def __init__(self, value: float, unit: Unit) -> None:
            self.value = value
            self.unit = unit

    # Register with unxt-api
    @plum.dispatch
    def unit(obj: Unit, /) -> Unit:
        """Identity for Unit."""
        return obj

    @plum.dispatch
    def unit_of(obj: Quantity, /) -> Unit:
        """Get unit from Quantity."""
        return obj.unit

    @plum.dispatch
    def dimension_of(obj: Unit, /) -> str:
        """Get dimension from Unit."""
        return obj.dimension

    @plum.dispatch
    def dimension_of(obj: Quantity, /) -> str:
        """Get dimension from Quantity."""
        return obj.unit.dimension

    @plum.dispatch
    def ustrip(obj: Quantity, /) -> float:
        """Strip units from Quantity."""
        return obj.value

    # Use the package
    meter = Unit("m", "length")
    distance = Quantity(100.0, meter)

    # Test all integrations
    assert unit(meter).symbol == "m"
    assert unit_of(distance).symbol == "m"
    assert dimension_of(meter) == "length"
    assert dimension_of(distance) == "length"
    assert ustrip(distance) == 100.0


# ==============================================================================
# Dispatch priority tests
# ==============================================================================


def test_more_specific_dispatch_wins() -> None:
    """Test that more specific dispatches take priority."""

    class BaseType:
        def __init__(self, value: Any) -> None:
            self.value = value

    class SpecificType(BaseType):
        pass

    # Register base dispatch
    @plum.dispatch
    def unit_of(obj: BaseType, /) -> str:
        return "base"

    # Register more specific dispatch
    @plum.dispatch
    def unit_of(obj: SpecificType, /) -> str:
        return "specific"

    # Test
    base_obj = BaseType(42)
    specific_obj = SpecificType(42)

    assert unit_of(base_obj) == "base"
    assert unit_of(specific_obj) == "specific"


def test_multiple_implementations_coexist() -> None:
    """Test that multiple implementations can coexist."""

    class TypeA:
        pass

    class TypeB:
        pass

    @plum.dispatch
    def dimension(obj: TypeA, /) -> str:
        return "dimension_a"

    @plum.dispatch
    def dimension(obj: TypeB, /) -> str:
        return "dimension_b"

    a = TypeA()
    b = TypeB()

    assert dimension(a) == "dimension_a"
    assert dimension(b) == "dimension_b"


# ==============================================================================
# Error handling tests
# ==============================================================================


def test_no_matching_dispatch() -> None:
    """Test that missing dispatches raise appropriate errors."""

    class UnknownType:
        pass

    obj = UnknownType()

    with pytest.raises(plum.resolver.NotFoundLookupError):
        api.dimension(obj)


def test_custom_error_in_implementation() -> None:
    """Test that custom errors in implementations are propagated."""

    class ErrorType:
        pass

    @plum.dispatch
    def unit_of(obj: ErrorType, /) -> str:
        msg = "Custom error in implementation"
        raise ValueError(msg)

    obj = ErrorType()

    with pytest.raises(ValueError, match="Custom error in implementation"):
        unit_of(obj)
