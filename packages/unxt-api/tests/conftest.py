"""Pytest configuration for unxt-api tests."""

from typing import Any

import pytest

# ==============================================================================
# Fixtures for custom types


@pytest.fixture
def custom_dimension_type():
    """Create a custom type with dimension support."""

    class CustomDimType:
        def __init__(self, dim_str: str) -> None:
            self.dim_str = dim_str

    return CustomDimType


@pytest.fixture
def custom_unit_type():
    """Create a custom type with unit support."""

    class CustomUnitType:
        def __init__(self, unit_str: str) -> None:
            self.unit_str = unit_str

    return CustomUnitType


@pytest.fixture
def custom_quantity_type():
    """Create a custom type with quantity support."""

    class CustomQuantity:
        def __init__(self, value: Any, unit_str: str) -> None:
            self.value = value
            self.unit_str = unit_str

    return CustomQuantity


@pytest.fixture
def custom_unitsystem_type():
    """Create a custom type with unit system support."""

    class CustomUnitSystem:
        def __init__(self, length: str, time: str) -> None:
            self.length_unit = length
            self.time_unit = time

    return CustomUnitSystem
