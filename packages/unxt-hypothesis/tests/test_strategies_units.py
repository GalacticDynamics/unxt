"""Tests for the units strategy."""
# pylint: disable=unreachable

import astropy.units as apyu
import hypothesis.strategies as st
from hypothesis import given, settings

import unxt as u
import unxt_hypothesis as ust


@given(unit=ust.units("length"))
@settings(max_examples=50)
def test_length_dimension(unit: u.AbstractUnit) -> None:
    """Test that generated length units have the correct dimension."""
    assert isinstance(unit, u.AbstractUnit)
    dim = u.dimension_of(unit)
    assert dim == u.dimension("length"), f"Expected length, got {dim}"


@given(unit=ust.units("velocity"))
@settings(max_examples=50)
def test_velocity_dimension(unit: u.AbstractUnit) -> None:
    """Test that generated velocity units have the correct dimension."""
    assert isinstance(unit, u.AbstractUnit)
    dim = u.dimension_of(unit)
    assert dim == u.dimension("velocity"), f"Expected velocity, got {dim}"


@given(unit=ust.units("energy"))
@settings(max_examples=50)
def test_energy_dimension(unit: u.AbstractUnit) -> None:
    """Test that generated energy units have the correct dimension."""
    assert isinstance(unit, u.AbstractUnit)
    dim = u.dimension_of(unit)
    assert dim == u.dimension("energy"), f"Expected energy, got {dim}"


@given(unit=ust.units("mass"))
@settings(max_examples=50)
def test_mass_dimension(unit: u.AbstractUnit) -> None:
    """Test that generated mass units have the correct dimension."""
    assert isinstance(unit, u.AbstractUnit)
    dim = u.dimension_of(unit)
    assert dim == u.dimension("mass"), f"Expected mass, got {dim}"


@given(unit=ust.units("time"))
@settings(max_examples=50)
def test_time_dimension(unit: u.AbstractUnit) -> None:
    """Test that generated time units have the correct dimension."""
    assert isinstance(unit, u.AbstractUnit)
    dim = u.dimension_of(unit)
    assert dim == u.dimension("time"), f"Expected time, got {dim}"


@given(unit=ust.units("force"))
@settings(max_examples=50)
def test_force_dimension(unit: u.AbstractUnit) -> None:
    """Test that generated force units have the correct dimension."""
    assert isinstance(unit, u.AbstractUnit)
    dim = u.dimension_of(unit)
    assert dim == u.dimension("force"), f"Expected force, got {dim}"


@given(unit=ust.units("length", integer_powers=True, max_complexity=0))
@settings(max_examples=30)
def test_no_complexity(unit: u.AbstractUnit) -> None:
    """Test that max_complexity=0 only generates base and composed units."""
    assert isinstance(unit, u.AbstractUnit)
    dim = u.dimension_of(unit)
    assert dim == u.dimension("length")
    # Should only be base unit or composed forms, no compound cancellations


@given(unit=ust.units("velocity", max_complexity=5))
@settings(max_examples=30)
def test_high_complexity(unit: u.AbstractUnit) -> None:
    """Test that max_complexity allows compound units."""
    assert isinstance(unit, u.AbstractUnit)
    dim = u.dimension_of(unit)
    assert dim == u.dimension("velocity")


@given(unit=ust.units("energy", integer_powers=False))
@settings(max_examples=30)
def test_non_integer_powers(unit: u.AbstractUnit) -> None:
    """Test that integer_powers=False works."""
    assert isinstance(unit, u.AbstractUnit)
    dim = u.dimension_of(unit)
    assert dim == u.dimension("energy")


def test_accepts_string_dimension() -> None:
    """Test that the strategy accepts string dimensions."""
    strategy = ust.units("length")
    # Should not raise an error
    assert strategy is not None


def test_accepts_physical_type() -> None:
    """Test that the strategy accepts PhysicalType dimensions."""
    dim = u.dimension("length")
    strategy = ust.units(dim)
    # Should not raise an error
    assert strategy is not None


@given(unit=ust.units("acceleration"))
@settings(max_examples=30)
def test_compound_dimension(unit: u.AbstractUnit) -> None:
    """Test with compound dimension (acceleration = length/time^2)."""
    assert isinstance(unit, u.AbstractUnit)
    dim = u.dimension_of(unit)
    assert dim == u.dimension("acceleration")


@given(unit=ust.units("angle"))
@settings(max_examples=30)
def test_dimensionless_angle(unit: u.AbstractUnit) -> None:
    """Test with angular dimension."""
    assert isinstance(unit, u.AbstractUnit)
    dim = u.dimension_of(unit)
    assert dim == u.dimension("angle")


@given(unit=ust.units("pressure"))
@settings(max_examples=30)
def test_pressure_dimension(unit: u.AbstractUnit) -> None:
    """Test with pressure dimension."""
    assert isinstance(unit, u.AbstractUnit)
    dim = u.dimension_of(unit)
    assert dim == u.dimension("pressure")


@given(unit=ust.units(st.sampled_from(["length", "time", "mass"])))
@settings(max_examples=50)
def test_sampled_from_dimension(unit: u.AbstractUnit) -> None:
    """Test that dimension can be a st.sampled_from strategy."""
    assert isinstance(unit, u.AbstractUnit)
    dim = u.dimension_of(unit)
    expected_dims = [u.dimension("length"), u.dimension("time"), u.dimension("mass")]
    assert dim in expected_dims, f"Expected one of {expected_dims}, got {dim}"


# Tests for derived_units


@given(unit=ust.derived_units("m"))
@settings(max_examples=50)
def test_derived_units_from_string(unit: u.AbstractUnit) -> None:
    """Test that derived_units works with string base."""
    assert isinstance(unit, u.AbstractUnit)
    dim = u.dimension_of(unit)
    assert dim == u.dimension("length")


@given(unit=ust.derived_units(apyu.Unit("kg")))
@settings(max_examples=50)
def test_derived_units_from_unit(unit: u.AbstractUnit) -> None:
    """Test that derived_units works with Unit base."""
    assert isinstance(unit, u.AbstractUnit)
    dim = u.dimension_of(unit)
    assert dim == u.dimension("mass")


@given(unit=ust.derived_units(ust.units("velocity")))
@settings(max_examples=50)
def test_derived_units_from_strategy(unit: u.AbstractUnit) -> None:
    """Test that derived_units works with strategy base."""
    assert isinstance(unit, u.AbstractUnit)
    dim = u.dimension_of(unit)
    assert dim == u.dimension("velocity")


@given(unit=ust.derived_units("m", max_complexity=0))
@settings(max_examples=30)
def test_derived_units_no_complexity(unit: u.AbstractUnit) -> None:
    """Test that max_complexity=0 limits to base and composed forms."""
    assert isinstance(unit, u.AbstractUnit)
    dim = u.dimension_of(unit)
    assert dim == u.dimension("length")


@given(unit=ust.derived_units("s", max_complexity=5))
@settings(max_examples=30)
def test_derived_units_high_complexity(unit: u.AbstractUnit) -> None:
    """Test that max_complexity allows more compound units."""
    assert isinstance(unit, u.AbstractUnit)
    dim = u.dimension_of(unit)
    assert dim == u.dimension("time")


@given(unit=ust.derived_units("J", integer_powers=False))
@settings(max_examples=30)
def test_derived_units_non_integer_powers(unit: u.AbstractUnit) -> None:
    """Test that integer_powers=False allows fractional powers."""
    assert isinstance(unit, u.AbstractUnit)
    dim = u.dimension_of(unit)
    assert dim == u.dimension("energy")
