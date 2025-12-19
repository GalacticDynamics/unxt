"""Tests for the unitsystems strategy."""
# pylint: disable=unreachable

from hypothesis import given, settings

import unxt as u
import unxt_hypothesis as ust


@given(usys=ust.unitsystems("m", "s", "kg", "rad"))
@settings(max_examples=30)
def test_mks_system(usys: u.AbstractUnitSystem) -> None:
    """Test MKS unit system generation."""
    assert isinstance(usys, u.AbstractUnitSystem)
    assert len(usys) == 4
    assert usys["length"] == u.unit("m")
    assert usys["time"] == u.unit("s")
    assert usys["mass"] == u.unit("kg")
    assert usys["angle"] == u.unit("rad")


@given(usys=ust.unitsystems("kpc", "Myr", "Msun", "rad"))
@settings(max_examples=30)
def test_galactic_system(usys: u.AbstractUnitSystem) -> None:
    """Test galactic unit system generation."""
    assert isinstance(usys, u.AbstractUnitSystem)
    assert len(usys) == 4
    assert u.dimension_of(usys["length"]) == u.dimension("length")
    assert u.dimension_of(usys["time"]) == u.dimension("time")
    assert u.dimension_of(usys["mass"]) == u.dimension("mass")


@given(usys=ust.unitsystems(ust.units("length"), "s", "kg", "rad"))
@settings(max_examples=50)
def test_varying_length_unit(usys: u.AbstractUnitSystem) -> None:
    """Test unit system with varying length units."""
    assert isinstance(usys, u.AbstractUnitSystem)
    assert len(usys) == 4
    # Length varies, but should always have length dimension
    assert u.dimension_of(usys["length"]) == u.dimension("length")
    # Other units are fixed
    assert usys["time"] == u.unit("s")
    assert usys["mass"] == u.unit("kg")
    assert usys["angle"] == u.unit("rad")


@given(usys=ust.unitsystems(ust.units("length"), ust.units("time"), "kg", "rad"))
@settings(max_examples=50)
def test_varying_length_and_time(usys: u.AbstractUnitSystem) -> None:
    """Test unit system with varying length and time units."""
    assert isinstance(usys, u.AbstractUnitSystem)
    assert len(usys) == 4
    # Length and time vary
    assert u.dimension_of(usys["length"]) == u.dimension("length")
    assert u.dimension_of(usys["time"]) == u.dimension("time")
    # Mass and angle are fixed
    assert usys["mass"] == u.unit("kg")
    assert usys["angle"] == u.unit("rad")


@given(
    usys=ust.unitsystems(
        ust.units("length"), ust.units("time"), ust.units("mass"), "rad"
    )
)
@settings(max_examples=50)
def test_three_varying_units(usys: u.AbstractUnitSystem) -> None:
    """Test unit system with three varying units."""
    assert isinstance(usys, u.AbstractUnitSystem)
    assert len(usys) == 4
    assert u.dimension_of(usys["length"]) == u.dimension("length")
    assert u.dimension_of(usys["time"]) == u.dimension("time")
    assert u.dimension_of(usys["mass"]) == u.dimension("mass")
    assert usys["angle"] == u.unit("rad")


@given(
    usys=ust.unitsystems(
        ust.units("length", max_complexity=3),
        ust.units("time", max_complexity=3),
        "Msun",
        "rad",
    )
)
@settings(max_examples=30)
def test_complex_varying_units(usys: u.AbstractUnitSystem) -> None:
    """Test unit system with complex varying units."""
    assert isinstance(usys, u.AbstractUnitSystem)
    assert len(usys) == 4
    assert u.dimension_of(usys["length"]) == u.dimension("length")
    assert u.dimension_of(usys["time"]) == u.dimension("time")


@given(usys=ust.unitsystems("m", "s", "kg", "rad"))
@settings(max_examples=30)
def test_four_unit_system(usys: u.AbstractUnitSystem) -> None:
    """Test system with four units."""
    assert isinstance(usys, u.AbstractUnitSystem)
    assert len(usys) == 4
    assert u.dimension_of(usys["length"]) == u.dimension("length")
    assert u.dimension_of(usys["time"]) == u.dimension("time")


def test_accepts_fixed_string() -> None:
    """Test that the strategy accepts fixed string units."""
    strategy = ust.unitsystems("m", "s", "kg")
    assert strategy is not None


def test_accepts_unit_strategy() -> None:
    """Test that the strategy accepts unit strategies."""
    strategy = ust.unitsystems(ust.units("length"), ust.units("time"), "kg")
    assert strategy is not None


@given(usys=ust.unitsystems("AU", "yr", "Msun", "rad"))
@settings(max_examples=30)
def test_astronomical_units(usys: u.AbstractUnitSystem) -> None:
    """Test unit system with astronomical units."""
    assert isinstance(usys, u.AbstractUnitSystem)
    assert len(usys) == 4
    assert usys["length"] == u.unit("AU")
    assert usys["time"] == u.unit("yr")
    assert usys["mass"] == u.unit("Msun")
