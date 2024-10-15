"""Test the `unxt.unitsystems` module."""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import astropy.units as u
import numpy as np
import pytest

from unxt import unitsystems
from unxt.unitsystems import (
    AbstractUnitSystem,
    DimensionlessUnitSystem,
    dimensionless,
    equivalent,
    unitsystem,
)


@pytest.fixture
def clean_unitsystems_registry(monkeypatch):
    clean_registry = {}
    monkeypatch.setattr(
        "unxt._src.units.system.base._UNITSYSTEMS_REGISTRY", clean_registry
    )
    return clean_registry


# ===================================================================


def test_unitsystem_from_() -> None:
    """Test the :class:`~unxt.UnitSystem` from_."""
    usys = unitsystem(5 * u.kpc, 50 * u.Myr, 1e5 * u.Msun, u.rad)
    assert np.isclose((8 * u.Myr).decompose(usys).value, 8 / 50)


def test_compare() -> None:
    """Test the :meth:`~unxt.UnitSystem.compare` method."""
    usys1 = unitsystem(u.kpc, u.Myr, u.radian, u.Msun, u.mas / u.yr)
    usys1_clone = unitsystem(u.kpc, u.Myr, u.radian, u.Msun, u.mas / u.yr)

    usys2 = unitsystem(u.kpc, u.Myr, u.radian, u.Msun, u.kiloarcsecond / u.yr)
    usys3 = unitsystem(u.kpc, u.Myr, u.radian, u.kg, u.mas / u.yr)

    assert usys1 == usys1_clone
    assert usys1_clone == usys1

    assert usys1 != usys2
    assert usys2 != usys1

    assert usys1 != usys3
    assert usys3 != usys1


def test_regression_dimension_aliases_spaces() -> None:
    usys = unitsystem(u.kpc, u.Myr, u.radian, u.Msun, u.mas / u.yr)
    assert usys["angular speed"] == usys["angular velocity"]


def test_pickle(tmpdir: Path) -> None:
    """Test pickling and unpickling a :class:`~unxt.UnitSystem`."""
    usys = unitsystem(u.kpc, u.Myr, u.radian, u.Msun)

    path = tmpdir / "test.pkl"
    with path.open(mode="wb") as f:
        pickle.dump(usys, f)

    with path.open(mode="rb") as f:
        usys2 = pickle.load(f)  # noqa: S301

    assert usys == usys2


def test_non_frozen(clean_unitsystems_registry):
    # Passes
    class NoFrozen1(AbstractUnitSystem):
        pass

    clean_unitsystems_registry.clear()

    # Fails
    with pytest.raises(TypeError, match="cannot inherit non-frozen"):

        @dataclass(slots=False)
        class NoFrozen2(AbstractUnitSystem):
            pass

    clean_unitsystems_registry.clear()

    # Passes
    @dataclass(frozen=True, slots=True)
    class NoFrozen3(AbstractUnitSystem):
        pass


@pytest.mark.usefixtures("clean_unitsystems_registry")
def test_non_unit_fields():
    """Test that non-Unit fields are skipped."""

    @dataclass(frozen=True, slots=True)
    class SomeNoneUnitFields(AbstractUnitSystem):
        a: Annotated[u.Unit, u.get_physical_type("length")]
        b: int

    assert SomeNoneUnitFields._base_field_names == ("a",)


@pytest.mark.usefixtures("clean_unitsystems_registry")
def test_wrong_annotation():
    """Test that non-Unit fields are skipped."""
    # No dimension annotation
    with pytest.raises(
        TypeError, match="Field 'a' must be an Annotated with a dimension"
    ):

        @dataclass(frozen=True, slots=True)
        class BadAnnotations(AbstractUnitSystem):
            a: Annotated[u.Unit, "no dimension annotation"]

    # Too many dimension annotations
    match = "Field 'a' must be an Annotated with only one dimension"
    with pytest.raises(TypeError, match=match):

        @dataclass(frozen=True, slots=True)
        class BadAnnotations(AbstractUnitSystem):
            a: Annotated[
                u.Unit, u.get_physical_type("length"), u.get_physical_type("time")
            ]


def test_unitsystem_already_registered():
    """Test that a unit system can only be registered once."""

    class MyUnitSystem(AbstractUnitSystem):
        length: Annotated[u.Unit, u.get_physical_type("length")]
        time: Annotated[u.Unit, u.get_physical_type("time")]

    assert MyUnitSystem._base_dimensions in unitsystems.UNITSYSTEMS_REGISTRY

    with pytest.raises(ValueError, match="already exists"):

        class MyUnitSystem(AbstractUnitSystem):
            length: Annotated[u.Unit, u.get_physical_type("length")]
            time: Annotated[u.Unit, u.get_physical_type("time")]


class TestDimensionlessUnitSystem:
    """Test `unxt.unitsystems.DimensionlessUnitSystem`."""

    def test_getitem(self) -> None:
        """Test :meth:`unxt.unitsystems.DimensionlessUnitSystem.__getitem__`."""
        assert dimensionless["dimensionless"] == u.one

        with pytest.raises(u.UnitConversionError):
            dimensionless["length"]

    def test_decompose(self) -> None:
        """Test that dimensionless unitsystem can be decomposed."""
        with pytest.raises(ValueError, match="can not be decomposed into"):
            (15 * u.kpc).decompose(dimensionless)

    def test_str(self) -> None:
        """Test that the string representation is correct."""
        assert str(dimensionless) == "DimensionlessUnitSystem()"


def test_dimensionless_singleton():
    """Test that the dimensionless unit system is a singleton."""
    assert DimensionlessUnitSystem() is dimensionless


def test_equivalent():
    """Test that equivalent unit systems are equal."""
    usys1 = unitsystem(u.kpc, u.Myr, u.radian, u.Msun, u.mas / u.yr)
    usys2 = unitsystem(u.km, u.yr, u.deg, u.kg, u.deg / u.s)
    assert equivalent(usys1, usys2)

    usys3 = unitsystem(u.kpc, u.Myr, u.radian)
    assert not equivalent(usys1, usys3)
