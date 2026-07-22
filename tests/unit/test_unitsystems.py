"""Test the `unxt.unitsystems` module."""

import itertools
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import astropy.units as apyu
import numpy as np
import pytest
from astropy import constants as const
from astropy.constants import G as const_G  # noqa: N811

from unxt import dimension, unit, unitsystems
from unxt._src.unitsystems.base import _UNITSYSTEMS_REGISTRY
from unxt.unitsystems import (
    AbstractUnitSystem,
    AbstractUSysFlag,
    DimensionlessUnitSystem,
    DynamicalSimUSysFlag,
    StandardUSysFlag,
    dimensionless,
    equivalent,
    unitsystem,
)


@pytest.fixture
def clean_unitsystems_registry(monkeypatch):
    clean_registry = {}
    monkeypatch.setattr(
        "unxt._src.unitsystems.base._UNITSYSTEMS_REGISTRY", clean_registry
    )
    return clean_registry


# ===================================================================


def test_unitsystem_from_() -> None:
    """Test the `unxt.AbstractUnitSystem.from_`."""
    usys = unitsystem(5 * apyu.kpc, 50 * apyu.Myr, 1e5 * apyu.Msun, "rad")
    assert np.isclose((8 * apyu.Myr).decompose(usys).value, 8 / 50)


def test_compare() -> None:
    """Test the `unxt.AbstractUnitSystem.compare` method."""
    usys1 = unitsystem("kpc", "Myr", "radian", "Msun", "mas / yr")
    usys1_clone = unitsystem("kpc", "Myr", "radian", "Msun", "mas / yr")

    usys2 = unitsystem("kpc", "Myr", "radian", "Msun", "kiloarcsecond / yr")
    usys3 = unitsystem("kpc", "Myr", "radian", "kg", "mas / yr")

    assert usys1 == usys1_clone
    assert usys1_clone == usys1

    assert usys1 != usys2
    assert usys2 != usys1

    assert usys1 != usys3
    assert usys3 != usys1


def test_regression_dimension_aliases_spaces() -> None:
    usys = unitsystem("kpc", "Myr", "radian", "Msun", "mas / yr")
    assert usys["angular speed"] == usys["angular velocity"]


def test_pickle(tmpdir: Path) -> None:
    """Test pickling and unpickling a `unxt.AbstractUnitSystem`."""
    usys = unitsystem("kpc", "Myr", "radian", "Msun")

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
        a: Annotated[apyu.Unit, dimension("length")]
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
            a: Annotated[apyu.Unit, "no dimension annotation"]

    # Too many dimension annotations
    match = "Field 'a' must be an Annotated with only one dimension"
    with pytest.raises(TypeError, match=match):

        @dataclass(frozen=True, slots=True)
        class BadAnnotations(AbstractUnitSystem):
            a: Annotated[apyu.Unit, dimension("length"), dimension("time")]


def test_unitsystem_already_registered():
    """Test that a unit system can only be registered once."""

    class MyUnitSystem(AbstractUnitSystem):
        absement: Annotated[apyu.Unit, dimension("absement")]
        time: Annotated[apyu.Unit, dimension("time")]

    assert MyUnitSystem._base_dimensions in unitsystems.UNITSYSTEMS_REGISTRY

    with pytest.raises(ValueError, match="already exists"):

        class MyUnitSystem(AbstractUnitSystem):
            absement: Annotated[apyu.Unit, dimension("absement")]
            time: Annotated[apyu.Unit, dimension("time")]

    # Clean up custom unit system from registry:
    del _UNITSYSTEMS_REGISTRY[MyUnitSystem._base_dimensions]


class TestDimensionlessUnitSystem:
    """Test `unxt.unitsystems.DimensionlessUnitSystem`."""

    def test_getitem(self) -> None:
        """Test :meth:`unxt.unitsystems.DimensionlessUnitSystem.__getitem__`."""
        assert dimensionless["dimensionless"] == apyu.one

        with pytest.raises(apyu.UnitConversionError):
            _ = dimensionless["length"]

    def test_decompose(self) -> None:
        """Test that dimensionless unitsystem can be decomposed."""
        with pytest.raises(ValueError, match="can not be decomposed into"):
            (15 * apyu.kpc).decompose(dimensionless)

    def test_str(self) -> None:
        """Test that the string representation is correct."""
        assert str(dimensionless) == "DimensionlessUnitSystem()"


def test_dimensionless_singleton():
    """Test that the dimensionless unit system is a singleton."""
    assert DimensionlessUnitSystem() is dimensionless


def test_equivalent():
    """Test that equivalent unit systems are equal."""
    usys1 = unitsystem("kpc", "Myr", "radian", "Msun", "mas / yr")
    usys2 = unitsystem("km", "yr", "deg", "kg", "deg / s")
    assert equivalent(usys1, usys2)

    usys3 = unitsystem("kpc", "Myr", "radian")
    assert not equivalent(usys1, usys3)


def test_extend():
    """Test adding additional units to a unit system."""
    usys1 = unitsystem("kpc", "Myr", "radian", "Msun", "km / s")
    usys2 = unitsystem(usys1, "mas / yr")
    assert usys2["angular speed"] == unit("mas / yr")

    usys3 = unitsystem(usys1, "mas / yr", "pc")
    assert usys3["angular speed"] == unit("mas / yr")
    assert usys3["length"] == unit("pc")  # overridden


def test_abstract_usys_flag():
    """Test that the abstract unit system flag fails."""
    with pytest.raises(TypeError, match="Do not use"):
        unitsystem(AbstractUSysFlag, "kpc")

    with pytest.raises(ValueError, match="unit system flag classes"):
        AbstractUSysFlag()


def test_standard_flag():
    """Test defining unit system with the standard flag."""
    usys1 = unitsystem(StandardUSysFlag, "kpc", "Myr")
    usys2 = unitsystem("kpc", "Myr")
    assert usys1 == usys2

    with pytest.raises(ValueError, match="unit system flag classes"):
        StandardUSysFlag()


def test_simulation_usys():
    """Test defining the simulation unit system with expected inputs."""
    tmp_G = const_G.decompose([apyu.kpc, apyu.Myr, apyu.Msun])
    usys1 = unitsystem(DynamicalSimUSysFlag, "kpc", "Myr", "rad")
    assert np.isclose((1 * usys1["mass"]).to_value("Msun"), 1 / tmp_G.value)

    usys2 = unitsystem(DynamicalSimUSysFlag, "kpc", "Msun", "rad")
    assert np.isclose((1 * usys2["time"]).to_value("Myr"), 1 / np.sqrt(tmp_G.value))

    base_units = ("kpc", "Myr", "Msun", "km / s")
    for u1, u2 in itertools.product(base_units, base_units):
        if u1 == u2:
            continue

        usys = unitsystem(DynamicalSimUSysFlag, u1, u2)

        # Verify each base dimension returns a valid unit.
        for dim in ("length", "mass", "time"):
            result = usys[dim]
            assert isinstance(result, apyu.UnitBase), (
                f"usys[{dim!r}] for ({u1!r}, {u2!r}) returned {result!r}, "
                f"expected an astropy UnitBase"
            )


# ===================================================================
# Natural unit systems (issues #228-#232)


@pytest.mark.parametrize(
    ("flag", "constants"),
    [
        (unitsystems.HEPUSysFlag, ("c", "hbar")),
        (unitsystems.GeometrizedUSysFlag, ("c", "G")),
        (unitsystems.PlanckUSysFlag, ("c", "hbar", "G", "k_B")),
    ],
)
def test_natural_unitsystem_constants_are_unity(flag, constants):
    """The defining constants of each natural system are numerically 1."""
    usys = unitsystem(flag)
    for name in constants:
        value = getattr(const, name).decompose(usys).value
        assert np.isclose(value, 1.0), (flag.__name__, name, value)


def test_atomic_unitsystem_is_natural():
    """Atomic units set m_e = hbar = e = 1 (charge can't use `decompose`)."""
    usys = unitsystems.atomic
    # base units carry the defining constants
    assert np.isclose((1 * usys["length"]).to_value("m"), const.a0.value)
    assert np.isclose((1 * usys["mass"]).to_value("kg"), const.m_e.value)
    assert np.isclose((1 * usys["electrical charge"]).to_value("C"), const.e.si.value)
    # hbar is unity in the (length, mass, time) sub-basis
    hbar = const.hbar.decompose([usys["length"], usys["mass"], usys["time"]]).value
    assert np.isclose(hbar, 1.0)


@pytest.mark.parametrize(
    ("name", "flag", "dims"),
    [
        ("hep", unitsystems.HEPUSysFlag, ["length", "mass", "time"]),
        ("geometrized", unitsystems.GeometrizedUSysFlag, ["length", "mass", "time"]),
        (
            "planck",
            unitsystems.PlanckUSysFlag,
            ["length", "mass", "time", "temperature"],
        ),
        (
            "atomic",
            unitsystems.AtomicUSysFlag,
            ["length", "mass", "time", "electrical charge"],
        ),
    ],
)
def test_natural_unitsystem_name_flag_and_dims(name, flag, dims):
    """Name lookup, flag construction, and the realization all agree."""
    from_name = unitsystem(name)
    from_flag = unitsystem(flag)
    realization = getattr(unitsystems, name)

    assert from_name == from_flag
    assert from_name == realization
    assert [str(d) for d in from_name.base_dimensions] == dims


def test_hep_energy_scale_override():
    """The HEP energy scale is configurable (default 1 GeV)."""
    gev = unitsystem(unitsystems.HEPUSysFlag)
    tev = unitsystem(unitsystems.HEPUSysFlag, energy="TeV")
    # larger energy -> smaller length/time, larger mass (hbar = c = 1)
    assert tev["time"] == gev["time"] / 1000
    assert tev["mass"] == gev["mass"] * 1000


def test_geometrized_length_scale_override():
    """The geometrized length scale is configurable (default 1 m)."""
    usys = unitsystem(unitsystems.GeometrizedUSysFlag, length="km")
    assert usys["length"] == unit("km")


@pytest.mark.parametrize(
    "flag",
    [
        unitsystems.HEPUSysFlag,
        unitsystems.GeometrizedUSysFlag,
        unitsystems.PlanckUSysFlag,
        unitsystems.AtomicUSysFlag,
    ],
)
def test_natural_unitsystem_rejects_extra_positional_args(flag):
    """Stray positional arguments fail fast rather than being silently ignored."""
    with pytest.raises(TypeError, match="does not take positional arguments"):
        unitsystem(flag, "km")


def test_natural_unitsystem_pickle(tmpdir: Path) -> None:
    """Natural unit systems round-trip through pickle."""
    for usys in (
        unitsystems.hep,
        unitsystems.geometrized,
        unitsystems.planck,
        unitsystems.atomic,
    ):
        path = tmpdir / "usys.pkl"
        with path.open(mode="wb") as f:
            pickle.dump(usys, f)
        with path.open(mode="rb") as f:
            usys2 = pickle.load(f)  # noqa: S301
        assert usys == usys2
