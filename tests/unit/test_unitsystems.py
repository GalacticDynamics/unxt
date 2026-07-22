"""Test the `unxt.unitsystems` module."""

import itertools
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import astropy.units as apyu
import numpy as np
import pytest
from astropy.constants import G as const_G  # noqa: N811

import unxt as u
from unxt import dimension, unit, unitsystems
from unxt._src.unitsystems.base import _UNITSYSTEMS_REGISTRY
from unxt.unitsystems import (
    AbstractUnitSystem,
    AbstractUSysFlag,
    DimensionlessUnitSystem,
    DynamicalSimUSysFlag,
    StandardUSysFlag,
    cgs,
    dimensionless,
    equivalent,
    galactic,
    si,
    solarsystem,
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


def test_unitsystem_unknown_name_raises_clear_error() -> None:
    """A single string that is not a registered system name errors clearly.

    ``unitsystem("m")`` looks up a *named* system (like ``"galactic"``); "m" is a
    unit, not a system name, so it must raise a helpful ``ValueError`` -- not a
    bare ``KeyError: 'm'`` -- that names the registered systems and points to the
    unit path (``unitsystem(unit(...))``).
    """
    with pytest.raises(ValueError, match="not a registered unit system") as exc_info:
        unitsystem("m")

    msg = str(exc_info.value)
    # The message lists every registered system so the user can self-correct.
    for name in ("cgs", "dimensionless", "galactic", "si", "solarsystem"):
        assert name in msg
    # ... and points at the unit path for the likely intent.
    assert "unit" in msg


def test_unitsystem_known_name_still_resolves() -> None:
    """Registered names still resolve (regression guard for the error path)."""
    assert unitsystem("galactic") == galactic


def test_no_arg_unitsystem_is_dimensionless() -> None:
    """``unitsystem()`` with no arguments is the dimensionless system.

    Previously it built a base-unit-less ``UnitSystem()`` that was **not**
    ``dimensionless`` and, worse, answered every derived-dimension lookup with a
    silent SI default (``unitsystem()["length"] == Unit("m")``). It now returns
    the ``dimensionless`` singleton, matching ``unitsystem(None)`` and
    ``unitsystem([])``, which already did.
    """
    usys = unitsystem()

    assert usys is dimensionless
    assert type(usys) is DimensionlessUnitSystem
    assert usys == dimensionless
    # Whatever ``dimensionless`` reports for base_units, the no-arg call agrees.
    assert usys.base_units == dimensionless.base_units

    # It must agree with the other empty spellings.
    assert unitsystem() is unitsystem(None)
    assert unitsystem() is unitsystem([])

    # A derived-dimension lookup must NOT silently return an SI unit -- the empty
    # system defines no length, so this raises just as ``dimensionless`` does
    # (see ``TestDimensionlessUnitSystem.test_getitem``).
    with pytest.raises(apyu.UnitConversionError):
        _ = usys["length"]


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


def test_getitem_resolves_every_base_dimension() -> None:
    """``usys[dim]`` must resolve for every base dimension of every realization.

    Regression: ``__getitem__`` recomputed the field name from the dimension via
    astropy's first physical-type alias, which does not match the field names
    declared on ``SIUnitSystem`` / ``CGSUnitSystem`` (e.g. dimension ``current``
    -> astropy alias ``electrical_current`` vs. field ``electric_current``), so
    looking up those dimensions raised ``AttributeError``.
    """
    for usys in (si, cgs, galactic, solarsystem, dimensionless):
        for dim, field in zip(
            usys.base_dimensions, usys._base_field_names, strict=True
        ):
            assert usys[dim] is getattr(usys, field)


def test_uconvert_to_realization_all_base_dimensions() -> None:
    """``uconvert``/``ustrip`` to a realization work for every base dimension.

    All four dimensions whose astropy alias differs from the declared field name
    (SI current/amount, CGS pressure/kinematic-viscosity) previously raised
    ``AttributeError`` from the ``__getitem__`` path.

    Expected units are read from the realization's *attributes*, never via
    ``__getitem__`` -- that is the path under test, so using it to compute the
    expected value would let the assertions pass vacuously.
    """
    assert u.uconvert(si, u.Q(3.0, "mA")).unit == si.electric_current
    assert u.uconvert(si, u.Q(1.0, "mmol")).unit == si.amount
    assert u.uconvert(cgs, u.Q(2.0, "Pa")).unit == cgs.pressure
    assert u.uconvert(cgs, u.Q(1.0, "m2 / s")).unit == cgs.kinematic_viscosity

    # ``ustrip`` resolves the unit through the same ``__getitem__`` path.
    assert np.isclose(u.ustrip(si, u.Q(3.0, "mA")), 0.003)
    assert np.isclose(u.ustrip(cgs, u.Q(2.0, "Pa")), 20.0)


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


def test_dimensionless_value_equality():
    """A freshly-built dimensionless system equals the ``dimensionless`` realization."""
    assert DimensionlessUnitSystem() == dimensionless


def test_building_system_does_not_mutate_realizations():
    """Building a system must never mutate the global si/cgs/dimensionless.

    Regression test: the built-in realizations were ``SingletonMixin`` frozen
    dataclasses, so constructing any SI/CGS/dimensionless-dimensioned system via
    the public ``unitsystem(...)`` API ran ``__init__`` on the cached singleton
    and overwrote its fields process-wide.
    """
    si_before = unitsystems.si.base_units
    cgs_before = unitsystems.cgs.base_units
    dimensionless_before = dimensionless.base_units

    # Build unrelated systems with the same dimensions but different units.
    custom_si = unitsystem("km", "g", "minute", "mol", "A", "K", "cd", "deg")
    custom_cgs = unitsystem("m", "kg", "s", "N", "J", "Pa", "Pa s", "m2 / s", "deg")
    custom_dimensionless = unitsystem(unit("percent"))

    # The globals are untouched.
    assert unitsystems.si.base_units == si_before
    assert unitsystems.cgs.base_units == cgs_before
    assert dimensionless.base_units == dimensionless_before

    # And the custom systems really do carry their own (different) units.
    assert custom_si.base_units != si_before
    assert custom_cgs.base_units != cgs_before
    assert custom_dimensionless.base_units != dimensionless_before


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
