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

import unxt as u
from unxt import dimension, unit, unitsystems
from unxt._src.unitsystems import base as us_base
from unxt._src.unitsystems.base import (
    _UNITSYSTEMS_BY_DIMSET,
    _UNITSYSTEMS_REGISTRY,
)
from unxt.unitsystems import (
    NAMED_UNIT_SYSTEMS,
    AbstractUnitSystem,
    AbstractUSysFlag,
    DimensionlessUnitSystem,
    DynamicalSimUSysFlag,
    StandardUSysFlag,
    cgs,
    dimensionless,
    equivalent,
    galactic,
    planck,
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
    # The by-dimension-set index is populated alongside the registry, so isolate
    # it too or classes defined in this test would leak into the real index.
    monkeypatch.setattr("unxt._src.unitsystems.base._UNITSYSTEMS_BY_DIMSET", {})
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
    # Derive the expected names from the registry so the test tracks add/remove
    # /rename of systems rather than a hard-coded list that can silently rot.
    assert NAMED_UNIT_SYSTEMS  # guard: an empty registry would vacuously pass
    for name in NAMED_UNIT_SYSTEMS:
        assert name in msg, f"{name!r} missing from the error message"
    # ... and points at the unit path for the likely intent. Match the specific
    # hint, not a bare "unit" (which also occurs in "unit system").
    assert "If you meant a unit" in msg
    assert "unxt.unit(" in msg


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


def test_unitsystem_from_list_of_units_builds_system() -> None:
    """A list/tuple of units builds a system from those units.

    A single-element list must not be mistaken for a named-system lookup
    (regression: ``unitsystem(["km"])`` used to unpack to ``unitsystem("km")``
    and raise).
    """
    usys = unitsystem(["km"])
    assert usys["length"] == unit("km")
    assert usys == unitsystem([unit("km")])

    # a multi-element list agrees with the variadic form
    assert unitsystem(["kpc", "Myr", "solMass", "rad"]) == unitsystem(
        "kpc", "Myr", "solMass", "rad"
    )


def test_unitsystem_identity_is_order_independent() -> None:
    """Set-like identity: the same units in any order give the same system.

    Results are equal and same-type; built-in shapes keep conventional order.
    """
    a = unitsystem("m", "s")
    b = unitsystem("s", "m")
    assert a == b
    assert type(a) is type(b)

    # built-in systems are matched by dimension set and keep their conventional
    # field order regardless of input order
    g1 = unitsystem("kpc", "Myr", "solMass", "rad")
    g2 = unitsystem("Myr", "rad", "kpc", "solMass")
    assert g1 == g2
    assert type(g1) is type(g2)
    assert g1 == galactic
    assert isinstance(g1, unitsystems.LTMAUnitSystem)
    # conventional field order (length, time, mass, angle) is preserved, not sorted
    assert g2.base_dimensions == galactic.base_dimensions


def test_natural_shape_construction_preserves_conventional_order() -> None:
    """Natural-system shapes keep conventional order for user construction.

    A shape like (length, mass, time, temperature) uses the pre-registered class
    in conventional order, not the alphabetical order a novel shape would sort
    into -- and is order-independent.
    """
    usys = unitsystem("m", "kg", "s", "K")
    assert type(usys) is type(planck)
    assert usys.base_dimensions == planck.base_dimensions
    assert unitsystem("K", "s", "m", "kg") == usys


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

    # Clean up custom unit system from the registry and its by-set index:
    del _UNITSYSTEMS_REGISTRY[MyUnitSystem._base_dimensions]
    del _UNITSYSTEMS_BY_DIMSET[frozenset(MyUnitSystem._base_dimensions)]


@pytest.mark.usefixtures("clean_unitsystems_registry")
def test_unitsystem_same_dimension_set_different_order_rejected():
    """A dimension *set* maps to a single class, regardless of field order.

    Two classes spanning the same dimensions but declaring them in a different
    order would leave the by-set index (and thus ``unitsystem(*units)``) to
    resolve by registration order; registration rejects the second one instead.
    """

    @dataclass(frozen=True, slots=True)
    class LengthTime(AbstractUnitSystem):
        length: Annotated[apyu.Unit, dimension("length")]
        time: Annotated[apyu.Unit, dimension("time")]

    with pytest.raises(ValueError, match="maps to a single unit-system class"):

        @dataclass(frozen=True, slots=True)
        class TimeLength(AbstractUnitSystem):
            time: Annotated[apyu.Unit, dimension("time")]
            length: Annotated[apyu.Unit, dimension("length")]

    # The rejected class left no partial entry in either registry; the by-set
    # index still points at the first (only) class for that set. Reference the
    # live dicts via the module so the fixture's monkeypatched copies are seen
    # (a top-level ``from ... import`` name would be bound to the original dict).
    dim_set = frozenset(LengthTime._base_dimensions)
    assert (dimension("time"), dimension("length")) not in us_base._UNITSYSTEMS_REGISTRY
    assert LengthTime._base_dimensions in us_base._UNITSYSTEMS_REGISTRY
    assert us_base._UNITSYSTEMS_BY_DIMSET[dim_set] is LengthTime


@pytest.mark.usefixtures("clean_unitsystems_registry")
def test_construction_consults_isolated_by_dimset_index():
    """``unitsystem(...)`` reads the fixture-isolated by-set index, not the real one.

    Regression: ``core`` bound ``_UNITSYSTEMS_BY_DIMSET`` at import, so a fixture
    that rebound only the ``base`` module's name left construction consulting the
    original index while registration wrote to the isolated one. Register a class
    under the fixture, then build its dimension set: ``unitsystem`` can only
    return *that* class if lookup and registration share the isolated dict.
    """

    @dataclass(frozen=True, slots=True)
    class IsolatedLengthTime(AbstractUnitSystem):
        length: Annotated[apyu.Unit, dimension("length")]
        time: Annotated[apyu.Unit, dimension("time")]

    built = unitsystem("m", "s")
    assert type(built) is IsolatedLengthTime
    assert built["length"] == unit("m")
    assert built["time"] == unit("s")


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
    """Atomic units set m_e = hbar = e = 4*pi*eps0 = 1.

    Verification strategy: m_e, e, and the Bohr radius are checked directly
    against their base-unit SI values; hbar is checked with ``decompose`` in the
    (length, mass, time) sub-basis; and 4*pi*eps0 by forming the electrostatic
    combination explicitly. ``decompose`` is not used for the charge itself:
    astropy cannot decompose a charge (``A s`` here, with ``A`` its irreducible
    electromagnetic base) into a basis that also contains ``s``.
    """
    usys = unitsystems.atomic
    length, mass, time = usys["length"], usys["mass"], usys["time"]
    charge = usys["electrical charge"]
    # base units carry the defining constants
    assert np.isclose((1 * length).to_value("m"), const.a0.value)
    assert np.isclose((1 * mass).to_value("kg"), const.m_e.value)
    assert np.isclose((1 * charge).to_value("C"), const.e.si.value)
    # hbar is unity in the (length, mass, time) sub-basis
    hbar = const.hbar.decompose([length, mass, time]).value
    assert np.isclose(hbar, 1.0)
    # 4*pi*eps0 is unity: eps0 has dimension charge**2 time**2 / (mass length**3)
    coulomb = (4 * np.pi * const.eps0) / (charge**2 * time**2 / mass / length**3)
    assert np.isclose(coulomb.decompose().value, 1.0)


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


def test_natural_unitsystem_pickle(tmp_path: Path) -> None:
    """Natural unit systems round-trip through pickle."""
    for usys in (
        unitsystems.hep,
        unitsystems.geometrized,
        unitsystems.planck,
        unitsystems.atomic,
    ):
        path = tmp_path / "usys.pkl"
        with path.open(mode="wb") as f:
            pickle.dump(usys, f)
        with path.open(mode="rb") as f:
            usys2 = pickle.load(f)  # noqa: S301
        assert usys == usys2
