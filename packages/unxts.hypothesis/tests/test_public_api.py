"""Smoke tests for the unxts.hypothesis public API."""

import pytest
import unxts.hypothesis
from hypothesis import given, strategies as st

import unxt as u


def test_all_symbols_present():
    for name in unxts.hypothesis.__all__:
        assert hasattr(unxts.hypothesis, name), f"unxts.hypothesis missing: {name}"


def test_version_is_a_string():
    assert isinstance(unxts.hypothesis.__version__, str)


@given(q=unxts.hypothesis.quantities(unit="km/s"))
def test_a_strategy_produces_a_quantity(q):
    assert q.unit == u.unit("km/s")


@given(dim=unxts.hypothesis.named_dimensions())
def test_named_dimensions_strategy(dim):
    assert u.dimension_of(dim) is dim


@given(unit=unxts.hypothesis.units("length"))
def test_units_strategy(unit):
    assert u.dimension_of(unit) == u.dimension("length")


# The strategy builds a unit system, and beartype (enabled in this suite via
# `UNXT_ENABLE_RUNTIME_TYPECHECKING`) samples container items with the global
# `random`, which Hypothesis deprecates inside a strategy. Not our randomness.
@pytest.mark.filterwarnings("ignore::hypothesis.errors.HypothesisDeprecationWarning")
@given(usys=unxts.hypothesis.unitsystems("m", "s", "kg", "rad"))
def test_unitsystems_strategy(usys):
    assert isinstance(usys, u.AbstractUnitSystem)


@given(angle=unxts.hypothesis.angles())
def test_angles_strategy(angle):
    assert isinstance(angle, u.Angle)


@given(
    angle=unxts.hypothesis.angles(
        wrap_to=st.just((u.Q(0, "deg"), u.Q(360, "deg"))), unit="deg"
    )
)
def test_angles_strategy_wrapped(angle):
    assert 0 <= float(angle.ustrip("deg")) < 360


@given(
    q=unxts.hypothesis.wrap_to(
        unxts.hypothesis.quantities("deg"),
        min=st.just(u.Q(0, "deg")),
        max=st.just(u.Q(360, "deg")),
    )
)
def test_wrap_to_strategy(q):
    assert 0 <= float(q.ustrip("deg")) < 360


@given(unit=unxts.hypothesis.derived_units("m", max_complexity=0))
def test_derived_units_without_compound_factors(unit):
    assert u.dimension_of(unit) == u.dimension("length")


@given(unit=unxts.hypothesis.derived_units("m", integer_powers=False))
def test_derived_units_with_fractional_powers(unit):
    assert u.dimension_of(unit) == u.dimension("length")


@given(unit=unxts.hypothesis.derived_units("m*s*kg*A*K*mol*cd*rad"))
def test_derived_units_when_every_si_base_is_used(unit):
    """No SI base is left over to build a cancelling factor from."""
    assert u.dimension_of(unit) == u.dimension_of(u.unit("m*s*kg*A*K*mol*cd*rad"))
