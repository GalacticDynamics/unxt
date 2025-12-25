"""Tests for st.from_type() with registered type strategies."""

import hypothesis.strategies as st
import pytest
from hypothesis import HealthCheck, given, settings

import unxt as u
import unxt_hypothesis as ust  # noqa: F401 - Import to trigger type strategy registrations


@given(q=st.from_type(u.AbstractQuantity))
@settings(max_examples=20)
def test_from_type_abstract_quantity(q: u.AbstractQuantity) -> None:
    """Test st.from_type() with AbstractQuantity."""
    assert isinstance(q, u.AbstractQuantity)
    # Should have a dimension
    assert u.dimension_of(q) is not None


@given(a=st.from_type(u.Angle))
@settings(max_examples=20)
def test_from_type_angle(a: u.Angle) -> None:
    """Test st.from_type() with Angle."""
    assert isinstance(a, u.Angle)
    # Should have angle dimension
    assert u.dimension_of(a) == u.dimension("angle")


@pytest.mark.filterwarnings(
    "ignore:Do not use the `random` module inside "
    "strategies:hypothesis.errors.HypothesisDeprecationWarning"
)
@given(us=st.from_type(u.AbstractUnitSystem))
@settings(
    max_examples=20,
    suppress_health_check=[HealthCheck.differing_executors],
)
def test_from_type_abstract_unitsystem(us: u.AbstractUnitSystem) -> None:
    """Test st.from_type() with AbstractUnitSystem."""
    assert isinstance(us, u.AbstractUnitSystem)
    # Unit systems can be empty or populated
    assert len(us) >= 0
