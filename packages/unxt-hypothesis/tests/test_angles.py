"""Tests for the angles strategy."""

import hypothesis.strategies as st
from hypothesis import given, settings

import unxt as u
from unxt_hypothesis import angles


@given(angle=angles())
@settings(max_examples=50)
def test_basic_angle(angle: u.Angle) -> None:
    """Test basic angle generation."""
    assert isinstance(angle, u.Angle)
    assert angle.shape == ()


@given(angle=angles(unit="deg", elements={"min_value": -180, "max_value": 180}))
@settings(max_examples=50)
def test_angle_with_bounds(angle: u.Angle) -> None:
    """Test angle generation with value bounds."""
    assert isinstance(angle, u.Angle)
    assert -180 <= angle.to("deg").value <= 180


@given(angle=angles(unit="rad"))
@settings(max_examples=50)
def test_angle_radians(angle: u.Angle) -> None:
    """Test angle generation in radians."""
    assert isinstance(angle, u.Angle)
    assert angle.unit == "rad"


@given(angle=angles(wrap_to=st.just((u.Q(0, "deg"), u.Q(360, "deg")))))
@settings(max_examples=50)
def test_angle_with_wrapping(angle: u.Angle) -> None:
    """Test angle generation with wrapping."""
    assert isinstance(angle, u.Angle)
    # The wrap_to strategy wraps values to [min, max) range
    # but floating point can give us exactly max
    angle_deg = angle.to("deg").value
    assert 0 <= angle_deg <= 360


@given(angle=angles(wrap_to=st.just((u.Q(-180, "deg"), u.Q(180, "deg")))))
@settings(max_examples=50)
def test_angle_with_symmetric_wrapping(angle: u.Angle) -> None:
    """Test angle generation with symmetric wrapping."""
    assert isinstance(angle, u.Angle)
    # The wrap_to strategy wraps values to [min, max) range
    # but floating point can give us exactly max
    angle_deg = angle.to("deg").value
    assert -180 <= angle_deg <= 180


@given(angle=angles(shape=5))
@settings(max_examples=30)
def test_angle_vector(angle: u.Angle) -> None:
    """Test vector angle generation."""
    assert isinstance(angle, u.Angle)
    assert angle.shape == (5,)


@given(angle=angles(shape=(2, 3)))
@settings(max_examples=30)
def test_angle_2d(angle: u.Angle) -> None:
    """Test 2D angle array generation."""
    assert isinstance(angle, u.Angle)
    assert angle.shape == (2, 3)


@given(angle=angles(shape=st.tuples(st.integers(1, 5), st.integers(1, 5))))
@settings(max_examples=30)
def test_angle_dynamic_shape(angle: u.Angle) -> None:
    """Test angle generation with dynamic shape."""
    assert isinstance(angle, u.Angle)
    assert len(angle.shape) == 2
    assert 1 <= angle.shape[0] <= 5
    assert 1 <= angle.shape[1] <= 5
