"""Tests for the quantities strategy."""
# pylint: disable=unreachable

import hypothesis.strategies as st
import jax.numpy as jnp
from hypothesis import given, settings

import unxt as u
import unxt_hypothesis as ust


@given(q=ust.quantities(unit="m"))
@settings(max_examples=50)
def test_scalar_length(q: u.Quantity) -> None:
    """Test scalar length quantity generation."""
    assert isinstance(q, u.Quantity)
    assert q.shape == ()
    assert u.dimension_of(q) == u.dimension("length")


@given(q=ust.quantities(unit="m/s", shape=3))
@settings(max_examples=50)
def test_vector_velocity(q: u.Quantity) -> None:
    """Test 1D velocity quantity generation."""
    assert isinstance(q, u.Quantity)
    assert q.shape == (3,)
    assert u.dimension_of(q) == u.dimension("velocity")


@given(q=ust.quantities(unit="kg", shape=(2, 3)))
@settings(max_examples=50)
def test_2d_mass(q: u.Quantity) -> None:
    """Test 2D mass quantity generation."""
    assert isinstance(q, u.Quantity)
    assert q.shape == (2, 3)
    assert u.dimension_of(q) == u.dimension("mass")


@given(q=ust.quantities(unit="J", dtype=jnp.float32))
@settings(max_examples=30)
def test_float32_energy_dtype(q: u.Quantity) -> None:
    """Test float32 dtype for energy."""
    assert isinstance(q, u.Quantity)
    assert q.dtype == jnp.float32
    assert u.dimension_of(q) == u.dimension("energy")


@given(q=ust.quantities(unit="s", dtype=jnp.float32, shape=5))
@settings(max_examples=30)
def test_float32_time_dtype(q: u.Quantity) -> None:
    """Test float32 dtype for time."""
    assert isinstance(q, u.Quantity)
    assert q.dtype == jnp.float32
    assert q.shape == (5,)
    assert u.dimension_of(q) == u.dimension("time")


@given(
    q=ust.quantities(unit="m", shape=st.tuples(st.integers(1, 5), st.integers(1, 5)))
)
@settings(max_examples=30)
def test_variable_shape_strategy(q: u.Quantity) -> None:
    """Test with a shape strategy."""
    assert isinstance(q, u.Quantity)
    assert len(q.shape) == 2
    assert all(1 <= s <= 5 for s in q.shape)
    assert u.dimension_of(q) == u.dimension("length")


@given(
    q=ust.quantities(
        unit="m",
        shape=3,
        elements=st.floats(min_value=0.0, max_value=100.0, width=32),
    )
)
@settings(max_examples=30)
def test_custom_elements(q: u.Quantity) -> None:
    """Test with custom element strategy."""
    assert isinstance(q, u.Quantity)
    assert q.shape == (3,)
    # All values should be between 0 and 100
    assert jnp.all((q.value >= 0) & (q.value <= 100))


@given(q=ust.quantities(unit="m", shape=10, unique=True))
@settings(max_examples=20)
def test_unique_elements(q: u.Quantity) -> None:
    """Test unique elements constraint."""
    assert isinstance(q, u.Quantity)
    assert q.shape == (10,)
    # All elements should be unique
    assert len(jnp.unique(q.value)) == 10


@given(q=ust.quantities(unit="rad"))
@settings(max_examples=30)
def test_angle_unit(q: u.Quantity) -> None:
    """Test with angle unit."""
    assert isinstance(q, u.Quantity)
    assert u.dimension_of(q) == u.dimension("angle")


@given(q=ust.quantities(unit="N"))
@settings(max_examples=30)
def test_force_unit(q: u.Quantity) -> None:
    """Test with force unit."""
    assert isinstance(q, u.Quantity)
    assert u.dimension_of(q) == u.dimension("force")


@given(q=ust.quantities(unit="Pa"))
@settings(max_examples=30)
def test_pressure_unit(q: u.Quantity) -> None:
    """Test with pressure unit."""
    assert isinstance(q, u.Quantity)
    assert u.dimension_of(q) == u.dimension("pressure")


@given(q=ust.quantities(unit="kpc"))
@settings(max_examples=30)
def test_astronomical_unit(q: u.Quantity) -> None:
    """Test with astronomical unit."""
    assert isinstance(q, u.Quantity)
    assert u.dimension_of(q) == u.dimension("length")


@given(q=ust.quantities(unit="km/s", shape=(3,)))
@settings(max_examples=30)
def test_compound_unit(q: u.Quantity) -> None:
    """Test with compound unit."""
    assert isinstance(q, u.Quantity)
    assert q.shape == (3,)
    assert u.dimension_of(q) == u.dimension("velocity")


def test_accepts_shape_none() -> None:
    """Test that shape=None creates scalars."""
    strategy = ust.quantities(unit="m", shape=None)
    assert strategy is not None


def test_accepts_shape_int() -> None:
    """Test that shape as int works."""
    strategy = ust.quantities(unit="m", shape=5)
    assert strategy is not None


def test_accepts_shape_tuple() -> None:
    """Test that shape as tuple works."""
    strategy = ust.quantities(unit="m", shape=(2, 3))
    assert strategy is not None


@given(q=ust.quantities(unit="eV", shape=()))
@settings(max_examples=20)
def test_explicit_scalar_shape(q: u.Quantity) -> None:
    """Test explicit scalar shape ()."""
    assert isinstance(q, u.Quantity)
    assert q.shape == ()
    assert u.dimension_of(q) == u.dimension("energy")


# Tests combining quantities and units strategies


@given(q=ust.quantities(unit=ust.units("length"), shape=3))
@settings(max_examples=50)
def test_length_varying_units(q: u.Quantity) -> None:
    """Test length quantity with varying units from units strategy."""
    assert isinstance(q, u.Quantity)
    assert q.shape == (3,)
    # Should always have length dimension
    assert u.dimension_of(q) == u.dimension("length")


@given(q=ust.quantities(unit=ust.units("velocity"), shape=()))
@settings(max_examples=50)
def test_scalar_velocity_varying_units(q: u.Quantity) -> None:
    """Test scalar velocity with varying units."""
    assert isinstance(q, u.Quantity)
    assert q.shape == ()
    assert u.dimension_of(q) == u.dimension("velocity")


@given(q=ust.quantities(unit=ust.units("mass"), shape=(2, 3)))
@settings(max_examples=30)
def test_2d_mass_varying_units(q: u.Quantity) -> None:
    """Test 2D mass quantity with varying units."""
    assert isinstance(q, u.Quantity)
    assert q.shape == (2, 3)
    assert u.dimension_of(q) == u.dimension("mass")


@given(q=ust.quantities(unit=ust.units("energy", max_complexity=3), dtype=jnp.float32))
@settings(max_examples=30)
def test_energy_complex_units(q: u.Quantity) -> None:
    """Test energy with complex compound units."""
    assert isinstance(q, u.Quantity)
    assert q.dtype == jnp.float32
    assert u.dimension_of(q) == u.dimension("energy")


@given(
    q=ust.quantities(
        unit=ust.units("length", integer_powers=False),
        shape=5,
        dtype=jnp.float32,
    )
)
@settings(max_examples=30)
def test_length_non_integer_powers(q: u.Quantity) -> None:
    """Test length with non-integer power units."""
    assert isinstance(q, u.Quantity)
    assert q.shape == (5,)
    assert q.dtype == jnp.float32
    assert u.dimension_of(q) == u.dimension("length")


# ==============================================================================
# Tests for dtype as a strategy
# ==============================================================================


@given(q=ust.quantities("m", dtype=st.sampled_from([jnp.float32, jnp.int32])))
@settings(max_examples=50)
def test_dtype_strategy_float_types(q: u.Quantity) -> None:
    """Test dtype as a strategy with different types."""
    assert isinstance(q, u.Quantity)
    assert q.dtype in (jnp.float32, jnp.int32)
    assert u.dimension_of(q) == u.dimension("length")


@given(
    q=ust.quantities(
        unit="kg",
        shape=(3,),
        dtype=st.sampled_from([jnp.float32, jnp.complex64]),
    )
)
@settings(max_examples=30)
def test_dtype_strategy_with_complex(q: u.Quantity) -> None:
    """Test dtype strategy including complex types."""
    assert isinstance(q, u.Quantity)
    assert q.shape == (3,)
    assert q.dtype in (jnp.float32, jnp.complex64)


@given(
    q=ust.quantities(
        unit="m/s",
        shape=(2, 3),
        dtype=st.just(jnp.float32),
    )
)
@settings(max_examples=20)
def test_dtype_strategy_just_float32(q: u.Quantity) -> None:
    """Test using st.just for a single dtype via strategy."""
    assert isinstance(q, u.Quantity)
    assert q.dtype == jnp.float32
    assert q.shape == (2, 3)


# ==============================================================================
# Tests for unit as a dimension
# ==============================================================================


@given(q=ust.quantities(unit=u.dimension("length"), shape=3))
@settings(max_examples=50)
def test_unit_from_length_dimension(q: u.Quantity) -> None:
    """Test generating units from length dimension."""
    assert isinstance(q, u.Quantity)
    assert q.shape == (3,)
    assert u.dimension_of(q) == u.dimension("length")


@given(q=ust.quantities(unit=u.dimension("velocity"), shape=()))
@settings(max_examples=50)
def test_unit_from_velocity_dimension(q: u.Quantity) -> None:
    """Test generating units from velocity dimension."""
    assert isinstance(q, u.Quantity)
    assert q.shape == ()
    assert u.dimension_of(q) == u.dimension("velocity")


@given(q=ust.quantities(unit=u.dimension("mass"), shape=(2, 3), dtype=jnp.float32))
@settings(max_examples=30)
def test_unit_from_mass_dimension_with_dtype(q: u.Quantity) -> None:
    """Test dimension with explicit dtype."""
    assert isinstance(q, u.Quantity)
    assert q.shape == (2, 3)
    assert q.dtype == jnp.float32
    assert u.dimension_of(q) == u.dimension("mass")


@given(q=ust.quantities(unit=u.dimension("energy"), shape=5))
@settings(max_examples=30)
def test_unit_from_energy_dimension(q: u.Quantity) -> None:
    """Test generating units from energy dimension."""
    assert isinstance(q, u.Quantity)
    assert q.shape == (5,)
    assert u.dimension_of(q) == u.dimension("energy")


# ==============================================================================
# Tests for unit as a dimension strategy
# ==============================================================================


@given(
    q=ust.quantities(
        st.sampled_from([u.dimension("length"), u.dimension("mass")]),
        shape=(),
    )
)
@settings(max_examples=50)
def test_unit_from_dimension_strategy(q: u.Quantity) -> None:
    """Test generating units from a strategy of dimensions."""
    assert isinstance(q, u.Quantity)
    assert q.shape == ()
    dim = u.dimension_of(q)
    assert dim in (u.dimension("length"), u.dimension("mass"))


@given(
    q=ust.quantities(
        st.sampled_from([u.dimension("velocity"), u.dimension("acceleration")]),
        shape=3,
        dtype=jnp.float32,
    )
)
@settings(max_examples=40)
def test_dimension_strategy_kinematic_quantities(q: u.Quantity) -> None:
    """Test dimension strategy for kinematic quantities."""
    assert isinstance(q, u.Quantity)
    assert q.shape == (3,)
    assert q.dtype == jnp.float32
    dim = u.dimension_of(q)
    assert dim in (u.dimension("velocity"), u.dimension("acceleration"))


@given(
    q=ust.quantities(
        st.sampled_from([u.dimension("angle"), u.dimension("dimensionless")]),
        shape=(2, 2),
    )
)
@settings(max_examples=30)
def test_dimension_strategy_angle_dimensionless(q: u.Quantity) -> None:
    """Test dimension strategy with angle and dimensionless."""
    assert isinstance(q, u.Quantity)
    assert q.shape == (2, 2)
    dim = u.dimension_of(q)
    assert dim in (u.dimension("angle"), u.dimension("dimensionless"))


# ==============================================================================
# Tests combining dtype and dimension strategies
# ==============================================================================


@given(
    q=ust.quantities(
        unit=st.sampled_from([u.dimension("length"), u.dimension("time")]),
        dtype=st.sampled_from([jnp.float32, jnp.int32]),
        shape=(),
    )
)
@settings(max_examples=40)
def test_combined_dimension_and_dtype_strategies(q: u.Quantity) -> None:
    """Test combining dimension and dtype strategies."""
    assert isinstance(q, u.Quantity)
    assert q.shape == ()
    assert q.dtype in (jnp.float32, jnp.int32)
    dim = u.dimension_of(q)
    assert dim in (u.dimension("length"), u.dimension("time"))


@given(
    q=ust.quantities(
        unit=u.dimension("force"),
        dtype=st.sampled_from([jnp.float32, jnp.complex64]),
        shape=(3, 3),
    )
)
@settings(max_examples=30)
def test_force_dimension_with_dtype_strategy(q: u.Quantity) -> None:
    """Test force dimension with varying dtype."""
    assert isinstance(q, u.Quantity)
    assert q.shape == (3, 3)
    assert q.dtype in (jnp.float32, jnp.complex64)
    assert u.dimension_of(q) == u.dimension("force")


# ==============================================================================
# Tests for elements parameter (value ranges)
# ==============================================================================


@given(
    q=ust.quantities(
        "kpc", shape=3, elements=st.floats(min_value=0, max_value=100, width=32)
    )
)
@settings(max_examples=50)
def test_positive_distance_elements(q: u.Quantity) -> None:
    """Test elements parameter for positive distances."""
    assert isinstance(q, u.Quantity)
    assert q.unit == u.unit("kpc")
    assert q.shape == (3,)
    # All values should be non-negative (suitable for distances)
    assert jnp.all(q.value >= 0)
    assert jnp.all(q.value <= 100)


@given(
    q=ust.quantities(
        "deg",
        shape=(),
        elements=st.floats(min_value=0, max_value=360, allow_nan=False, width=32),
    )
)
@settings(max_examples=50)
def test_longitude_angle_range(q: u.Quantity) -> None:
    """Test elements for longitude angles (0-360 degrees)."""
    assert isinstance(q, u.Quantity)
    assert q.unit == u.unit("deg")
    assert 0 <= q.value <= 360


@given(
    q=ust.quantities(
        "deg",
        shape=100,
        elements=st.floats(min_value=-90, max_value=90, allow_nan=False, width=32),
    )
)
@settings(max_examples=30)
def test_latitude_angle_range(q: u.Quantity) -> None:
    """Test elements for latitude angles (-90 to 90 degrees)."""
    assert isinstance(q, u.Quantity)
    assert q.unit == u.unit("deg")
    assert q.shape == (100,)
    assert jnp.all(q.value >= -90)
    assert jnp.all(q.value <= 90)


@given(
    q=ust.quantities(
        "rad",
        shape=(5, 5),
        elements=st.floats(min_value=0, max_value=6.0, allow_nan=False, width=32),
    )
)
@settings(max_examples=30)
def test_azimuthal_angle_radians(q: u.Quantity) -> None:
    """Test elements for azimuthal angles in radians (0 to 2π)."""
    assert isinstance(q, u.Quantity)
    assert q.unit == u.unit("rad")
    assert q.shape == (5, 5)
    assert jnp.all(q.value >= 0)
    assert jnp.all(q.value <= 6.0)


@given(
    q=ust.quantities(
        "m",
        shape=10,
        elements=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, width=32),
    )
)
@settings(max_examples=30)
def test_physical_length_scale(q: u.Quantity) -> None:
    """Test elements for physical lengths with realistic bounds."""
    assert isinstance(q, u.Quantity)
    assert q.shape == (10,)
    # All values in reasonable range (1 meter to 1 kilometer)
    assert jnp.all(q.value >= 1.0)
    assert jnp.all(q.value <= 1000.0)


# ==============================================================================
# Tests for wrap_to strategy
# ==============================================================================


@given(
    angle=ust.wrap_to(
        ust.quantities("deg", quantity_cls=u.Angle),
        min=u.Quantity(0, "deg"),
        max=u.Quantity(360, "deg"),
    )
)
@settings(max_examples=50)
def test_wrap_to_longitude_range(angle: u.Angle) -> None:
    """Test wrapping angles to longitude range [0, 360)."""
    assert isinstance(angle, u.Angle)
    # wrap_to wraps to [min, max), but floating point can give us max
    # Use ustrip to properly handle unit conversions
    angle_deg = u.ustrip("deg", angle)
    assert 0 <= angle_deg <= 360


@given(
    angle=ust.wrap_to(
        ust.quantities("rad", quantity_cls=u.Angle, shape=10),
        min=u.Quantity(0, "rad"),
        max=u.Quantity(6.28, "rad"),
    )
)
@settings(max_examples=30)
def test_wrap_to_angle_array(angle: u.Angle) -> None:
    """Test wrapping angle arrays to range."""
    assert isinstance(angle, u.Angle)
    assert angle.shape == (10,)
    # wrap_to wraps to [min, max), but floating point can give us max
    # Use ustrip to properly handle unit conversions
    angle_rad = u.ustrip("rad", angle)
    assert jnp.all(angle_rad >= 0)
    assert jnp.all(angle_rad <= 6.28)


@given(
    lat=ust.wrap_to(
        ust.quantities("deg", quantity_cls=u.Angle, shape=()),
        min=u.Quantity(-90, "deg"),
        max=u.Quantity(90, "deg"),
    )
)
@settings(max_examples=30)
def test_wrap_to_latitude_range(lat: u.Angle) -> None:
    """Test wrapping angles to latitude range [-90, 90)."""
    assert isinstance(lat, u.Angle)
    # wrap_to wraps to [min, max), but floating point can give us max
    # Use ustrip to properly handle unit conversions
    lat_deg = u.ustrip("deg", lat)
    assert -90 <= lat_deg <= 90


# Tests for quantity_cls parameter
# ==============================================================================


@given(angle=ust.quantities("rad", quantity_cls=u.Angle, shape=3))
@settings(max_examples=30)
def test_quantities_angle_class(angle: u.Angle) -> None:
    """Test creating Angle instances via quantity_cls parameter."""
    assert isinstance(angle, u.Angle)
    assert angle.unit == u.unit("rad")
    assert angle.shape == (3,)


@given(quantity=ust.quantities("m", quantity_cls=u.Quantity, shape=()))
@settings(max_examples=30)
def test_quantities_quantity_class(quantity: u.Quantity) -> None:
    """Test creating Quantity instances via quantity_cls parameter."""
    assert isinstance(quantity, u.Quantity)
    assert quantity.unit == u.unit("m")
    assert quantity.shape == ()
