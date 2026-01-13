# pylint: disable=import-error, too-many-lines

"""Test the Array API."""

import pickle
import re

import astropy.units as apyu
import equinox as eqx
import jax
import jax.numpy as jax_xp
import numpy as np
import pytest
from hypothesis import example, given, settings, strategies as st
from hypothesis.extra.array_api import make_strategies_namespace
from hypothesis.extra.numpy import array_shapes as np_array_shapes, arrays as np_arrays
from jax.dtypes import canonicalize_dtype
from jaxtyping import Array, TypeCheckError
from plum import convert, parametric

import quaxed.numpy as jnp

import unxt as u
import unxt_hypothesis as ust
from unxt.quantity import AbstractParametricQuantity

xps = make_strategies_namespace(jax_xp)


jaxint = canonicalize_dtype(int)
jaxfloat = canonicalize_dtype(float)

integers_strategy = st.integers(
    min_value=np.iinfo(jaxint).min, max_value=np.iinfo(jaxint).max
)
floats_strategy = st.floats(
    min_value=np.finfo(jaxfloat).min, max_value=np.finfo(jaxfloat).max
)


@given(
    value=integers_strategy
    | floats_strategy
    | st.lists(integers_strategy)
    | st.tuples(integers_strategy)
    | st.lists(floats_strategy)
    | st.tuples(floats_strategy)
    # | st.lists(st.lists(integers_strategy))  # TODO: enable nested lists
    # | st.lists(st.lists(floats_strategy))
    | np_arrays(
        dtype=np.float32,
        shape=np_array_shapes(),
        elements={"allow_nan": False, "allow_infinity": False},
    )
    | xps.arrays(
        dtype=np.float32,
        shape=xps.array_shapes(),
        elements={"allow_nan": False, "allow_infinity": False},
    )
)
@example(value=0)  # int
@example(value=1.0)  # int
@example(value=[1])  # list[int]
@example(value=(1,))  # tuple[int, ...]
@example(value=[1.0])  # list[float]
@example(value=(1.0,))  # list[float]
@example(value=[[1]])  # list[list[int]]
@example(value=[[1.0]])  # list[list[int]]
def test_properties(value):
    """Test the properties of Quantity."""
    q = u.Q(value, "m")
    expected = jnp.asarray(value)

    # Test the value
    assert np.array_equal(q.value, expected)

    # Test the shape
    assert q.shape == expected.shape

    # Test materialise
    with pytest.raises(RuntimeError):
        q.materialise()

    # Test aval
    assert q.aval() == jax.typeof(expected)

    # Test enable_materialise
    assert np.array_equal(q.enable_materialise().value, q.value)


def test_parametric():
    """Test the parametric strategy."""
    # Inferred
    q = u.Q(1, "m")
    (dims,) = q._type_parameter
    assert dims == u.dimension("length")

    # Explicit
    q = u.Q["length"](1, "m")
    (dims,) = q._type_parameter
    assert dims == u.dimension("length")

    q = u.Q["length"](jnp.ones((1, 2)), "m")
    (dims,) = q._type_parameter
    assert dims == u.dimension("length")

    # type-checks
    with pytest.raises(ValueError, match=re.escape("Physical type mismatch.")):
        u.Q["time"](1, "m")


@pytest.mark.parametrize("unit", [apyu.m, "meter"])
def test_unit(unit):
    """Test the unit."""
    assert u.Q(1, unit).unit == unit


def test_array_namespace():
    """Test the array namespace."""
    assert u.Q(1, "m").__array_namespace__() is jnp


def test_uconvert():
    """Test the ``u.Q.uconvert`` method."""
    q = u.Q(1, "m")
    assert jnp.equal(q.uconvert("km"), u.Q(0.001, "km"))


def test_ustrip():
    """Test the ``u.Q.ustrip`` method."""
    q = u.Q(1, "m")
    assert q.ustrip("km") == u.Q(0.001, "km").value


def test_uconvert_value_with_units():
    """Test the ``u.uconvert_value`` function with unit objects."""
    # Convert 1 km to meters
    result = u.uconvert_value(u.unit("m"), u.unit("km"), 1)
    assert jnp.isclose(result, 1000.0)

    # Convert array of values
    result = u.uconvert_value(u.unit("m"), u.unit("km"), jnp.array([1, 2, 3]))
    assert np.allclose(result, [1000.0, 2000.0, 3000.0])


def test_uconvert_value_with_strings():
    """Test the ``u.uconvert_value`` function with unit strings."""
    # Convert 1 km to meters
    result = u.uconvert_value("m", "km", 1)
    assert jnp.isclose(result, 1000.0)

    # Convert with different units
    result = u.uconvert_value("cm", "m", 5)
    assert jnp.isclose(result, 500.0)

    # Array of values
    result = u.uconvert_value("mm", "cm", jnp.array([1, 2, 3]))
    assert np.allclose(result, [10.0, 20.0, 30.0])


def test_uconvert_value_with_quantity():
    """Test the ``u.uconvert_value`` convenience dispatch with Quantity objects."""
    # Convert a Quantity object using uconvert_value
    q = u.Q(1, "km")
    result = u.uconvert_value("m", "km", q)
    assert isinstance(result, u.Quantity)
    assert jnp.isclose(result.value, 1000.0)
    assert result.unit == u.unit("m")

    # With unit objects
    result = u.uconvert_value(u.unit("m"), u.unit("km"), q)
    assert isinstance(result, u.Quantity)
    assert jnp.isclose(result.value, 1000.0)
    assert result.unit == u.unit("m")


def test_uconvert_value_with_unit_system():
    """Test the ``u.uconvert_value`` function with unit systems."""
    # Convert to galactic unit system preferred units for length
    result = u.uconvert_value(u.unitsystems.galactic, "km", 1e17)
    # Should convert km to kpc (the galactic system's preferred length unit)
    expected_unit = u.unitsystems.galactic[u.dimension("length")]
    assert expected_unit == u.unit("kpc")
    # 1e17 km in kpc
    expected = u.uconvert_value("kpc", "km", 1e17)
    assert jnp.isclose(result, expected)


def test_uconvert_value_incompatible_units():
    """Test that incompatible unit conversions raise errors."""
    # Length to time - should raise
    with pytest.raises((apyu.UnitConversionError, Exception)):
        u.uconvert_value("s", "m", 1)

    # Mass to length - should raise
    with pytest.raises((apyu.UnitConversionError, Exception)):
        u.uconvert_value("kg", "m", 1)


def test_uconvert_value_vs_uconvert():
    """Test the relationship between uconvert_value and uconvert."""
    # uconvert_value on raw values should match uconvert on quantities
    q = u.Q(1, "km")

    # Using uconvert on quantity
    result_quantity = u.uconvert("m", q)

    # Using uconvert_value on raw value
    result_value = u.uconvert_value("m", "km", 1)

    # Values should match
    assert jnp.isclose(result_quantity.value, result_value)
    assert result_quantity.unit == u.unit("m")


def test_uconvert_value_with_array_quantities():
    """Test uconvert_value convenience dispatch with array Quantities."""
    q = u.Q([1, 2, 3], "km")
    result = u.uconvert_value("m", "km", q)

    assert isinstance(result, u.Quantity)
    assert np.allclose(result.value, [1000.0, 2000.0, 3000.0])
    assert result.unit == u.unit("m")


def test_uconvert_value_preserves_dtype():
    """Test that uconvert_value preserves input dtype."""
    # Float32 input
    result = u.uconvert_value("m", "km", jnp.array(1.0, dtype=jnp.float32))
    assert result.dtype == jnp.float32

    # Int32 input
    result = u.uconvert_value("m", "km", jnp.array(1, dtype=jnp.int32))
    assert result.dtype == jnp.float32  # Conversion produces float


def test_uconvert_value_with_jax_transformations():
    """Test that uconvert_value works with JAX transformations."""

    def convert_km_to_m(x):
        return u.uconvert_value("m", "km", x)

    # JIT compilation
    jitted_convert = jax.jit(convert_km_to_m)
    result = jitted_convert(5.0)
    assert jnp.isclose(result, 5000.0)

    # vmap over array
    values = jnp.array([1, 2, 3, 4, 5])
    result = jax.vmap(convert_km_to_m)(values)
    expected = jnp.array([1000, 2000, 3000, 4000, 5000], dtype=jnp.float32)
    assert np.allclose(result, expected)


def test_uconvert_value_error_handling_quantity():
    """Test error handling when converting incompatible Quantity with uconvert_value."""
    q_length = u.Q(1, "m")
    # Try to convert length quantity to time units - should raise
    with pytest.raises(apyu.UnitConversionError, match="not convertible"):
        u.uconvert_value("s", "m", q_length)


def test_getitem():
    """Test the ``u.Q.__getitem__`` method."""
    # Scalar - cannot index
    q = u.Q(1, "m")
    with pytest.raises((TypeError, IndexError)):
        _ = q[0]

    # 1D array
    q = u.Q([1, 2, 3], "m")
    assert q[0] == u.Q(1, "m")
    assert q[1] == u.Q(2, "m")
    assert q[-1] == u.Q(3, "m")

    # Slicing
    assert np.array_equal(q[:2], u.Q([1, 2], "m"))
    assert np.array_equal(q[1:], u.Q([2, 3], "m"))
    assert np.array_equal(q[::2], u.Q([1, 3], "m"))

    # 2D array
    q = u.Q([[1, 2, 3], [4, 5, 6]], "m")
    assert q[0, 0] == u.Q(1, "m")
    assert np.array_equal(q[0, :], u.Q([1, 2, 3], "m"))
    assert np.array_equal(q[:, 0], u.Q([1, 4], "m"))
    assert np.array_equal(q[0], u.Q([1, 2, 3], "m"))

    # Boolean indexing
    q = u.Q([1, 2, 3, 4], "m")
    mask = jnp.array([True, False, True, False])
    assert np.array_equal(q[mask], u.Q([1, 3], "m"))

    # Advanced indexing
    indices = jnp.array([0, 2])
    assert np.array_equal(q[indices], u.Q([1, 3], "m"))


def test_len():
    """Test the ``len(Quantity)`` method."""
    # Length 3
    q = u.Q([1, 2, 3], "m")
    assert len(q) == 3

    # Scalar
    q = u.Q(1, "m")
    assert len(q) == 0


def test_add():
    """Test the ``Quantity.__add__`` method."""
    # Scalar addition
    q1 = u.Q(1, "m")
    q2 = u.Q(2, "m")
    assert q1 + q2 == u.Q(3, "m")

    # Array addition
    q1 = u.Q([1, 2, 3], "m")
    q2 = u.Q([4, 5, 6], "m")
    assert np.array_equal(q1 + q2, u.Q([5, 7, 9], "m"))

    # Unit conversion in addition
    q1 = u.Q(1, "m")
    q2 = u.Q(1, "km")
    result = q1 + q2
    assert jnp.isclose(result.value, 1001.0)
    assert result.unit == u.unit("m")

    # Incompatible units
    q1 = u.Q(1, "m")
    q2 = u.Q(1, "s")
    with pytest.raises(Exception):  # noqa: B017, PT011
        _ = q1 + q2

    # Broadcasting
    q1 = u.Q([[1, 2], [3, 4]], "m")
    q2 = u.Q([10, 20], "m")
    result = q1 + q2
    assert np.array_equal(result, u.Q([[11, 22], [13, 24]], "m"))


def test_radd():
    """Test the ``Quantity.__radd__`` method."""
    # Dimensionless quantities can be added with each other
    q1 = u.Q(1, "")
    q2 = u.Q(5, "")
    result = q1 + q2
    assert jnp.isclose(result.value, 6.0)
    assert result.unit == u.unit("")

    # Non-zero scalar should fail with dimensioned quantities
    q = u.Q(1, "m")
    with pytest.raises(Exception):  # noqa: B017, PT011
        _ = 5 + q


def test_sub():
    """Test the ``Quantity.__sub__`` method."""
    # Scalar subtraction
    q1 = u.Q(3, "m")
    q2 = u.Q(2, "m")
    result = q1 - q2
    assert jnp.isclose(result.value, 1.0)
    assert result.unit == u.unit("m")

    # Array subtraction
    q1 = u.Q([5, 7, 9], "m")
    q2 = u.Q([1, 2, 3], "m")
    assert np.array_equal(q1 - q2, u.Q([4, 5, 6], "m"))

    # Unit conversion in subtraction
    q1 = u.Q(1.0, "km")
    q2 = u.Q(1.0, "m")
    result = q1 - q2
    assert jnp.isclose(result.value, 0.999, rtol=1e-3)
    assert result.unit == u.unit("km")

    # Incompatible units
    q1 = u.Q(1, "m")
    q2 = u.Q(1, "s")
    with pytest.raises(Exception):  # noqa: B017, PT011
        _ = q1 - q2


def test_rsub():
    """Test the ``Quantity.__rsub__`` method."""
    # Subtracting dimensionless quantity from scalar
    q = u.Q(1, "")
    result = 5 - q
    assert jnp.isclose(result.value, 4.0)
    assert result.unit == u.unit("")

    # Non-zero scalar should fail with dimensioned quantity
    q = u.Q(1, "m")
    with pytest.raises(Exception):  # noqa: B017, PT011
        _ = 5 - q


def test_mul():
    """Test the ``Quantity.__mul__`` method."""
    # Scalar multiplication
    q1 = u.Q(2, "m")
    q2 = u.Q(3, "s")
    result = q1 * q2
    assert result.value == 6
    assert result.unit == u.unit("m s")

    # Multiplication by scalar (dimensionless)
    q = u.Q(2, "m")
    assert q * 3 == u.Q(6, "m")
    assert 3 * q == u.Q(6, "m")

    # Array multiplication
    q1 = u.Q([1, 2, 3], "m")
    q2 = u.Q([4, 5, 6], "s")
    result = q1 * q2
    assert np.array_equal(result.value, [4, 10, 18])
    assert result.unit == u.unit("m s")

    # Multiplication creates compound units
    length = u.Q(5, "m")
    width = u.Q(3, "m")
    area = length * width
    assert area.value == 15
    assert u.dimension_of(area) == u.dimension("area")

    # Broadcasting
    q1 = u.Q([[1, 2], [3, 4]], "m")
    q2 = u.Q([2, 3], "s")
    result = q1 * q2
    assert np.array_equal(result.value, [[2, 6], [6, 12]])
    assert result.unit == u.unit("m s")


def test_rmul():
    """Test the ``Quantity.__rmul__`` method."""
    # Scalar * Quantity
    q = u.Q(2, "m")
    assert 3 * q == u.Q(6, "m")

    # Array scalar * Quantity
    q = u.Q([1, 2, 3], "m")
    assert np.array_equal(2 * q, u.Q([2, 4, 6], "m"))

    # NumPy/JAX array * Quantity
    arr = jnp.array([1, 2, 3])
    q = u.Q(2, "m")
    result = arr * q
    assert np.array_equal(result.value, [2, 4, 6])
    assert result.unit == u.unit("m")


def test_matmul():
    """Test the ``Quantity.__matmul__`` method."""
    # Vector dot product
    q1 = u.Q([1, 2, 3], "m")
    q2 = u.Q([4, 5, 6], "kg")
    result = q1 @ q2
    assert result.value == 32  # 1*4 + 2*5 + 3*6
    assert result.unit == u.unit("m kg")

    # Matrix-vector multiplication
    q1 = u.Q([[1, 2], [3, 4]], "m")
    q2 = u.Q([5, 6], "s")
    result = q1 @ q2
    assert np.array_equal(result.value, [17, 39])  # [1*5+2*6, 3*5+4*6]
    assert result.unit == u.unit("m s")

    # Matrix-matrix multiplication
    q1 = u.Q([[1, 2], [3, 4]], "m")
    q2 = u.Q([[5, 6], [7, 8]], "s")
    result = q1 @ q2
    assert np.array_equal(result.value, [[19, 22], [43, 50]])
    assert result.unit == u.unit("m s")

    # Dimensionless matmul
    q = u.Q([[1, 2], [3, 4]], "m")
    arr = jnp.array([5, 6])
    result = q @ arr
    assert np.array_equal(result.value, [17, 39])
    assert result.unit == u.unit("m")


def test_rmatmul():
    """Test the ``Quantity.__rmatmul__`` method."""
    # Array @ Quantity
    arr = jnp.array([1, 2, 3])
    q = u.Q([4, 5, 6], "m")
    result = arr @ q
    assert result.value == 32  # 1*4 + 2*5 + 3*6
    assert result.unit == u.unit("m")

    # Matrix @ Quantity
    arr = jnp.array([[1, 2], [3, 4]])
    q = u.Q([5, 6], "kg")
    result = arr @ q
    assert np.array_equal(result.value, [17, 39])
    assert result.unit == u.unit("kg")


def test_pow():
    """Test the ``Quantity.__pow__" method."""
    # Scalar power with integer
    q = u.Q(2.0, "m")
    result = q**2
    assert jnp.isclose(result.value, 4.0)
    assert result.unit == u.unit("m2")

    # Array power
    q = u.Q([1.0, 2.0, 3.0], "m")
    result = q**2
    assert np.allclose(result.value, jnp.array([1.0, 4.0, 9.0]))
    assert result.unit == u.unit("m2")

    # Negative power
    q = u.Q(2.0, "m")
    result = q**-1
    assert jnp.isclose(result.value, 0.5)
    assert result.unit == u.unit("1/m")

    # Zero power (dimensionless result)
    q = u.Q(5.0, "m")
    result = q**0
    assert jnp.isclose(result.value, 1.0)
    assert u.dimension_of(result) == u.dimension("dimensionless")


def test_rpow():
    """Test the ``Quantity.__rpow__`` method."""
    # Scalar base with dimensionless Quantity exponent
    q = u.Q(2.0, "")  # dimensionless
    result = 3.0**q
    assert jnp.isclose(result.value, 9.0)
    assert result.unit == u.unit("")

    # Exponent must be dimensionless
    q = u.Q(2.0, "m")
    with pytest.raises(Exception):  # noqa: B017, PT011
        _ = 3.0**q


def test_truediv():
    """Test the ``Quantity.__truediv__`` method."""
    # Scalar division
    q1 = u.Q(6, "m")
    q2 = u.Q(2, "s")
    result = q1 / q2
    assert result.value == 3
    assert result.unit == u.unit("m/s")

    # Division by dimensionless scalar
    q = u.Q(6, "m")
    result = q / 2
    assert result.value == 3
    assert result.unit == u.unit("m")

    # Array division
    q1 = u.Q([10, 20, 30], "m")
    q2 = u.Q([2, 4, 5], "s")
    result = q1 / q2
    assert np.array_equal(result.value, [5, 5, 6])
    assert result.unit == u.unit("m/s")

    # Division creates compound units
    distance = u.Q(100, "m")
    time = u.Q(10, "s")
    velocity = distance / time
    assert velocity.value == 10
    assert u.dimension_of(velocity) == u.dimension("velocity")

    # Same-dimension division produces dimensionless
    q1 = u.Q(10, "m")
    q2 = u.Q(2, "m")
    result = q1 / q2
    assert result.value == 5
    assert u.dimension_of(result) == u.dimension("dimensionless")


def test_rtruediv():
    """Test the ``Quantity.__rtruediv__`` method."""
    # Dimensionless scalar / Quantity
    q = u.Q(2, "m")
    result = 10 / q
    assert result.value == 5
    assert result.unit == u.unit("1/m")

    # Array / Quantity
    arr = jnp.array([10, 20, 30])
    q = u.Q(2, "s")
    result = arr / q
    assert np.array_equal(result.value, [5, 10, 15])
    assert result.unit == u.unit("1/s")

    # Special case: 1 / Quantity
    q = u.Q(4, "m")
    result = 1 / q
    assert result.value == 0.25
    assert result.unit == u.unit("1/m")


def test_and():
    """Test the ``Quantity.__and__`` method."""
    # Bitwise AND for integer dimensionless Quantities
    q1 = u.Q(jnp.array([1, 2, 3], dtype=jnp.int32), "")
    q2 = u.Q(jnp.array([1, 0, 3], dtype=jnp.int32), "")

    # Bitwise AND works for dimensionless integers and returns plain array
    result = q1 & q2
    assert np.array_equal(result, jnp.array([1, 0, 3], dtype=jnp.int32))

    # Should fail with dimensioned quantities
    q1 = u.Q(jnp.array([1, 2, 3], dtype=jnp.int32), "m")
    q2 = u.Q(jnp.array([1, 0, 3], dtype=jnp.int32), "m")
    with pytest.raises(Exception):  # noqa: B017, PT011
        _ = q1 & q2


def test_gt():
    """Test the ``Quantity.__gt__`` method."""
    # Test with a scalar
    q = u.Q(1, "m")
    assert q > u.Q(0, "m")
    assert not q > u.Q(1, "m")
    assert not q > u.Q(2, "m")

    # Test with an array
    q = u.Q([1, 2, 3], "m")
    assert np.array_equal(q > u.Q(0, "m"), [True, True, True])
    assert np.array_equal(q > u.Q(1, "m"), [False, True, True])

    # Test with incompatible units
    # TODO: better equinox exception matching
    with pytest.raises(Exception):  # noqa: B017, PT011
        _ = q > u.Q(0, "s")

    # Test special case w/out units
    assert u.Q(1, "m") > 0
    assert np.array_equal(u.Q([-1, 0, 1], "m") > 0, [False, False, True])


def test_ge():
    """Test the ``Quantity.__ge__`` method."""
    # Test with a scalar
    q = u.Q(1, "m")
    assert q >= u.Q(0, "m")
    assert q >= u.Q(1, "m")
    assert not q >= u.Q(2, "m")

    # Test with an array
    q = u.Q([1, 2, 3], "m")
    assert np.array_equal(q >= u.Q(0, "m"), [True, True, True])
    assert np.array_equal(q >= u.Q(1, "m"), [True, True, True])
    assert np.array_equal(q >= u.Q(2, "m"), [False, True, True])

    # Test with incompatible units
    # TODO: better equinox exception matching
    with pytest.raises(Exception):  # noqa: B017, PT011
        _ = q >= u.Q(0, "s")

    # Test special case w/out units
    assert u.Q(1, "m") >= 0
    assert np.array_equal(u.Q([-1, 0, 1], "m") >= 0, [False, True, True])


def test_lt():
    """Test the ``Quantity.__lt__`` method."""
    # Test with a scalar
    q = u.Q(1, "m")
    assert not q < u.Q(0, "m")
    assert not q < u.Q(1, "m")
    assert q < u.Q(2, "m")

    # Test with an array
    q = u.Q([1, 2, 3], "m")
    assert np.array_equal(q < u.Q(0, "m"), [False, False, False])
    assert np.array_equal(q < u.Q(1, "m"), [False, False, False])
    assert np.array_equal(q < u.Q(2, "m"), [True, False, False])

    # Test with incompatible units
    # TODO: better equinox exception matching
    with pytest.raises(Exception):  # noqa: B017, PT011
        _ = q < u.Q(0, "s")

    # Test special case w/out units
    assert u.Q(-1, "m") < 0
    assert np.array_equal(u.Q([-1, 0, 1], "m") < 0, [True, False, False])


def test_le():
    """Test the ``u.Q.__le__`` method."""
    # Test with a scalar
    q = u.Q(1, "m")
    assert not q <= u.Q(0, "m")
    assert q <= u.Q(1, "m")
    assert q <= u.Q(2, "m")

    # Test with an array
    q = u.Q([1, 2, 3], "m")
    assert np.array_equal(q <= u.Q(0, "m"), [False, False, False])
    assert np.array_equal(q <= u.Q(1, "m"), [True, False, False])
    assert np.array_equal(q <= u.Q(2, "m"), [True, True, False])

    # Test with incompatible units
    # TODO: better equinox exception matching
    with pytest.raises(Exception):  # noqa: B017, PT011
        _ = q <= u.Q(0, "s")

    # Test special case w/out units
    assert u.Q(-1, "m") <= 0
    assert np.array_equal(u.Q([-1, 0, 1], "m") <= 0, [True, True, False])


def test_eq():
    """Test the ``Quantity.__eq__`` method."""
    # Test with a scalar
    q = u.Q(1, "m")
    assert not q == u.Q(0, "m")  # noqa: SIM201
    assert q == u.Q(1, "m")
    assert not q == u.Q(2, "m")  # noqa: SIM201

    # Test with an array
    q = u.Q([1, 2, 3], "m")
    assert np.array_equal(q == u.Q(0, "m"), [False, False, False])
    assert np.array_equal(q == u.Q(1, "m"), [True, False, False])
    assert np.array_equal(q == u.Q(2, "m"), [False, True, False])

    # Test with incompatible units
    with pytest.raises(Exception):  # noqa: B017, PT011
        _ = jnp.equal(q, u.Q(0, "s"))

    # Test special case w/out units
    assert u.Q(0, "m") == 0
    assert np.array_equal(u.Q([-1, 0, 1], "m") == 0, [False, True, False])


def test_ne():
    """Test the ``Quantity.__ne__`` method."""
    # Test with a scalar
    q = u.Q(1, "m")
    assert q != u.Q(0, "m")
    assert q == u.Q(1, "m")
    assert q != u.Q(2, "m")

    # Test with an array
    q = u.Q([1, 2, 3], "m")
    assert np.array_equal(q != u.Q(0, "m"), [True, True, True])
    assert np.array_equal(q != u.Q(1, "m"), [False, True, True])
    assert np.array_equal(q != u.Q(2, "m"), [True, False, True])

    # Test with incompatible units
    # TODO: better equinox exception matching
    with pytest.raises(Exception):  # noqa: B017, PT011
        _ = jnp.not_equal(q, u.Q(0, "s"))
    with pytest.raises(Exception):  # noqa: B017, PT011
        _ = jnp.not_equal(q, u.Q(4, "s"))

    # Test special case w/out units
    assert u.Q(1, "m") != 0
    assert np.array_equal(u.Q([-1, 0, 1], "m") != 0, [True, False, True])


def test_neg():
    """Test the ``Quantity.__neg__`` method."""
    # Test with a scalar
    q = u.Q(1, "m")
    assert -q == u.Q(-1, "m")

    # Test with an array
    q = u.Q([1, 2, 3], "m")
    assert np.array_equal(-q.value, [-1, -2, -3])
    assert (-q).unit == u.unit("m")


def test_flatten():
    """Test the ``Quantity.flatten`` method."""
    # Test with a scalar
    q = u.Q(1, "m")
    assert q.flatten() == u.Q(1, "m")

    # Test with an array
    q = u.Q([[1, 2, 3], [4, 5, 6]], "m")
    assert np.array_equal(q.flatten().value, [1, 2, 3, 4, 5, 6])
    assert q.flatten().unit == u.unit("m")


def test_reshape():
    """Test the ``Quantity.reshape`` method."""
    # Test with a scalar
    q = u.Q(1, "m")
    assert q.reshape(1, 1) == u.Q(1, "m")

    # Test with an array
    q = u.Q([1, 2, 3, 4, 5, 6], "m")
    assert np.array_equal(q.reshape(2, 3).value, [[1, 2, 3], [4, 5, 6]])
    assert q.reshape(2, 3).unit == u.unit("m")


def test_hypot():
    """Test the ``jnp.hypot`` method."""
    q1 = u.Q(3, "m")
    q2 = u.Q(4, "m")
    assert jnp.hypot(q1, q2) == u.Q(5, "m")

    q1 = u.Q([1, 2, 3], "m")
    q2 = u.Q([4, 5, 6], "m")
    assert all(jnp.hypot(q1, q2) == u.Q([4.1231055, 5.3851647, 6.7082043], "m"))


def test_mod():
    """Test taking the modulus."""
    q = u.Q(480.0, "deg")

    with pytest.raises(RuntimeError):
        _ = q % 2

    with pytest.raises(apyu.UnitConversionError):
        _ = q % u.Q(2, "m")

    got = q % u.Q(360, "deg")
    expect = u.Q(120, "deg")
    assert got == expect


# --------------------------------------------------------------


def test_at():
    """Test the ``Quantity.at`` method."""
    x = jnp.arange(10, dtype=float)
    q = u.Q(x, "km")

    # Get
    # TODO: test fill_value
    assert q.at[1].get() == u.Q(1.0, "km")
    assert np.array_equal(q.at[:3].get(), u.Q([0.0, 1, 2], "km"))

    # Set
    q2 = q.at[1].set(u.Q(1.2, "km"))
    assert q2[1] == u.Q(1.2, "km")
    assert q[1] == u.Q(1.0, "km")  # original is unchanged

    q2 = q.at[:3].set(u.Q([1.2, 2.3, 3.4], "km"))
    assert np.array_equal(q2[:3], u.Q([1.2, 2.3, 3.4], "km"))
    assert np.array_equal(q[:3], u.Q([0.0, 1, 2], "km"))  # original is unchanged

    # Apply
    with pytest.raises(NotImplementedError):
        q.at[1].apply(lambda x: x + 1)

    # Add
    q2 = q.at[1].add(u.Q(1.2, "km"))
    assert q2[1] == u.Q(2.2, "km")
    assert q[1] == u.Q(1.0, "km")  # original is unchanged

    # Multiply
    q2 = q.at[1].mul(2)
    assert q2[1] == u.Q(2.0, "km")
    assert q[1] == u.Q(1.0, "km")  # original is unchanged

    with pytest.raises((RuntimeError, TypeCheckError)):
        q.at[1].mul(u.Q(2, "m"))

    # Divide
    q2 = q.at[1].divide(2)
    assert q2[1] == u.Q(0.5, "km")
    assert q[1] == u.Q(1.0, "km")  # original is unchanged

    with pytest.raises((RuntimeError, TypeCheckError)):
        q.at[1].divide(u.Q(2, "m"))

    # Power
    with pytest.raises(NotImplementedError):
        q.at[1].power(2)

    # Min
    q2 = q.at[1].min(u.Q(0.5, "km"))
    assert q2[1] == u.Q(0.5, "km")
    assert q[1] == u.Q(1.0, "km")  # original is unchanged

    q2 = q.at[1].min(u.Q(1.5, "km"))
    assert q2[1] == u.Q(1.0, "km")
    assert q[1] == u.Q(1.0, "km")  # original is unchanged

    # Max
    q2 = q.at[1].max(u.Q(1.5, "km"))
    assert q2[1] == u.Q(1.5, "km")
    assert q[1] == u.Q(1.0, "km")  # original is unchanged

    q2 = q.at[1].max(u.Q(0.5, "km"))
    assert q2[1] == u.Q(1.0, "km")
    assert q[1] == u.Q(1.0, "km")  # original is unchanged


# ---------------------------


@parametric
class NewQuantity(AbstractParametricQuantity):
    """Quantity with a flag."""

    value: Array = eqx.field(converter=jnp.asarray)
    unit: str = eqx.field(converter=u.unit)
    flag: bool = eqx.field(static=True, kw_only=True)


def test_parametric_pickle_dumps_with_kw_fields():
    x = NewQuantity([1, 2, 3], "m", flag=True)
    assert isinstance(pickle.dumps(x), bytes)


# ===============================================================
# Astropy


def test_from_astropy():
    """Test the ``Quantity.from_(AstropyQuantity)`` method."""
    apyq = apyu.Quantity(1, "m")
    q = u.Q.from_(apyq)
    assert isinstance(q, u.Q)
    assert np.equal(q.value, apyq.value)
    assert q.unit == apyq.unit


def test_convert_to_astropy():
    """Test the ``convert(Quantity, AstropyQuantity)`` method."""
    q = u.Q(1, "m")
    apyq = convert(q, apyu.Quantity)
    assert isinstance(apyq, apyu.Quantity)
    assert apyq == apyu.Quantity(1, "m")


##############################################################################


def test_is_unit_convertible():
    """Test `unxt.is_unit_convertible`."""
    # Unit
    assert u.is_unit_convertible(apyu.km, apyu.kpc) is True

    # unit is a str
    assert u.is_unit_convertible("km", "m") is True

    # Bad unit
    assert u.is_unit_convertible(apyu.s, apyu.m) is False

    # Quantity
    assert u.is_unit_convertible(apyu.kpc, u.Q(1, "km")) is True

    # unit is a str
    assert u.is_unit_convertible("km", u.Q(1, "km")) is True

    # Bad quantity
    assert u.is_unit_convertible(apyu.m, u.Q(1, "s")) is False


##############################################################################
# Property-based tests using unxt-hypothesis


class TestQuantityProperties:
    """Property-based tests for Quantity using hypothesis strategies."""

    @settings(max_examples=20, deadline=None)
    @given(q=ust.quantities("m", shape=st.integers(min_value=0, max_value=10)))
    def test_unit_preserved(self, q):
        """Test that unit is preserved across operations."""
        assert q.unit == u.unit("m")

    @settings(max_examples=20, deadline=None)
    @given(q1=ust.quantities("m"), q2=ust.quantities("m"))
    def test_addition_commutative(self, q1, q2):
        """Test that addition is commutative: q1 + q2 == q2 + q1."""
        result1 = q1 + q2
        result2 = q2 + q1
        assert jnp.allclose(result1.value, result2.value)
        assert result1.unit == result2.unit

    @settings(max_examples=20, deadline=None)
    @given(q1=ust.quantities("m"), q2=ust.quantities("m"), q3=ust.quantities("m"))
    def test_addition_associative(self, q1, q2, q3):
        """Test that addition is associative: (q1 + q2) + q3 == q1 + (q2 + q3)."""
        result1 = (q1 + q2) + q3
        result2 = q1 + (q2 + q3)
        assert jnp.allclose(result1.value, result2.value)
        assert result1.unit == result2.unit

    @settings(max_examples=20, deadline=None)
    @given(q=ust.quantities(""))  # Use dimensionless
    def test_additive_identity(self, q):
        """Test that 0 is the additive identity for dimensionless: q + 0 == q."""
        result = q + 0
        assert jnp.allclose(result.value, q.value)
        assert result.unit == q.unit

    @settings(max_examples=20, deadline=None)
    @given(q=ust.quantities("m"))
    def test_additive_inverse(self, q):
        """Test additive inverse: q + (-q) == 0."""
        result = q + (-q)
        assert jnp.allclose(result.value, 0.0, atol=1e-6)

    @settings(max_examples=20, deadline=None)
    @given(q1=ust.quantities("m"), q2=ust.quantities("s"))
    def test_multiplication_creates_compound_units(self, q1, q2):
        """Test that multiplying different units creates compound units."""
        result = q1 * q2
        assert jnp.allclose(result.value, q1.value * q2.value)
        # Verify it's a compound unit with both dimensions
        assert result.unit == u.unit("m s")

    @settings(max_examples=20, deadline=None)
    @given(q1=ust.quantities("m"), q2=ust.quantities("m"))
    def test_division_same_units_dimensionless(self, q1, q2):
        """Test that dividing same units gives dimensionless result."""
        # Avoid division by zero
        if jnp.any(q2.value == 0):
            return
        result = q1 / q2
        # Result should be dimensionless
        assert u.dimension_of(result) == u.dimension("dimensionless")

    @given(
        q=ust.quantities(
            "m",
            shape=st.integers(min_value=1, max_value=5),
            elements=st.floats(min_value=-100, max_value=100, width=32),
        )
    )
    def test_negation_involutive(self, q):
        """Test that negation is involutive: -(-q) == q."""
        neg = -q
        result = -neg
        assert jnp.allclose(result.value, q.value)
        assert result.unit == q.unit

    @settings(max_examples=20, deadline=None)
    @given(
        q=ust.quantities("m", elements=st.floats(min_value=1, max_value=100, width=32))
    )
    def test_power_identity(self, q):
        """Test that q^1 == q."""
        result = q**1
        assert jnp.allclose(result.value, q.value)
        assert result.unit == q.unit

    @settings(max_examples=20, deadline=None)
    @given(
        q=ust.quantities(
            "m",
            elements=st.floats(min_value=0.10000000149011612, max_value=10, width=32),
        )
    )
    def test_power_zero(self, q):
        """Test that q^0 == 1 (dimensionless)."""
        result = q**0
        assert jnp.allclose(result.value, 1.0)
        assert u.dimension_of(result) == u.dimension("dimensionless")

    @given(
        q=ust.quantities(
            "m",
            shape=st.integers(min_value=0, max_value=10),
            elements=st.floats(min_value=-100, max_value=100, width=32),
        )
    )
    def test_reshape_preserves_values(self, q):
        """Test that reshape preserves all values."""
        if q.size == 0:
            return
        flat = q.flatten()
        assert jnp.allclose(jnp.sort(flat.value), jnp.sort(q.flatten().value))

    @settings(max_examples=20, deadline=None)
    @given(q1=ust.quantities("m"), q2=ust.quantities("km"))
    def test_compatible_unit_comparison(self, q1, q2):
        """Test comparisons work with compatible units."""
        # Just verify they can be compared without error
        try:
            _ = q1 > q2.uconvert("m")
            _ = q1 < q2.uconvert("m")
            _ = q1 >= q2.uconvert("m")
            _ = q1 <= q2.uconvert("m")
        except Exception as e:  # noqa: BLE001
            pytest.fail(f"Compatible unit comparison failed: {e}")

    @given(
        q=ust.quantities(
            u.dimension("length"),  # Use dimension to get varied length units
            shape=3,
            elements=st.floats(min_value=0.10000000149011612, max_value=100, width=32),
        )
    )
    def test_dimension_preserved_across_units(self, q):
        """Test that physical dimension is preserved regardless of unit."""
        assert u.dimension_of(q) == u.dimension("length")

    @given(
        length=ust.quantities(
            "m", elements=st.floats(min_value=0, max_value=100, width=32)
        ),
        time=ust.quantities(
            "s",
            elements=st.floats(min_value=0.10000000149011612, max_value=10, width=32),
        ),
    )
    def test_velocity_from_length_time(self, length, time):
        """Test that length/time gives velocity dimension."""
        velocity = length / time
        assert u.dimension_of(velocity) == u.dimension("velocity")

    @given(
        mass=ust.quantities(
            "kg", elements=st.floats(min_value=0, max_value=100, width=32)
        ),
        velocity=ust.quantities(
            "m/s", elements=st.floats(min_value=0, max_value=100, width=32)
        ),
    )
    def test_momentum_from_mass_velocity(self, mass, velocity):
        """Test that mass * velocity gives momentum dimension."""
        momentum = mass * velocity
        # Momentum has dimensions of mass * length / time
        expected_dim = u.dimension("mass") * u.dimension("length") / u.dimension("time")
        assert u.dimension_of(momentum) == expected_dim

    @given(
        q=ust.quantities(
            st.sampled_from(["m", "km", "pc", "kpc"]),
            shape=st.integers(min_value=0, max_value=5),
            elements=st.floats(min_value=0.10000000149011612, max_value=1000, width=32),
        )
    )
    def test_various_length_units(self, q):
        """Test quantities with various length units."""
        assert u.dimension_of(q) == u.dimension("length")
        # Verify conversion works
        q_m = q.uconvert("m")
        assert q_m.unit == u.unit("m")

    @given(
        q=ust.quantities(
            "rad",
            quantity_cls=u.Angle,
            elements=st.floats(
                min_value=0, max_value=6.2831854820251465, width=32
            ),  # 2*pi in float32
        )
    )
    def test_angle_creation(self, q):
        """Test creating Angle instances with hypothesis."""
        assert isinstance(q, u.Angle)
        assert u.dimension_of(q) == u.dimension("angle")

    @settings(max_examples=50, deadline=None)
    @given(
        q1=ust.quantities(
            "m", shape=3, elements=st.floats(min_value=-10, max_value=10, width=32)
        ),
        q2=ust.quantities(
            "m", shape=3, elements=st.floats(min_value=-10, max_value=10, width=32)
        ),
    )
    def test_dot_product_properties(self, q1, q2):
        """Test dot product (matmul) properties."""
        # Dot product should be commutative for vectors
        result1 = q1 @ q2
        result2 = q2 @ q1
        assert jnp.isclose(result1.value, result2.value)

    @given(
        q=ust.quantities(
            "m",
            shape=st.tuples(
                st.integers(min_value=1, max_value=5),
                st.integers(min_value=1, max_value=5),
            ),
            elements=st.floats(min_value=-10, max_value=10, width=32),
        )
    )
    def test_flatten_and_reshape(self, q):
        """Test that flatten and reshape are inverses."""
        original_shape = q.shape
        flat = q.flatten()
        reshaped = flat.reshape(*original_shape)
        assert reshaped.shape == original_shape
        assert jnp.allclose(reshaped.value, q.value)

    @given(
        base=ust.quantities(
            "",
            shape=(),
            elements=st.floats(min_value=0.10000000149011612, max_value=10, width=32),
        ),
        exponent=st.integers(min_value=-3, max_value=3),
    )
    def test_power_laws(self, base, exponent):
        """Test power law properties: (q^a)^b == q^(a*b)."""
        if exponent == 0:
            return  # Skip trivial case
        result1 = (base**exponent) ** 2
        result2 = base ** (exponent * 2)
        assert jnp.isclose(result1.value, result2.value, rtol=1e-5)


##############################################################################
# Usage examples as tests


class TestQuantityUsageExamples:
    """Usage examples showing how to use Quantity primitives."""

    def test_position_vector(self):
        """Example: Working with 3D position vectors."""
        # Create a position in 3D space
        position = u.Q([1.0, 2.0, 3.0], "kpc")

        # Convert to different units
        position_pc = position.uconvert("pc")
        assert jnp.allclose(position_pc.value, jnp.array([1000.0, 2000.0, 3000.0]))

        # Calculate distance from origin (using norm)
        distance_squared = jnp.sum(position**2)
        assert distance_squared.unit == u.unit("kpc2")
        distance = jnp.sqrt(distance_squared)
        assert jnp.isclose(distance.value, jnp.sqrt(14.0), rtol=1e-5)

    def test_velocity_calculation(self):
        """Example: Calculate velocity from displacement and time."""
        displacement = u.Q(100.0, "km")
        time = u.Q(2.0, "hr")

        velocity = displacement / time
        assert velocity.unit == u.unit("km / hr")

        # Convert to m/s
        velocity_si = velocity.uconvert("m/s")
        assert jnp.isclose(velocity_si.value, 13.888889, rtol=1e-5)

    def test_kinetic_energy(self):
        """Example: Calculate kinetic energy KE = 0.5 * m * v^2."""
        mass = u.Q(1000.0, "kg")
        velocity = u.Q(20.0, "m/s")

        kinetic_energy = 0.5 * mass * velocity**2
        assert jnp.isclose(kinetic_energy.value, 200000.0)

        # Check dimensions
        expected_dim = (
            u.dimension("mass") * u.dimension("length") ** 2 / u.dimension("time") ** 2
        )
        assert u.dimension_of(kinetic_energy) == expected_dim

    def test_force_calculation(self):
        """Example: Calculate force F = m * a."""
        mass = u.Q(5.0, "kg")
        acceleration = u.Q(10.0, "m/s2")

        force = mass * acceleration
        assert jnp.isclose(force.value, 50.0)

        # Verify force dimension
        expected_dim = (
            u.dimension("mass") * u.dimension("length") / u.dimension("time") ** 2
        )
        assert u.dimension_of(force) == expected_dim

    def test_work_done(self):
        """Example: Calculate work W = F · d (dot product)."""
        force = u.Q([10.0, 0.0, 0.0], "N")
        displacement = u.Q([5.0, 3.0, 0.0], "m")

        work = force @ displacement  # Dot product
        assert jnp.isclose(work.value, 50.0)

        # Work has units of force * distance
        assert work.unit == u.unit("N m")

    def test_array_operations(self):
        """Example: Batch operations on arrays of quantities."""
        # Array of distances
        distances = u.Q([1.0, 2.0, 3.0, 4.0, 5.0], "km")

        # Filter distances greater than 2.5 km
        mask = distances > u.Q(2.5, "km")
        filtered = distances[mask]
        assert np.array_equal(filtered.value, [3.0, 4.0, 5.0])

        # Square all distances
        areas = distances**2
        assert areas.unit == u.unit("km2")
        assert np.array_equal(areas.value, [1.0, 4.0, 9.0, 16.0, 25.0])

    def test_unit_conversion_chains(self):
        """Example: Chaining unit conversions."""
        # Start with parsecs
        distance = u.Q(10.0, "pc")

        # Convert through multiple units
        distance_kpc = distance.uconvert("kpc")
        assert jnp.isclose(distance_kpc.value, 0.01)

        distance_m = distance.uconvert("m")
        assert jnp.isclose(distance_m.value, 3.0856776e17, rtol=1e-5)

        # All represent the same physical quantity
        assert u.dimension_of(distance) == u.dimension("length")
        assert u.dimension_of(distance_kpc) == u.dimension("length")
        assert u.dimension_of(distance_m) == u.dimension("length")

    def test_matrix_operations(self):
        """Example: Matrix operations with quantities."""
        # Rotation matrix (dimensionless)
        theta = np.pi / 4  # 45 degrees
        rotation = jnp.array(
            [[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]]
        )

        # Position vector
        position = u.Q([1.0, 0.0], "m")

        # Rotate position
        rotated = rotation @ position
        expected = jnp.array([1 / jnp.sqrt(2), 1 / jnp.sqrt(2)])
        assert jnp.allclose(rotated.value, expected)
        assert rotated.unit == u.unit("m")

    def test_broadcasting(self):
        """Example: Broadcasting with quantities."""
        # Single time point
        time = u.Q(5.0, "s")

        # Multiple velocities
        velocities = u.Q([10.0, 20.0, 30.0], "m/s")

        # Calculate distances (broadcasts time across velocities)
        distances = velocities * time
        assert np.array_equal(distances.value, [50.0, 100.0, 150.0])
        assert distances.unit == u.unit("m s / s")  # Simplifies to m

    def test_aggregation_operations(self):
        """Example: Aggregation operations on quantity arrays."""
        values = u.Q([1.0, 2.0, 3.0, 4.0, 5.0], "kg")

        # Sum
        total = jnp.sum(values)
        assert jnp.isclose(total.value, 15.0)
        assert total.unit == u.unit("kg")

        # Mean
        mean = jnp.mean(values)
        assert jnp.isclose(mean.value, 3.0)
        assert mean.unit == u.unit("kg")

        # Max/Min
        max_val = jnp.max(values)
        min_val = jnp.min(values)
        assert max_val.value == 5.0
        assert min_val.value == 1.0
