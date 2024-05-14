# pylint: disable=import-error, too-many-lines

"""Test the Array API."""

import astropy.units as u
import jax
import jax.experimental.array_api as jax_xp
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given, strategies as st
from hypothesis.extra.array_api import make_strategies_namespace
from hypothesis.extra.numpy import array_shapes as np_array_shapes, arrays as np_arrays
from jax.dtypes import canonicalize_dtype

import quaxed.array_api
import quaxed.numpy as qnp

from unxt import Quantity, can_convert_unit

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
    q = Quantity(value, u.m)
    expected = jnp.asarray(value)

    # Test the value
    assert jnp.array_equal(q.value, expected)

    # Test the shape
    assert q.shape == expected.shape

    # Test materialise
    with pytest.raises(RuntimeError):
        q.materialise()

    # Test aval
    assert q.aval() == jax.core.get_aval(expected)

    # Test enable_materialise
    assert jnp.array_equal(q.enable_materialise().value, q.value)


def test_parametric():
    """Test the parametric strategy."""
    # Inferred
    q = Quantity(1, u.m)
    (dimensions,) = q._type_parameter  # noqa: SLF001
    assert dimensions == u.get_physical_type(u.m)

    # Explicit
    q = Quantity["length"](1, u.m)
    (dimensions,) = q._type_parameter  # noqa: SLF001
    assert dimensions == u.get_physical_type(u.m)

    q = Quantity["length"](jnp.ones((1, 2)), u.m)
    (dimensions,) = q._type_parameter  # noqa: SLF001
    assert dimensions == u.get_physical_type(u.m)

    # type-checks
    with pytest.raises(ValueError, match="Physical type mismatch."):
        Quantity["time"](1, u.m)


@pytest.mark.parametrize("unit", [u.m, "meter"])
def test_unit(unit):
    """Test the unit."""
    assert Quantity(1, unit).unit == unit


def test_array_namespace():
    """Test the array namespace."""
    assert Quantity(1, u.m).__array_namespace__() is quaxed.array_api


def test_to_units():
    """Test the ``Quantity.to_units`` method."""
    q = Quantity(1, u.m)
    assert qnp.equal(q.to_units(u.km), Quantity(0.001, u.km))


def test_to_units_value():
    """Test the ``Quantity.to_units_value`` method."""
    q = Quantity(1, u.m)
    assert q.to_units_value(u.km) == Quantity(0.001, u.km).value


@pytest.mark.skip("TODO")
def test_getitem():
    """Test the ``Quantity.__getitem__`` method."""
    raise NotImplementedError


def test_len():
    """Test the ``len(Quantity)`` method."""
    q = Quantity([1, 2, 3], u.m)
    assert len(q) == 3


@pytest.mark.skip("TODO")
def test_add():
    """Test the ``Quantity.__add__`` method."""
    raise NotImplementedError


@pytest.mark.skip("TODO")
def test_radd():
    """Test the ``Quantity.__radd__`` method."""
    raise NotImplementedError


@pytest.mark.skip("TODO")
def test_sub():
    """Test the ``Quantity.__sub__`` method."""
    raise NotImplementedError


@pytest.mark.skip("TODO")
def test_rsub():
    """Test the ``Quantity.__rsub__`` method."""
    raise NotImplementedError


@pytest.mark.skip("TODO")
def test_mul():
    """Test the ``Quantity.__mul__`` method."""
    raise NotImplementedError


@pytest.mark.skip("TODO")
def test_rmul():
    """Test the ``Quantity.__rmul__`` method."""
    raise NotImplementedError


@pytest.mark.skip("TODO")
def test_matmul():
    """Test the ``Quantity.__matmul__`` method."""
    raise NotImplementedError


@pytest.mark.skip("TODO")
def test_rmatmul():
    """Test the ``Quantity.__rmatmul__`` method."""
    raise NotImplementedError


@pytest.mark.skip("TODO")
def test_pow():
    """Test the ``Quantity.__pow__`` method."""
    raise NotImplementedError


@pytest.mark.skip("TODO")
def test_rpow():
    """Test the ``Quantity.__rpow__`` method."""
    raise NotImplementedError


@pytest.mark.skip("TODO")
def test_truediv():
    """Test the ``Quantity.__truediv__`` method."""
    raise NotImplementedError


@pytest.mark.skip("TODO")
def test_rtruediv():
    """Test the ``Quantity.__rtruediv__`` method."""
    raise NotImplementedError


@pytest.mark.skip("TODO")
def test_and():
    """Test the ``Quantity.__and__`` method."""
    raise NotImplementedError


def test_gt():
    """Test the ``Quantity.__gt__`` method."""
    # Test with a scalar
    q = Quantity(1, u.m)
    assert q > Quantity(0, u.m)
    assert not q > Quantity(1, u.m)
    assert not q > Quantity(2, u.m)

    # Test with an array
    q = Quantity([1, 2, 3], u.m)
    assert jnp.array_equal(q > Quantity(0, u.m), jnp.array([True, True, True]))
    assert jnp.array_equal(q > Quantity(1, u.m), jnp.array([False, True, True]))

    # Test with incompatible units
    assert jnp.array_equal(q > Quantity(0, u.s), jnp.array([False, False, False]))


def test_ge():
    """Test the ``Quantity.__ge__`` method."""
    # Test with a scalar
    q = Quantity(1, u.m)
    assert q >= Quantity(0, u.m)
    assert q >= Quantity(1, u.m)
    assert not q >= Quantity(2, u.m)

    # Test with an array
    q = Quantity([1, 2, 3], u.m)
    assert jnp.array_equal(q >= Quantity(0, u.m), jnp.array([True, True, True]))
    assert jnp.array_equal(q >= Quantity(1, u.m), jnp.array([True, True, True]))
    assert jnp.array_equal(q >= Quantity(2, u.m), jnp.array([False, True, True]))

    # Test with incompatible units
    assert jnp.array_equal(q >= Quantity(0, u.s), jnp.array([False, False, False]))


def test_lt():
    """Test the ``Quantity.__lt__`` method."""
    # Test with a scalar
    q = Quantity(1, u.m)
    assert not q < Quantity(0, u.m)
    assert not q < Quantity(1, u.m)
    assert q < Quantity(2, u.m)

    # Test with an array
    q = Quantity([1, 2, 3], u.m)
    assert jnp.array_equal(q < Quantity(0, u.m), jnp.array([False, False, False]))
    assert jnp.array_equal(q < Quantity(1, u.m), jnp.array([False, False, False]))
    assert jnp.array_equal(q < Quantity(2, u.m), jnp.array([True, False, False]))

    # Test with incompatible units
    assert jnp.array_equal(q < Quantity(0, u.s), jnp.array([False, False, False]))


def test_le():
    """Test the ``Quantity.__le__`` method."""
    # Test with a scalar
    q = Quantity(1, u.m)
    assert not q <= Quantity(0, u.m)
    assert q <= Quantity(1, u.m)
    assert q <= Quantity(2, u.m)

    # Test with an array
    q = Quantity([1, 2, 3], u.m)
    assert jnp.array_equal(q <= Quantity(0, u.m), jnp.array([False, False, False]))
    assert jnp.array_equal(q <= Quantity(1, u.m), jnp.array([True, False, False]))
    assert jnp.array_equal(q <= Quantity(2, u.m), jnp.array([True, True, False]))

    # Test with incompatible units
    assert jnp.array_equal(q <= Quantity(0, u.s), jnp.array([False, False, False]))


def test_eq():
    """Test the ``Quantity.__eq__`` method."""
    # Test with a scalar
    q = Quantity(1, u.m)
    assert not q == Quantity(0, u.m)  # noqa: SIM201
    assert q == Quantity(1, u.m)
    assert not q == Quantity(2, u.m)  # noqa: SIM201

    # Test with an array
    q = Quantity([1, 2, 3], u.m)
    assert jnp.array_equal(q == Quantity(0, u.m), jnp.array([False, False, False]))
    assert jnp.array_equal(q == Quantity(1, u.m), jnp.array([True, False, False]))
    assert jnp.array_equal(q == Quantity(2, u.m), jnp.array([False, True, False]))

    # Test with incompatible units
    assert jnp.array_equal(q == Quantity(0, u.s), jnp.array([False, False, False]))


def test_ne():
    """Test the ``Quantity.__ne__`` method."""
    # Test with a scalar
    q = Quantity(1, u.m)
    assert q != Quantity(0, u.m)
    assert q == Quantity(1, u.m)
    assert q != Quantity(2, u.m)

    # Test with an array
    q = Quantity([1, 2, 3], u.m)
    assert jnp.array_equal(q != Quantity(0, u.m), jnp.array([True, True, True]))
    assert jnp.array_equal(q != Quantity(1, u.m), jnp.array([False, True, True]))
    assert jnp.array_equal(q != Quantity(2, u.m), jnp.array([True, False, True]))

    # Test with incompatible units
    assert jnp.array_equal(q != Quantity(0, u.s), jnp.array([False, False, False]))


def test_neg():
    """Test the ``Quantity.__neg__`` method."""
    # Test with a scalar
    q = Quantity(1, u.m)
    assert -q == Quantity(-1, u.m)

    # Test with an array
    q = Quantity([1, 2, 3], u.m)
    assert jnp.array_equal(-q.value, jnp.array([-1, -2, -3]))
    assert (-q).unit == u.m


def test_flatten():
    """Test the ``Quantity.flatten`` method."""
    # Test with a scalar
    q = Quantity(1, u.m)
    assert q.flatten() == Quantity(1, u.m)

    # Test with an array
    q = Quantity([[1, 2, 3], [4, 5, 6]], u.m)
    assert jnp.array_equal(q.flatten().value, jnp.array([1, 2, 3, 4, 5, 6]))
    assert q.flatten().unit == u.m


def test_reshape():
    """Test the ``Quantity.reshape`` method."""
    # Test with a scalar
    q = Quantity(1, u.m)
    assert q.reshape(1, 1) == Quantity(1, u.m)

    # Test with an array
    q = Quantity([1, 2, 3, 4, 5, 6], u.m)
    assert jnp.array_equal(q.reshape(2, 3).value, jnp.array([[1, 2, 3], [4, 5, 6]]))
    assert q.reshape(2, 3).unit == u.m


def test_hypot():
    """Test the ``jnp.hypot`` method."""
    q1 = Quantity(3, u.m)
    q2 = Quantity(4, u.m)
    assert qnp.hypot(q1, q2) == Quantity(5, u.m)

    q1 = Quantity([1, 2, 3], u.m)
    q2 = Quantity([4, 5, 6], u.m)
    assert all(qnp.hypot(q1, q2) == Quantity([4.1231055, 5.3851647, 6.7082043], u.m))


def test_mod():
    """Test taking the modulus."""
    q = Quantity(480.0, "deg")

    with pytest.raises(AttributeError):
        _ = q % 2

    with pytest.raises(u.UnitConversionError):
        _ = q % Quantity(2, "m")

    got = q % Quantity(360, "deg")
    expect = Quantity(120, "deg")
    assert got == expect


# ===============================================================
# Unknown


def test_convert_to_unknown():
    """Test the ``Quantity.convert_to`` method with an unknown format."""
    q = Quantity(1, u.m)
    with pytest.raises(TypeError, match="Unknown format <class 'int'>."):
        q.convert_to(int)


# ===============================================================
# Astropy


def test_from_astropy():
    """Test the ``Quantity.constructor(AstropyQuantity)`` method."""
    apyq = u.Quantity(1, u.m)
    q = Quantity.constructor(apyq)
    assert isinstance(q, Quantity)
    assert np.equal(q.value, apyq.value)
    assert q.unit == apyq.unit


def test_convert_to_astropy():
    """Test the ``Quantity.convert_to(AstropyQuantity)`` method."""
    q = Quantity(1, u.m)
    apyq = q.convert_to(u.Quantity)
    assert isinstance(apyq, u.Quantity)
    assert apyq == u.Quantity(1, u.m)


##############################################################################


def test_can_convert_unit():
    """Test :func:`unxt.can_convert_unit`."""
    # Unit
    assert can_convert_unit(u.km, u.kpc) is True

    # Bad unit
    assert can_convert_unit(u.s, u.m) is False

    # Quantity
    assert can_convert_unit(Quantity(1, u.km), u.kpc) is True

    # Bad quantity
    assert can_convert_unit(Quantity(1, u.s), u.m) is False
