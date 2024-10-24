# pylint: disable=import-error, too-many-lines

"""Test the Array API."""

import pickle

import astropy.units as u
import equinox as eqx
import jax
import jax.numpy as jax_xp
import numpy as np
import pytest
from hypothesis import example, given, strategies as st
from hypothesis.extra.array_api import make_strategies_namespace
from hypothesis.extra.numpy import array_shapes as np_array_shapes, arrays as np_arrays
from jax.dtypes import canonicalize_dtype
from jaxtyping import TypeCheckError
from plum import convert, parametric

import quaxed.numpy as jnp

from unxt import AbstractParametricQuantity, Quantity, is_unit_convertible, units

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
    q = Quantity(value, "m")
    expected = jnp.asarray(value)

    # Test the value
    assert np.array_equal(q.value, expected)

    # Test the shape
    assert q.shape == expected.shape

    # Test materialise
    with pytest.raises(RuntimeError):
        q.materialise()

    # Test aval
    assert q.aval() == jax.core.get_aval(expected)

    # Test enable_materialise
    assert np.array_equal(q.enable_materialise().value, q.value)


def test_parametric():
    """Test the parametric strategy."""
    # Inferred
    q = Quantity(1, "m")
    (dimensions,) = q._type_parameter
    assert dimensions == u.get_physical_type(u.m)

    # Explicit
    q = Quantity["length"](1, "m")
    (dimensions,) = q._type_parameter
    assert dimensions == u.get_physical_type(u.m)

    q = Quantity["length"](jnp.ones((1, 2)), "m")
    (dimensions,) = q._type_parameter
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
    assert Quantity(1, "m").__array_namespace__() is jnp


def test_to_units():
    """Test the ``Quantity.to_units`` method."""
    q = Quantity(1, "m")
    assert jnp.equal(q.to_units("km"), Quantity(0.001, "km"))


def test_to_units_value():
    """Test the ``Quantity.to_units_value`` method."""
    q = Quantity(1, "m")
    assert q.to_units_value("km") == Quantity(0.001, "km").value


@pytest.mark.skip("TODO")
def test_getitem():
    """Test the ``Quantity.__getitem__`` method."""
    raise NotImplementedError


def test_len():
    """Test the ``len(Quantity)`` method."""
    q = Quantity([1, 2, 3], "m")
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
    q = Quantity(1, "m")
    assert q > Quantity(0, "m")
    assert not q > Quantity(1, "m")
    assert not q > Quantity(2, "m")

    # Test with an array
    q = Quantity([1, 2, 3], "m")
    assert np.array_equal(q > Quantity(0, "m"), [True, True, True])
    assert np.array_equal(q > Quantity(1, "m"), [False, True, True])

    # Test with incompatible units
    assert np.array_equal(q > Quantity(0, "s"), [False, False, False])


def test_ge():
    """Test the ``Quantity.__ge__`` method."""
    # Test with a scalar
    q = Quantity(1, "m")
    assert q >= Quantity(0, "m")
    assert q >= Quantity(1, "m")
    assert not q >= Quantity(2, "m")

    # Test with an array
    q = Quantity([1, 2, 3], "m")
    assert np.array_equal(q >= Quantity(0, "m"), [True, True, True])
    assert np.array_equal(q >= Quantity(1, "m"), [True, True, True])
    assert np.array_equal(q >= Quantity(2, "m"), [False, True, True])

    # Test with incompatible units
    assert np.array_equal(q >= Quantity(0, "s"), [False, False, False])


def test_lt():
    """Test the ``Quantity.__lt__`` method."""
    # Test with a scalar
    q = Quantity(1, "m")
    assert not q < Quantity(0, "m")
    assert not q < Quantity(1, "m")
    assert q < Quantity(2, "m")

    # Test with an array
    q = Quantity([1, 2, 3], "m")
    assert np.array_equal(q < Quantity(0, "m"), [False, False, False])
    assert np.array_equal(q < Quantity(1, "m"), [False, False, False])
    assert np.array_equal(q < Quantity(2, "m"), [True, False, False])

    # Test with incompatible units
    assert np.array_equal(q < Quantity(0, "s"), [False, False, False])


def test_le():
    """Test the ``Quantity.__le__`` method."""
    # Test with a scalar
    q = Quantity(1, "m")
    assert not q <= Quantity(0, "m")
    assert q <= Quantity(1, "m")
    assert q <= Quantity(2, "m")

    # Test with an array
    q = Quantity([1, 2, 3], "m")
    assert np.array_equal(q <= Quantity(0, "m"), [False, False, False])
    assert np.array_equal(q <= Quantity(1, "m"), [True, False, False])
    assert np.array_equal(q <= Quantity(2, "m"), [True, True, False])

    # Test with incompatible units
    assert np.array_equal(q <= Quantity(0, "s"), [False, False, False])


def test_eq():
    """Test the ``Quantity.__eq__`` method."""
    # Test with a scalar
    q = Quantity(1, "m")
    assert not q == Quantity(0, "m")  # noqa: SIM201
    assert q == Quantity(1, "m")
    assert not q == Quantity(2, "m")  # noqa: SIM201

    # Test with an array
    q = Quantity([1, 2, 3], "m")
    assert np.array_equal(q == Quantity(0, "m"), [False, False, False])
    assert np.array_equal(q == Quantity(1, "m"), [True, False, False])
    assert np.array_equal(q == Quantity(2, "m"), [False, True, False])

    # Test with incompatible units
    assert np.array_equal(q == Quantity(0, "s"), [False, False, False])


def test_ne():
    """Test the ``Quantity.__ne__`` method."""
    # Test with a scalar
    q = Quantity(1, "m")
    assert q != Quantity(0, "m")
    assert q == Quantity(1, "m")
    assert q != Quantity(2, "m")

    # Test with an array
    q = Quantity([1, 2, 3], "m")
    assert np.array_equal(q != Quantity(0, "m"), [True, True, True])
    assert np.array_equal(q != Quantity(1, "m"), [False, True, True])
    assert np.array_equal(q != Quantity(2, "m"), [True, False, True])

    # Test with incompatible units
    assert np.array_equal(q != Quantity(0, "s"), [True, True, True])
    assert np.array_equal(q != Quantity(4, "s"), [True, True, True])


def test_neg():
    """Test the ``Quantity.__neg__`` method."""
    # Test with a scalar
    q = Quantity(1, "m")
    assert -q == Quantity(-1, "m")

    # Test with an array
    q = Quantity([1, 2, 3], "m")
    assert np.array_equal(-q.value, [-1, -2, -3])
    assert (-q).unit == units("m")


def test_flatten():
    """Test the ``Quantity.flatten`` method."""
    # Test with a scalar
    q = Quantity(1, "m")
    assert q.flatten() == Quantity(1, "m")

    # Test with an array
    q = Quantity([[1, 2, 3], [4, 5, 6]], u.m)
    assert np.array_equal(q.flatten().value, [1, 2, 3, 4, 5, 6])
    assert q.flatten().unit == u.m


def test_reshape():
    """Test the ``Quantity.reshape`` method."""
    # Test with a scalar
    q = Quantity(1, "m")
    assert q.reshape(1, 1) == Quantity(1, "m")

    # Test with an array
    q = Quantity([1, 2, 3, 4, 5, 6], "m")
    assert np.array_equal(q.reshape(2, 3).value, [[1, 2, 3], [4, 5, 6]])
    assert q.reshape(2, 3).unit == u.m


def test_hypot():
    """Test the ``jnp.hypot`` method."""
    q1 = Quantity(3, "m")
    q2 = Quantity(4, "m")
    assert jnp.hypot(q1, q2) == Quantity(5, "m")

    q1 = Quantity([1, 2, 3], "m")
    q2 = Quantity([4, 5, 6], "m")
    assert all(jnp.hypot(q1, q2) == Quantity([4.1231055, 5.3851647, 6.7082043], u.m))


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


# --------------------------------------------------------------


def test_at():
    """Test the ``Quantity.at`` method."""
    x = jnp.arange(10, dtype=float)
    q = Quantity(x, "km")

    # Get
    # TODO: test fill_value
    assert q.at[1].get() == Quantity(1.0, "km")
    assert np.array_equal(q.at[:3].get(), Quantity([0.0, 1, 2], "km"))

    # Set
    q2 = q.at[1].set(Quantity(1.2, "km"))
    assert q2[1] == Quantity(1.2, "km")
    assert q[1] == Quantity(1.0, "km")  # original is unchanged

    q2 = q.at[:3].set(Quantity([1.2, 2.3, 3.4], "km"))
    assert np.array_equal(q2[:3], Quantity([1.2, 2.3, 3.4], "km"))
    assert np.array_equal(q[:3], Quantity([0.0, 1, 2], "km"))  # original is unchanged

    # Apply
    with pytest.raises(NotImplementedError):
        q.at[1].apply(lambda x: x + 1)

    # Add
    q2 = q.at[1].add(Quantity(1.2, "km"))
    assert q2[1] == Quantity(2.2, "km")
    assert q[1] == Quantity(1.0, "km")  # original is unchanged

    # Multiply
    q2 = q.at[1].mul(2)
    assert q2[1] == Quantity(2.0, "km")
    assert q[1] == Quantity(1.0, "km")  # original is unchanged

    with pytest.raises((RuntimeError, TypeCheckError)):
        q.at[1].mul(Quantity(2, "m"))

    # Divide
    q2 = q.at[1].divide(2)
    assert q2[1] == Quantity(0.5, "km")
    assert q[1] == Quantity(1.0, "km")  # original is unchanged

    with pytest.raises((RuntimeError, TypeCheckError)):
        q.at[1].divide(Quantity(2, "m"))

    # Power
    with pytest.raises(NotImplementedError):
        q.at[1].power(2)

    # Min
    q2 = q.at[1].min(Quantity(0.5, "km"))
    assert q2[1] == Quantity(0.5, "km")
    assert q[1] == Quantity(1.0, "km")  # original is unchanged

    q2 = q.at[1].min(Quantity(1.5, "km"))
    assert q2[1] == Quantity(1.0, "km")
    assert q[1] == Quantity(1.0, "km")  # original is unchanged

    # Max
    q2 = q.at[1].max(Quantity(1.5, "km"))
    assert q2[1] == Quantity(1.5, "km")
    assert q[1] == Quantity(1.0, "km")  # original is unchanged

    q2 = q.at[1].max(Quantity(0.5, "km"))
    assert q2[1] == Quantity(1.0, "km")
    assert q[1] == Quantity(1.0, "km")  # original is unchanged


# ---------------------------


@parametric
class NewQuantity(AbstractParametricQuantity):
    """Quantity with a flag."""

    flag: bool = eqx.field(static=True, kw_only=True)


def test_parametric_pickle_dumps_with_kw_fields():
    x = NewQuantity([1, 2, 3], "m", flag=True)
    assert isinstance(pickle.dumps(x), bytes)


# ===============================================================
# Astropy


def test_from_astropy():
    """Test the ``Quantity.from_(AstropyQuantity)`` method."""
    apyq = u.Quantity(1, "m")
    q = Quantity.from_(apyq)
    assert isinstance(q, Quantity)
    assert np.equal(q.value, apyq.value)
    assert q.unit == apyq.unit


def test_convert_to_astropy():
    """Test the ``convert(Quantity, AstropyQuantity)`` method."""
    q = Quantity(1, "m")
    apyq = convert(q, u.Quantity)
    assert isinstance(apyq, u.Quantity)
    assert apyq == u.Quantity(1, "m")


##############################################################################


def test_is_unit_convertible():
    """Test `unxt.is_unit_convertible`."""
    # Unit
    assert is_unit_convertible(u.km, u.kpc) is True

    # unit is a str
    assert is_unit_convertible("km", "m") is True

    # Bad unit
    assert is_unit_convertible(u.s, u.m) is False

    # Quantity
    assert is_unit_convertible(u.kpc, Quantity(1, u.km)) is True

    # unit is a str
    assert is_unit_convertible("km", Quantity(1, u.km)) is True

    # Bad quantity
    assert is_unit_convertible(u.m, Quantity(1, u.s)) is False
