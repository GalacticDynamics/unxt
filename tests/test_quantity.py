# pylint: disable=import-error, too-many-lines

"""Test the Array API."""

import array_api_jax_compat
import astropy.units as u
import jax
import jax.experimental.array_api as jax_xp
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given, strategies as st
from hypothesis.extra.array_api import make_strategies_namespace
from hypothesis.extra.numpy import (
    array_shapes as np_array_shapes,
    arrays as np_arrays,
)

from jax_quantity import Quantity

xps = make_strategies_namespace(jax_xp)

jax.config.update("jax_enable_x64", val=True)

integers_strategy = st.integers(
    min_value=np.iinfo(np.int64).min, max_value=np.iinfo(np.int64).max
)
floats_strategy = st.floats(
    min_value=np.finfo(np.float64).min, max_value=np.finfo(np.float64).max
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
        dtype=np.float64,
        shape=np_array_shapes(),
        elements={"allow_nan": False, "allow_infinity": False},
    )
    | xps.arrays(
        dtype=xps.floating_dtypes(),
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


@pytest.mark.parametrize("unit", [u.m, "meter"])
def test_unit(unit):
    """Test the unit."""
    assert Quantity(1, unit).unit == unit


def test_array_namespace():
    """Test the array namespace."""
    assert Quantity(1, u.m).__array_namespace__() is array_api_jax_compat


def test_to():
    """Test the ``Quantity.to`` method."""
    q = Quantity(1, u.m)
    assert q.to(u.km) == Quantity(0.001, u.km)


def test_to_value():
    """Test the ``Quantity.to`` method."""
    q = Quantity(1, u.m)
    assert q.to_value(u.km) == Quantity(0.001, u.km).value


@pytest.mark.skip("TODO")
def test_getitem():
    """Test the ``Quantity.__getitem__`` method."""
    raise NotImplementedError


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


@pytest.mark.skip("TODO")
def test_gt():
    """Test the ``Quantity.__gt__`` method."""
    raise NotImplementedError


@pytest.mark.skip("TODO")
def test_ge():
    """Test the ``Quantity.__ge__`` method."""
    raise NotImplementedError


@pytest.mark.skip("TODO")
def test_lt():
    """Test the ``Quantity.__lt__`` method."""
    raise NotImplementedError


@pytest.mark.skip("TODO")
def test_le():
    """Test the ``Quantity.__le__`` method."""
    raise NotImplementedError


@pytest.mark.skip("TODO")
def test_eq():
    """Test the ``Quantity.__eq__`` method."""
    raise NotImplementedError


@pytest.mark.skip("TODO")
def test_ne():
    """Test the ``Quantity.__ne__`` method."""
    raise NotImplementedError


@pytest.mark.skip("TODO")
def test_neg():
    """Test the ``Quantity.__neg__`` method."""
    raise NotImplementedError


# ===============================================================
# Astropy


def test_from_astropy():
    """Test the ``Quantity.constructor(AstropyQuantity)`` method."""
    apyq = u.Quantity(1, u.m)
    q = Quantity.constructor(apyq)
    assert isinstance(q, Quantity)
    assert np.equal(q.value, apyq.value)
    assert q.unit == apyq.unit


def test_to_astropy():
    """Test the ``Quantity.as_type(AstropyQuantity)`` method."""
    q = Quantity(1, u.m)
    apyq = q.as_type(u.Quantity)
    assert isinstance(apyq, u.Quantity)
    assert apyq == u.Quantity(1, u.m)
