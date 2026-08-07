"""Tests for `quax` registrations that the array-API suites do not reach."""

import jax.numpy as jnp
import numpy as np
import pytest
import quax
from jax import lax

import quaxed.numpy as qnp

import unxt as u
from unxt._src.quantity.register_primitives import cond_p_q


def test_cond_on_a_quantity_operand():
    """``lax.cond`` with an array predicate strips the quantity operand."""
    f = quax.quaxify(lambda p, x: lax.cond(p, lambda v: v, lambda v: 2 * v, x))
    assert np.isclose(np.asarray(f(jnp.asarray(1, dtype=bool), u.Q(1.0, "m"))), 1.0)


def test_cond_on_a_quantity_predicate_is_unsupported():
    """A quantity *predicate* has no meaning and is refused."""
    q = u.Q(1.0, "m")
    with pytest.raises(NotImplementedError):
        cond_p_q(q, q)


def test_angle_divided_by_angle_is_dimensionless():
    """``Angle / Angle`` degrades to a plain, dimensionless `Quantity`."""
    a = u.Angle(1.0, "deg")
    got = qnp.divide(a, a)
    assert isinstance(got, u.quantity.Quantity)
    assert got.unit == u.unit("")
    assert np.isclose(np.asarray(got.value), 1.0)


def test_scatter_add_quantity_operand_and_updates():
    """``scatter_add`` with quantity operand *and* updates keeps the unit."""
    dnums = lax.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )
    got = quax.quaxify(lax.scatter_add)(
        u.Q(jnp.ones(4), "m"), jnp.asarray([[1]]), u.Q(jnp.asarray([9.0]), "m"), dnums
    )
    assert got.unit == u.unit("m")
    assert np.allclose(np.asarray(got.value), [1.0, 10.0, 1.0, 1.0])
