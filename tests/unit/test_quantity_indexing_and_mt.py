"""Tests for Quantity.mT, and the .at[...] get/apply/power methods."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import quax

import unxt as u


def test_static_quantity_mt_transposes():
    """StaticQuantity.mT returns the matrix transpose (was a TypeError)."""
    sq = u.StaticQuantity([[0, 1], [2, 3]], "m")
    assert sq.mT.value.tolist() == [[0, 2], [1, 3]]
    assert sq.mT.unit == u.unit("m")
    # and the regular Quantity still works
    assert u.Q([[0, 1], [2, 3]], "m").mT.value.tolist() == [[0, 2], [1, 3]]


def test_at_get_fill_value_accepts_a_quantity():
    """.at[oob].get(fill_value=Quantity) works (was unhashable ArrayImpl)."""
    q = u.Q([1.0, 2.0, 3.0], "m")
    got = q.at[10].get(mode="fill", fill_value=u.Q(-1.0, "m"))
    assert got.unit == u.unit("m")
    assert float(got.value) == -1.0
    # unit conversion of the fill value is honoured
    got_km = q.at[10].get(mode="fill", fill_value=u.Q(-0.001, "km"))
    assert float(got_km.value) == -1.0


def test_at_get_traced_fill_value_raises_clear_error():
    """A traced fill_value raises a clear TypeError, not a concretization error."""

    @jax.jit
    def f(fill):
        q = u.Q([1.0, 2.0, 3.0], "m")
        return q.at[10].get(mode="fill", fill_value=u.Q(fill, "m"))

    with pytest.raises(TypeError, match="requires a concrete scalar fill value"):
        f(-1.0)


def test_at_apply_and_power_raise_explanatory_errors():
    """.at[...].apply / .power raise NotImplementedError *with a message*."""
    q = u.Q([1.0, 2.0, 3.0], "m")
    with pytest.raises(NotImplementedError, match="apply is not implemented"):
        q.at[0].apply(lambda x: x)
    with pytest.raises(NotImplementedError, match="power is not supported"):
        q.at[0].power(2)


def test_at_helper_and_ref_reprs():
    """``.at`` and ``.at[...]`` name themselves, not the jax base classes."""
    q = u.Q([1.0, 2.0, 3.0], "m")
    assert repr(q.at).startswith("_QuantityIndexUpdateHelper(")
    assert repr(q.at[0]).startswith("_QuantityIndexUpdateRef(")


def test_scatter_add_between_quantities():
    """``.at[...].add`` propagates units, both operands being quantities."""
    q = u.Q([1.0, 1.0], "m")
    got = q.at[jnp.asarray([1])].add(u.Q([9.0], "m"))
    assert got.unit == u.unit("m")
    assert np.allclose(np.asarray(got.value), [1.0, 10.0])


def test_scatter_add_array_operand_with_quantity_updates():
    """A raw-array operand borrows the units of the quantity updates."""
    idx = jnp.asarray([1])
    got = quax.quaxify(lambda a, b: a.at[idx].add(b))(jnp.ones(2), u.Q([9.0], "m"))
    assert got.unit == u.unit("m")
    assert np.allclose(np.asarray(got.value), [1.0, 10.0])
