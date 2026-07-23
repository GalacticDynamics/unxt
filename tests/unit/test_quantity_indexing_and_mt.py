"""Tests for Quantity.mT, and the .at[...] get/apply/power methods."""

import pytest

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


def test_at_apply_and_power_raise_explanatory_errors():
    """.at[...].apply / .power raise NotImplementedError *with a message*."""
    q = u.Q([1.0, 2.0, 3.0], "m")
    with pytest.raises(NotImplementedError, match="apply is not implemented"):
        q.at[0].apply(lambda x: x)
    with pytest.raises(NotImplementedError, match="power is not supported"):
        q.at[0].power(2)
