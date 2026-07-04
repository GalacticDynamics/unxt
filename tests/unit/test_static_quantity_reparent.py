"""Characterization: StaticQuantity is a non-parametric AbstractQuantity."""

import jax
import numpy as np

import unxt as u
from unxt.quantity import AbstractParametricQuantity, AbstractQuantity


def test_staticquantity_is_non_parametric():
    sq = u.StaticQuantity(np.array([1.0, 2.0]), "m")
    assert isinstance(sq, AbstractQuantity)
    assert not isinstance(sq, AbstractParametricQuantity)
    # The class itself is not plum-parametric anymore.
    assert type(sq) is u.StaticQuantity


def test_staticquantity_repr_unchanged():
    sq = u.StaticQuantity(np.array([1.0, 2.0]), "m")
    assert repr(sq) == "StaticQuantity(array([1., 2.]), unit='m')"


def test_staticquantity_hash_and_eq():
    a = u.StaticQuantity(np.array([1.0, 2.0]), "m")
    b = u.StaticQuantity(np.array([1.0, 2.0]), "m")
    assert isinstance(hash(a), int)
    assert hash(a) == hash(b)
    assert bool(a == b)


def test_staticquantity_jit_static_argument():
    """Its purpose: usable as a jit static arg; equal values do not retrace."""
    calls = {"n": 0}

    @jax.jit
    def f(x, sq):
        calls["n"] += 1
        return x * float(np.asarray(sq.value))

    sq1 = u.StaticQuantity(np.array(2.0), "m")
    sq2 = u.StaticQuantity(np.array(2.0), "m")  # equal → no retrace
    sq3 = u.StaticQuantity(np.array(3.0), "m")  # different → retrace
    f = jax.jit(f, static_argnums=1)
    _ = f(1.0, sq1)
    _ = f(1.0, sq2)
    n_after_equal = calls["n"]
    _ = f(1.0, sq3)
    assert calls["n"] == n_after_equal + 1


def test_staticquantity_promotions():
    sq = u.StaticQuantity(np.array(2.0), "m")
    q = u.Quantity(3.0, "m")
    assert isinstance(sq + sq, u.StaticQuantity)
    assert isinstance(sq + q, u.Quantity)
