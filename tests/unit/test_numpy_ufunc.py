"""Test numpy ufunc support on quantities via ``__array_ufunc__``.

NumPy ufuncs (e.g. ``np.add``, ``np.multiply``, ``np.sqrt``) must propagate
units instead of silently dropping them.
"""

from unittest.mock import Mock

import numpy as np
import pytest

import unxt as u
from unxt._src.quantity import register_ufuncs


@pytest.fixture(autouse=True)
def _clean_ufunc_registry():
    """Restore the global ufunc registry after each test."""
    saved = dict(register_ufuncs._UFUNC_REGISTRY)
    yield
    register_ufuncs._UFUNC_REGISTRY.clear()
    register_ufuncs._UFUNC_REGISTRY.update(saved)


def test_multiply_propagates_units():
    """``np.multiply`` on two lengths yields an area."""
    q1 = u.Q(5.0, "m")
    q2 = u.Q(3.0, "m")

    got = np.multiply(q1, q2)

    assert isinstance(got, u.Q)
    assert got.unit == u.unit("m2")
    assert got.value == 15.0


def test_sqrt_scales_units():
    """``np.sqrt`` halves the unit exponent: ``sqrt(m2) -> m``."""
    q = u.Q(4.0, "m2")

    got = np.sqrt(q)

    assert isinstance(got, u.Q)
    assert got.unit == u.unit("m")
    assert got.value == 2.0


def test_add_reduce_1d():
    """``np.add.reduce`` sums a 1-D quantity, keeping the unit."""
    q = u.Q([2.0, 3.0, 4.0], "m")

    got = np.add.reduce(q)

    assert isinstance(got, u.Q)
    assert got.unit == u.unit("m")
    assert got.value == 9.0


def test_add_reduce_2d_default_axis():
    """``np.add.reduce`` reduces over axis 0 by default (numpy semantics)."""
    q = u.Q([[1.0, 2.0], [3.0, 4.0]], "m")

    got = np.add.reduce(q)

    assert isinstance(got, u.Q)
    assert got.unit == u.unit("m")
    assert np.array_equal(np.asarray(got.value), np.asarray([4.0, 6.0]))


def test_multiply_reduce_scales_units():
    """``np.multiply.reduce`` multiplies values and units."""
    q = u.Q([2.0, 3.0, 4.0], "m")

    got = np.multiply.reduce(q)

    assert isinstance(got, u.Q)
    assert got.unit == u.unit("m3")
    assert got.value == 24.0


def test_add_accumulate():
    """``np.add.accumulate`` is a cumulative sum that keeps the unit."""
    q = u.Q([2.0, 3.0, 4.0], "m")

    got = np.add.accumulate(q)

    assert isinstance(got, u.Q)
    assert got.unit == u.unit("m")
    assert np.array_equal(np.asarray(got.value), np.asarray([2.0, 5.0, 9.0]))


def test_call_with_plain_array_operand():
    """A ufunc called with a plain ndarray and a quantity keeps the unit."""
    q = u.Q([1.0, 2.0], "m")
    arr = np.array([10.0, 20.0])

    got = np.multiply(arr, q)

    assert isinstance(got, u.Q)
    assert got.unit == u.unit("m")
    assert np.array_equal(np.asarray(got.value), np.asarray([10.0, 40.0]))


def test_unregistered_custom_ufunc_raises():
    """An unregistered custom ufunc errors loudly instead of dropping units."""
    doubler = np.frompyfunc(lambda x: 2 * x, 1, 1)
    q = u.Q(3.0, "m")

    with pytest.raises(TypeError):
        doubler(q)


def test_registered_custom_ufunc():
    """A custom ufunc works once a unit-aware handler is registered."""
    doubler = np.frompyfunc(lambda x: 2 * x, 1, 1)

    @u.quantity.register_ufunc(doubler)
    def _(ufunc, method, x, /, **kw):
        return u.Q(2 * x.value, x.unit)

    got = doubler(u.Q(3.0, "m"))

    assert isinstance(got, u.Q)
    assert got.unit == u.unit("m")
    assert got.value == 6.0


def test_name_collision_custom_ufunc_not_delegated():
    """A non-builtin ufunc whose name collides with a builtin is not delegated.

    Delegation must key on the ufunc *identity*, not merely its ``__name__``,
    so an unregistered custom ufunc named e.g. ``"add"`` errors loudly rather
    than being silently routed to ``quaxed.numpy.add``.
    """
    # A custom ufunc named "add" that is not numpy's built-in ``np.add``.
    # (numba's @vectorize produces exactly this; not installed here, so a
    # spec'd Mock stands in -- it is an instance of ``np.ufunc`` for dispatch.)
    fake = Mock(spec=np.ufunc)
    fake.__name__ = "add"
    q = u.Q([1.0, 2.0], "m")

    assert register_ufuncs.apply_ufunc(fake, "__call__", (q, q), {}) is NotImplemented
    assert register_ufuncs.apply_ufunc(fake, "reduce", (q,), {}) is NotImplemented
    assert register_ufuncs.apply_ufunc(fake, "accumulate", (q,), {}) is NotImplemented


def test_unsupported_method_raises():
    """An unsupported ufunc method errors loudly (no silent unit drop)."""
    q = u.Q([1.0, 2.0], "m")

    with pytest.raises(TypeError):
        np.add.outer(q, q)


def test_bare_quantity_also_supported():
    """``__array_ufunc__`` is inherited by all quantity types."""
    q = u.quantity.BareQuantity(5.0, "m")

    got = np.multiply(q, q)

    assert got.unit == u.unit("m2")
    assert got.value == 25.0
