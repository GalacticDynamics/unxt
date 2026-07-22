"""Tests for physical equivalence of quantities (unit-aware).

`==` on `StaticValue`-backed quantities is unit-blind (label match); `equivalent`
(and the `is_equivalent` method) is the unit-aware "same physical quantity" check.
"""

import numpy as np

import unxt as u
from unxt.quantity import StaticValue


def test_static_equivalent_across_units_where_equality_is_false():
    """The #708 case: equivalent True while unit-blind ``==`` is False."""
    sv1 = StaticValue(np.array([1.0, 2.0]))
    sv_km = StaticValue(np.array([0.001, 0.002]))  # == [1, 2] m

    a = u.Q(sv1, "m")
    b = u.Q(sv_km, "km")

    assert (a == b) is False  # unit-blind equality
    assert u.equivalent(a, b) is True  # unit-aware equivalence
    assert a.is_equivalent(b) is True  # method parity


def test_static_equivalent_scalar_bool_and_unequal_values():
    """StaticValue-backed equivalence returns a scalar bool."""
    sv1 = StaticValue(np.array([1.0, 2.0]))
    sv_other = StaticValue(np.array([9.0, 9.0]))
    assert u.equivalent(u.Q(sv1, "m"), u.Q(sv1, "m")) is True
    assert u.equivalent(u.Q(sv1, "m"), u.Q(sv_other, "m")) is False


def test_incompatible_dimensions_are_not_equivalent():
    """Incompatible dimensions -> False (never raises)."""
    sv = StaticValue(np.array([1.0, 2.0]))
    assert u.equivalent(u.Q(sv, "m"), u.Q(sv, "s")) is False
    # array-backed too
    assert u.equivalent(u.Q([1.0], "m"), u.Q([1.0], "s")) is False


def test_array_backed_equivalent_is_elementwise_and_unit_aware():
    """Array-backed equivalence mirrors ``==`` (element-wise), unit-aware."""
    got = u.equivalent(u.Q([1.0, 2.0], "m"), u.Q([0.001, 0.009], "km"))
    assert isinstance(got, u.quantity.Quantity)
    assert got.unit == u.unit("")
    assert np.array_equal(np.asarray(got.value), np.asarray([True, False]))


def test_equivalent_is_reflexive():
    """A quantity is equivalent to itself."""
    q = u.Q([1.0, 2.0, 3.0], "m")
    assert bool(np.all(np.asarray(u.equivalent(q, q).value)))


def test_equivalent_is_shared_plum_function_with_unitsystems():
    """`unxt.equivalent` is the same dispatched function unit systems use."""
    assert u.equivalent is u.unitsystems.equivalent
    # unit-system dispatch still works (no regression)
    assert u.equivalent(u.unitsystem("m", "s", "kg"), u.unitsystem("km", "s", "kg"))
