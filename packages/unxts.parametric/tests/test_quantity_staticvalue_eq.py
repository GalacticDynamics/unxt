"""The default `Quantity` (not only `ParametricQuantity`) gets StaticValue eq."""

import numpy as np
from unxts.parametric import PQ

import unxt as u


def test_quantity_staticvalue_equality_scalar_bool():
    """A StaticValue-backed `Quantity` returns a scalar bool from ``==``."""
    sv1 = u.quantity.StaticValue(np.array([1.0, 2.0]))
    sv2 = u.quantity.StaticValue(np.array([1.0, 2.0]))
    sv3 = u.quantity.StaticValue(np.array([3.0, 4.0]))

    result = u.Q(sv1, "m") == u.Q(sv2, "m")
    assert isinstance(result, bool), f"expected scalar bool, got {type(result)}"
    assert result is True
    assert (u.Q(sv1, "m") == u.Q(sv3, "m")) is False

    # Equality is unit-blind: different unit labels compare not-equal.
    sv_km = u.quantity.StaticValue(np.array([0.001, 0.002]))
    assert (u.Q(sv1, "m") == u.Q(sv_km, "km")) is False

    # Incompatible dimensions are never equal.
    sv_s = u.quantity.StaticValue(np.array([1.0, 2.0]))
    assert (u.Q(sv1, "m") == u.Q(sv_s, "s")) is False


def test_quantity_and_parametric_staticvalue_equality_match():
    """`Quantity` and `ParametricQuantity` agree on StaticValue equality."""
    sv1 = u.quantity.StaticValue(np.array([1.0, 2.0]))
    sv2 = u.quantity.StaticValue(np.array([1.0, 2.0]))
    assert (u.Q(sv1, "m") == u.Q(sv2, "m")) == (PQ(sv1, "m") == PQ(sv2, "m"))
