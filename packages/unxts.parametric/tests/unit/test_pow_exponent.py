"""``pow`` with a dimensionless ``ParametricQuantity`` exponent.

The parametric-exponent ``pow`` rule is registered by ``unxts.parametric``.
"""

from unxts.parametric import PQ

import quaxed.numpy as jnp

import unxt as u


def test_pow_quantity_power():
    """A dimensionless parametric quantity (``PQ``) is a valid exponent."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "m")
    y = PQ(jnp.asarray(4, dtype=float), "")
    got = jnp.pow(x, y)
    exp = u.Q(jnp.pow(x.value, y.value), "m4")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)
