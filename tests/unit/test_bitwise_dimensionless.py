"""Bitwise/logical ops on quantities require dimensionless operands.

The error must name the operation and both *actual* units, not leak astropy's
misleading "'m' and '' are not convertible" (which implies a dimensionless
operand the user never wrote).
"""

import jax.numpy as jnp
import pytest

import quaxed.numpy as qnp

import unxt as u


def test_bool_add_on_dimensionful_gives_clear_error():
    """`Q(bool, 'm') + Q(bool, 'm')` (JAX lowers to or_p) names both 'm' units."""
    q = u.Q(jnp.array([True, False]), "m")
    with pytest.raises(ValueError, match=r"dimensionless.*'m'.*'m'"):
        _ = q + q


@pytest.mark.parametrize("op", [qnp.bitwise_or, qnp.bitwise_and, qnp.bitwise_xor])
def test_bitwise_on_dimensionful_raises_clear_error(op):
    with pytest.raises(ValueError, match=r"dimensionless.*'m'.*'s'"):
        _ = op(u.Q(1, "m"), u.Q(2, "s"))


@pytest.mark.parametrize("op", [qnp.bitwise_or, qnp.bitwise_and, qnp.bitwise_xor])
def test_bitwise_on_dimensionless_still_works(op):
    got = op(u.Q(6, ""), u.Q(3, ""))
    assert got.unit == u.unit("")
