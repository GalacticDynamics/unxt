"""Bitwise/logical ops on quantities require dimensionless operands.

The error must name the operation and both *actual* units, not leak astropy's
misleading "'m' and '' are not convertible" (which implies a dimensionless
operand the user never wrote).
"""

import jax.numpy as jnp
import pytest

import quaxed.numpy as qnp

import unxt as u

# (callable, op-name as it appears in the error message)
BITWISE_OPS = [
    (qnp.bitwise_or, "or"),
    (qnp.bitwise_and, "and"),
    (qnp.bitwise_xor, "xor"),
]


def test_bool_add_on_dimensionful_gives_clear_error():
    """`Q(bool, 'm') + Q(bool, 'm')` (JAX lowers to or_p) names or and both 'm'."""
    q = u.Q(jnp.array([True, False]), "m")
    pattern = r"bitwise/logical or.*dimensionless.*'m'.*'m'"
    with pytest.raises(ValueError, match=pattern):
        _ = q + q


@pytest.mark.parametrize(("op", "name"), BITWISE_OPS)
def test_bitwise_qq_on_dimensionful_raises_clear_error(op, name):
    """Two dimensionful quantities: the message names the op and both units."""
    with pytest.raises(
        ValueError, match=rf"bitwise/logical {name}.*dimensionless.*'m'.*'s'"
    ):
        _ = op(u.Q(1, "m"), u.Q(2, "s"))


@pytest.mark.parametrize(("op", "name"), BITWISE_OPS)
def test_bitwise_quantity_array_raises_clear_error(op, name):
    """A dimensionful quantity mixed with a plain array is caught just as clearly.

    Both operand orders route through the quantity/array overloads, which strip
    to dimensionless too and would otherwise leak astropy's confusing message.
    """
    pattern = rf"bitwise/logical {name}.*dimensionless.*'m'"
    with pytest.raises(ValueError, match=pattern):
        _ = op(u.Q(jnp.array([1]), "m"), jnp.array([2]))
    with pytest.raises(ValueError, match=pattern):
        _ = op(jnp.array([2]), u.Q(jnp.array([1]), "m"))


def test_bitwise_not_on_dimensionful_raises_clear_error():
    """The unary `not_p` overload is guarded as well."""
    with pytest.raises(ValueError, match=r"bitwise/logical not.*dimensionless.*'m'"):
        _ = qnp.bitwise_not(u.Q(jnp.array([1]), "m"))


@pytest.mark.parametrize(("op", "name"), BITWISE_OPS)
def test_bitwise_on_dimensionless_still_works(op, name):
    """The dimensionless path is unchanged, for both quantity/quantity and array."""
    assert op(u.Q(6, ""), u.Q(3, "")).unit == u.unit("")
    assert op(u.Q(jnp.array([1]), ""), jnp.array([2])).unit == u.unit("")
    assert op(jnp.array([2]), u.Q(jnp.array([1]), "")).unit == u.unit("")


def test_bitwise_not_on_dimensionless_still_works():
    assert qnp.bitwise_not(u.Q(1, "")).unit == u.unit("")
