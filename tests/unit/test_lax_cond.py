"""Tests for ``lax.cond`` on quantities."""

import jax
import jax.numpy as jnp
import pytest
import quax
from astropy.units import UnitConversionError
from jax import lax

import unxt as u
from unxt._src.quantity.register_primitives import cond_p_q


def _cond(p, x):
    return lax.cond(p, lambda v: 2 * v, lambda v: v, x)


@pytest.mark.parametrize("jit", [False, True])
def test_cond_on_a_quantity_operand_keeps_its_unit(jit):
    """An array predicate with a quantity operand keeps the operand's unit."""
    f = quax.quaxify(jax.jit(_cond) if jit else _cond)
    got = f(jnp.ones((), dtype=bool), u.Q(1.0, "m"))
    assert u.ustrip("m", got) == 2.0


@pytest.mark.parametrize("jit", [False, True])
def test_cond_on_a_dimensionless_quantity_predicate(jit):
    """A dimensionless boolean quantity can be the predicate."""
    f = quax.quaxify(jax.jit(_cond) if jit else _cond)
    assert u.ustrip("m", f(u.Q(jnp.ones((), dtype=bool), ""), u.Q(1.0, "m"))) == 2.0
    assert u.ustrip("m", f(u.Q(jnp.zeros((), dtype=bool), ""), u.Q(1.0, "m"))) == 1.0


def test_cond_on_a_dimensionful_index_is_refused():
    """A dimensionful index has no meaning and is refused."""
    with pytest.raises(UnitConversionError):
        cond_p_q(u.Q(1, "m"), u.Q(1.0, "m"), branches=())
