"""Tests for the experimental unit-checked ``where``."""

import astropy.units as apyu
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import unxt as u
from unxt import experimental

_COND = jnp.asarray([True, False])


def test_where_converts_second_branch_to_first_unit() -> None:
    """``y`` is converted to ``x``'s unit; the result is in ``x``'s unit."""
    got = experimental.where(_COND, u.Q([1.0, 2.0], "m"), u.Q([0.003, 0.004], "km"))
    assert got.unit == u.unit("m")
    assert np.allclose(np.asarray(got.value), [1.0, 4.0])


def test_where_incompatible_units_raise() -> None:
    """Selecting between non-convertible units raises, like ``concat``."""
    with pytest.raises(apyu.UnitConversionError):
        experimental.where(_COND, u.Q([1.0, 2.0], "m"), u.Q([1.0, 2.0], "s"))


@pytest.mark.parametrize("raw_side", ["x", "y"])
def test_where_rejects_a_raw_array_branch(raw_side: str) -> None:
    """A raw-array branch is rejected -- no silent unit adoption.

    This is the whole point of the helper: unlike ``jnp.where`` (which must
    adopt the unit so JAX masking keeps working), this refuses the ambiguous mix
    and tells the caller to wrap the array.
    """
    q = u.Q([1.0, 2.0], "m")
    raw = jnp.asarray([3.0, 4.0])
    x, y = (raw, q) if raw_side == "x" else (q, raw)
    with pytest.raises(TypeError, match="both branches to be Quantities"):
        experimental.where(_COND, x, y)


def test_where_works_under_jit() -> None:
    """The unit conversion and selection survive ``jax.jit``."""
    fn = jax.jit(lambda a, b: experimental.where(_COND, a, b))
    got = fn(u.Q([1.0, 2.0], "m"), u.Q([0.003, 0.004], "km"))
    assert got.unit == u.unit("m")
    assert np.allclose(np.asarray(got.value), [1.0, 4.0])


def test_where_preserves_angle_type() -> None:
    """The result keeps the first branch's concrete type (e.g. ``Angle``)."""
    x = u.Angle([10.0, 20.0], "deg")
    y = u.Angle([1.0, 2.0], "rad")
    got = experimental.where(_COND, x, y)
    assert isinstance(got, u.Angle)
    assert got.unit == u.unit("deg")


def test_where_dimensionless_branches() -> None:
    """Two dimensionless quantities select without conversion."""
    got = experimental.where(_COND, u.Q([1.0, 2.0], ""), u.Q([3.0, 4.0], ""))
    assert got.unit == u.unit("")
    assert np.allclose(np.asarray(got.value), [1.0, 4.0])
