"""Unit tests specific to ``ParametricQuantity``."""

import pickle
import re

import equinox as eqx
import jax.numpy as jnp
import pytest
from jaxtyping import Array
from plum import parametric
from unxts.parametric import PQ, AbstractParametricQuantity

import unxt as u


def test_parametric():
    """Test the parametric strategy (``ParametricQuantity`` / ``PQ``)."""
    # Inferred
    q = PQ(1, "m")
    (dims,) = q._type_parameter
    assert dims == u.dimension("length")

    # Explicit
    q = PQ["length"](1, "m")
    (dims,) = q._type_parameter
    assert dims == u.dimension("length")

    q = PQ["length"](jnp.ones((1, 2)), "m")
    (dims,) = q._type_parameter
    assert dims == u.dimension("length")

    # type-checks
    with pytest.raises(ValueError, match=re.escape("Physical type mismatch.")):
        PQ["time"](1, "m")

    # The lightweight default ``Quantity`` does NOT dimension-check: the
    # subscript is accepted but the unit-dimension mismatch does not raise.
    u.Q["time"](1, "m")


def test_rpow():
    """Test the ``ParametricQuantity.__rpow__`` method."""
    # Scalar base with dimensionless ParametricQuantity exponent.
    # ``pow`` with an array/scalar base only has a registered rule for a
    # *parametric* dimensionless exponent (``pow_p_vq``), so use ``PQ``.
    q = PQ(2.0, "")  # dimensionless
    result = 3.0**q
    assert jnp.isclose(result.value, 9.0)
    assert result.unit == u.unit("")

    # Exponent must be dimensionless
    q = PQ(2.0, "m")
    with pytest.raises(Exception):  # noqa: B017, PT011
        _ = 3.0**q


@parametric
class NewQuantity(AbstractParametricQuantity):
    """ParametricQuantity with a flag."""

    value: Array = eqx.field(converter=jnp.asarray)
    unit: str = eqx.field(converter=u.unit)
    flag: bool = eqx.field(static=True, kw_only=True)


def test_parametric_pickle_dumps_with_kw_fields():
    x = NewQuantity([1, 2, 3], "m", flag=True)
    assert isinstance(pickle.dumps(x), bytes)
