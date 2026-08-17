"""Unit tests specific to ``ParametricQuantity``."""

import copy as pycopy
import pickle
import re

import equinox as eqx
import jax.numpy as jnp
import pytest
import unxts.parametric as up
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


def test_type_parameter_from_a_unit_without_a_named_dimension():
    """A unit astropy cannot name falls back to a unit-derived dimension."""
    cls = up.ParametricQuantity[u.unit("m5/s3")]
    assert "m5" in str(u.dimension_of(cls))


class TestReconstructionProtocol:
    """`__reduce__` is the single reconstruction path for parametric quantities.

    `__getnewargs_ex__` used to be defined alongside it (both landed in #209,
    "fix: pickling parametric types") but was never reachable: the copy and
    pickle protocols consult ``__reduce_ex__`` -> ``__reduce__`` first. It was
    removed; these tests pin the behaviour it was supposed to provide.
    """

    @pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
    def test_pickle_roundtrip_every_protocol(self, protocol):
        """Every pickle protocol reconstructs value, unit and type parameter."""
        q = up.PQ([1, 2, 3], "m")
        got = pickle.loads(pickle.dumps(q, protocol=protocol))  # noqa: S301
        assert type(got) is type(q)
        assert got.unit == q.unit
        assert jnp.array_equal(got.value, q.value)

    @pytest.mark.parametrize("copier", [pycopy.copy, pycopy.deepcopy])
    def test_copy_roundtrip(self, copier):
        """`copy` and `deepcopy` reconstruct through the same path."""
        q = up.PQ([1, 2, 3], "m")
        got = copier(q)
        assert type(got) is type(q)
        assert got.unit == q.unit
        assert jnp.array_equal(got.value, q.value)

    def test_getnewargs_ex_is_not_defined(self):
        """The superseded hook is gone, so there is no second, dead path.

        `object` does not define it either, so a plain ``hasattr`` is a
        sufficient check.
        """
        assert not hasattr(up.PQ, "__getnewargs_ex__")
