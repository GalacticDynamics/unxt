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


class TestGetNewArgsEx:
    """`__getnewargs_ex__` returns the reconstruction args for `__new__`."""

    def test_returns_no_positional_args_and_all_fields_as_kwargs(self):
        """The contract: no positional args, every field as a keyword."""
        q = up.PQ([1, 2, 3], "m")
        args, kwargs = q.__getnewargs_ex__()
        assert args == ()
        assert dict(kwargs).keys() == {"value", "unit"}
        assert dict(kwargs)["unit"] == u.unit("m")

    def test_reduce_takes_precedence_on_every_copy_path(self):
        """Note: nothing *reaches* `__getnewargs_ex__` in practice.

        `AbstractParametricQuantity` also defines `__reduce__`, which the copy
        and pickle protocols consult first, so `copy`, `deepcopy` and `pickle`
        all bypass this method. It is kept as the documented `__new__` contract
        and exercised directly above; this test pins the precedence so a future
        change to `__reduce__` does not silently alter reconstruction.
        """
        q = up.PQ([1, 2, 3], "m")
        calls = []
        original = type(q).__getnewargs_ex__

        def spy(self):
            calls.append(1)
            return original(self)

        type(q).__getnewargs_ex__ = spy
        try:
            pycopy.copy(q)
            pycopy.deepcopy(q)
            pickle.loads(pickle.dumps(q))  # noqa: S301
        finally:
            type(q).__getnewargs_ex__ = original

        assert calls == []
