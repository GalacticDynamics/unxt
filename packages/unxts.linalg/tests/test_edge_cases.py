"""Error paths and rarely-taken branches of ``unxts.linalg``."""

import jax
import jax.numpy as jnp
import numpy as np
import plum
import pytest
import quax
from jax.interpreters import ad
from unxts.linalg import QuantityMatrix as QMat, UnitsMatrix, det, inv
from unxts.linalg._src._det import _det_jvp
from unxts.linalg._src._inv import _inv_jvp
from unxts.linalg._src._register_primitives import transpose_qm

import quaxed.numpy as qnp

import unxt as u

# =============================================================================
# UnitsMatrix


def test_normalize_unit_rejects_a_non_unit():
    with pytest.raises(TypeError, match="Expected an AbstractUnit or unit string"):
        UnitsMatrix((1.0, 2.0))


def test_rejects_an_object_array_with_too_many_dimensions():
    arr = np.empty((2, 2, 2), dtype=object)
    arr.flat[:] = [u.unit("m")] * 8
    with pytest.raises(ValueError, match="only supports 1D or 2D"):
        UnitsMatrix(arr)


def test_rejects_an_empty_sequence():
    with pytest.raises(ValueError, match="at least one element"):
        UnitsMatrix(())


@pytest.mark.parametrize(
    "structure",
    [
        (("m", "s"), ("m",)),  # ragged rows
        (("m", ("s",)),),  # nested leaf inside a row
        ("m", ("s",)),  # mixed leaf and nested
    ],
)
def test_rejects_a_ragged_structure(structure):
    with pytest.raises(ValueError, match="ragged structure"):
        UnitsMatrix(structure)


def test_elementwise_rejects_a_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        UnitsMatrix(("m", "s")) * UnitsMatrix(("m", "s", "kg"))


def test_rmul_matches_mul():
    um = UnitsMatrix(("m", "s"))
    assert (u.unit("kg") * um) == (um * u.unit("kg"))


def test_equality_edge_cases():
    um = UnitsMatrix(("m", "s"))
    assert um != UnitsMatrix(("m", "s", "kg"))  # shape mismatch
    assert um != ("m", 1.0)  # not convertible to a UnitsMatrix
    assert um.__eq__(object()) is NotImplemented


def test_iterating_a_2d_units_matrix_yields_rows():
    um = UnitsMatrix((("m", "s"), ("kg", "rad")))
    rows = list(um)
    assert all(isinstance(row, UnitsMatrix) for row in rows)
    assert rows[0] == UnitsMatrix(("m", "s"))


def test_indexing_a_scalar_element_returns_a_unit():
    um = UnitsMatrix((("m", "s"), ("kg", "rad")))
    assert um[np.asarray(0), np.asarray(1)] == u.unit("s")
    assert um[0] == UnitsMatrix(("m", "s"))


# =============================================================================
# QuantityMatrix


def test_convert_to_quantity_rejects_an_empty_unit_structure():
    """An empty `UnitsMatrix` has no unit to convert to; `is_uniform` is vacuous."""
    qm = QMat._mk(value=jnp.zeros(0), unit=UnitsMatrix(np.empty(0, dtype=object)))
    with pytest.raises(ValueError, match="no unit entries"):
        plum.convert(qm, u.Q)


def test_uconvert_to_the_same_units_is_a_no_op():
    qm = QMat(jnp.ones((2, 2)), (("m", "s"), ("m", "s")))
    assert u.uconvert(qm.unit, qm) is qm


# =============================================================================
# det / inv


@pytest.mark.parametrize("primitive", [det, inv])
def test_requires_at_least_2d(primitive):
    with pytest.raises(ValueError, match="requires at least 2-D input"):
        jax.eval_shape(quax.quaxify(primitive), jnp.ones(3))


@pytest.mark.parametrize("primitive", [det, inv])
def test_requires_a_square_matrix(primitive):
    with pytest.raises(ValueError, match="requires a square matrix"):
        jax.eval_shape(quax.quaxify(primitive), jnp.ones((2, 3)))


def test_det_of_a_quantity_matrix_requires_a_2d_unit_structure():
    qm = QMat(jnp.ones((2, 2)), ("m", "m"))
    with pytest.raises(ValueError, match="requires a 2-D unit structure"):
        quax.quaxify(det)(qm)


# =============================================================================
# registered primitives


def test_reduce_sum_over_an_unsupported_axis_combination():
    """Summing both logical axes of a 2-D unit structure has no single unit."""
    qm = QMat(jnp.ones((2, 2)), (("m", "s"), ("m", "s")))
    with pytest.raises(NotImplementedError, match="unsupported reduction"):
        quax.quaxify(lambda q: jnp.sum(q))(qm)


def test_quantity_matrix_division_variants():
    """Every ``div`` dispatch keeps the per-element units straight."""
    qm = QMat(jnp.full((2, 2), 4.0), (("m", "m"), ("m", "m")))
    other = QMat(jnp.full((2, 2), 2.0), (("s", "s"), ("s", "s")))
    arr = jnp.full((2, 2), 2.0)
    q = u.Q(2.0, "s")

    assert (qm / other).unit == UnitsMatrix((("m / s", "m / s"), ("m / s", "m / s")))
    assert (qm / q).unit == UnitsMatrix((("m / s", "m / s"), ("m / s", "m / s")))
    assert (q / qm).unit == UnitsMatrix((("s / m", "s / m"), ("s / m", "s / m")))
    assert (qm / arr).unit == qm.unit
    assert np.allclose(np.asarray((qm / arr).value), 2.0)


def test_zero_tangent_rule_short_circuits():
    """The JVP rules return a zero tangent for a symbolically-zero input."""
    x = jnp.asarray([[2.0, 0.0], [0.0, 3.0]])
    zero = ad.Zero(jax.typeof(x))

    _, det_tangent = _det_jvp((x,), (zero,))
    assert np.allclose(np.asarray(det_tangent), 0.0)

    _, inv_tangent = _inv_jvp((x,), (zero,))
    assert np.allclose(np.asarray(inv_tangent), 0.0)


class TestTransposeRankGuard:
    """`transpose_qm` only implements the matrix transpose."""

    def test_non_identity_permutation_below_rank_2(self):
        """A non-identity permutation of a rank-1 value has no matrix transpose.

        The identity permutation is handled just above as a no-op, so this is
        the only way to reach the rank guard.
        """
        v = QMat(jnp.array([1.0, 2.0]), unit=("m", "s"))
        with pytest.raises(NotImplementedError, match=r"requires ndim >= 2"):
            transpose_qm(v, permutation=(1,))


class TestTracedScalarGather:
    """A scalar gather with a *traced* index falls back to the uniform unit."""

    def test_scalar_gather_under_jit(self):
        """Under `jax.jit` the index is a tracer, so the unit cannot be read off it.

        With uniform units the answer does not depend on which element is
        picked, so the fallback returns that shared unit.
        """
        v = QMat(jnp.array([1.0, 2.0, 3.0]), unit=("m", "m", "m"))
        got = jax.jit(lambda q, i: qnp.take(q, i))(v, jnp.asarray(1))
        assert isinstance(got, u.Quantity)
        assert got.unit == u.unit("m")
        assert got.value == pytest.approx(2.0)

    def test_scalar_gather_under_jit_rejects_mixed_units(self):
        """Heterogeneous units make the traced pick ambiguous, so it raises."""
        v = QMat(jnp.array([1.0, 2.0, 3.0]), unit=("m", "s", "kg"))
        with pytest.raises(ValueError, match="requires all units to be equal"):
            jax.jit(lambda q, i: qnp.take(q, i))(v, jnp.asarray(1))
