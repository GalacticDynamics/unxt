"""Tests for unxts.linalg.QuantityMatrix."""

import math

import astropy.units as apu
import jax
import jax.numpy as jnp
import numpy as np
import plum
import pytest
import quax
from astropy.units import imperial  # registers °F
from jax import lax
from unxts.linalg import (
    QuantityMatrix as QMat,
    UnitsMatrix,
    matmul,
    matvec,
    vecdot,
    vecmat,
)
from unxts.linalg._src import (
    _convert_value_matrix,
    _convert_value_vector,
    det as qm_det,
    inv as qm_inv,
)
from unxts.linalg._src._register_primitives import (
    _check_contract,
    _wrap_operand,
    gather_qm,
    transpose_qm,
)

import quaxed.numpy as qnp

import unxt as u

# ---------------------------------------------------------------------------
# Unit shorthands (visual noise reduction)
# ---------------------------------------------------------------------------

_m = u.unit("m")
_s = u.unit("s")
_kg = u.unit("kg")
_rad = u.unit("rad")
_km = u.unit("km")
_ms = u.unit("ms")
_g = u.unit("g")
_deg = u.unit("deg")
_min = u.unit("min")
_dimless = u.unit("")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def unit_2x2():
    """Return a simple 2x2 unit grid: m, s, kg, rad."""
    return ((_m, _s), (_kg, _rad))


@pytest.fixture
def qm_2x2(unit_2x2):
    """Return a 2x2 QuantityMatrix with values 1-4."""
    return QMat(value=jnp.array([[1.0, 2.0], [3.0, 4.0]]), unit=unit_2x2)


@pytest.fixture
def unit_2x2_alt():
    """Alternative 2x2 units convertible to unit_2x2: km, ms, g, deg."""
    return ((_km, _ms), (_g, _deg))


@pytest.fixture
def unit_1d():
    """Return a simple 1D unit tuple: m, s, kg."""
    return (_m, _s, _kg)


@pytest.fixture
def qm_1d(unit_1d):
    """Return a 1D QuantityMatrix (vector) with values 1-3."""
    return QMat(value=jnp.array([1, 2, 3]), unit=unit_1d)


@pytest.fixture
def unit_1d_alt():
    """Alternative 1D units convertible to unit_1d: km, ms, g."""
    return (_km, _ms, _g)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    """Tests for QuantityMatrix construction and basic properties."""

    def test_value_unit_shape_mismatch_rejected(self):
        """Value's trailing shape must match the unit structure."""
        # 1-D: value length 3 vs unit length 2
        with pytest.raises(ValueError, match="does not match"):
            QMat(jnp.array([1.0, 2.0, 3.0]), unit=(_m, _s))
        # 2-D: value (2, 2) vs unit (2, 3)
        with pytest.raises(ValueError, match="does not match"):
            QMat(jnp.ones((2, 2)), unit=((_m, _s, _kg), (_m, _s, _kg)))
        # value has fewer dims than the unit structure
        with pytest.raises(ValueError, match="fewer"):
            QMat(jnp.array([1.0, 2.0]), unit=((_m, _s),))

    def test_leading_batch_dims_allowed(self):
        """Extra *leading* axes are batch dims and are accepted."""
        qm = QMat(jnp.ones((5, 3, 2, 2)), unit=((_m, _s), (_kg, _m)))
        assert qm.shape == (5, 3, 2, 2)
        assert qm.ndim == 2  # logical

    def test_from_cdict_rejects_matrix_values(self):
        """from_cdict rejects QuantityMatrix / UnitsMatrix values with a clear error."""
        inner = QMat(jnp.array([1.0, 2.0]), unit=(_m, _s))
        with pytest.raises(TypeError, match="scalar-unit"):
            QMat.from_cdict({"a": inner})
        with pytest.raises(TypeError, match="scalar-unit"):
            QMat.from_cdict({"a": UnitsMatrix((_m, _s))})

    def test_batched_indexing_leaves_unit(self):
        """Indexing a batch axis reduces the value but keeps the unit structure."""
        qm = QMat(jnp.arange(12.0).reshape(3, 2, 2), unit=((_m, _s), (_kg, _m)))
        # Select a batch element -> a single matrix with the full unit.
        assert qm[0].value.shape == (2, 2)
        assert qm[0].unit == ((_m, _s), (_kg, _m))
        # Batch slice keeps the batch axis and the unit.
        assert qm[0:2].value.shape == (2, 2, 2)
        assert qm[0:2].unit == ((_m, _s), (_kg, _m))
        # Batch index + logical row -> 1-D vector with that row's units.
        assert qm[0, 1].value.shape == (2,)
        assert qm[0, 1].unit == (_kg, _m)
        # Batch index + full element -> scalar Quantity.
        assert qm[0, 1, 0].unit == _kg
        # Ellipsis / newaxis are not supported (deterministic error).
        with pytest.raises(NotImplementedError, match="Ellipsis"):
            _ = qm[..., 0]

    def test_shape(self, qm_2x2):
        assert qm_2x2.shape == (2, 2)

    def test_n_rows(self, qm_2x2):
        assert qm_2x2.shape[-2] == 2

    def test_n_cols(self, qm_2x2):
        assert qm_2x2.shape[-1] == 2

    def test_value(self, qm_2x2):
        expected = jnp.array([[1, 2], [3, 4]])
        assert jnp.array_equal(qm_2x2.value, expected)

    def test_unit(self, qm_2x2, unit_2x2):
        assert qm_2x2.unit == unit_2x2

    def test_aval(self, qm_2x2):
        aval = qm_2x2.aval()
        assert aval.shape == (2, 2)
        assert jnp.issubdtype(aval.dtype, jnp.floating)

    def test_materialise_raises(self, qm_2x2):
        with pytest.raises(RuntimeError, match="materialise"):
            qm_2x2.materialise()

    def test_batch_dims(self):
        """Batch dimensions are supported via leading axes."""
        qm = QMat(jnp.ones((5, 3, 2)), unit=((_m, _s), (_m, _s), (_m, _s)))
        assert qm.shape == (5, 3, 2)
        assert qm.shape[-2] == 3
        assert qm.shape[-1] == 2

    def test_1x1(self):
        """Degenerate 1x1 matrix."""
        qm = QMat(jnp.array([[42]]), unit=((_m,),))
        assert qm.shape[-2] == 1
        assert qm.shape[-1] == 1

    def test_nonsquare(self):
        """Non-square 2x3 matrix."""
        qm = QMat(jnp.ones((2, 3)), unit=((_m, _s, _kg), (_m, _s, _kg)))
        assert qm.shape[-2] == 2
        assert qm.shape[-1] == 3

    def test_unit_is_unitsmatrix(self, qm_2x2):
        """The ``unit`` field is always a ``UnitsMatrix`` instance."""
        assert isinstance(qm_2x2.unit, UnitsMatrix)

    def test_unit_converter_from_plain_tuples(self):
        """Plain nested tuples (of strings) are converted to ``UnitsMatrix``."""
        qm = QMat(jnp.array([[1]]), unit=(("m",),))
        assert isinstance(qm.unit, UnitsMatrix)
        assert qm.unit[0, 0] == _m

    def test_1d_construction(self, qm_1d, unit_1d):
        """1D vector construction."""
        assert qm_1d.ndim == 1
        assert qm_1d.shape == (3,)
        assert qm_1d.shape[-1] == 3
        assert qm_1d.unit == unit_1d

    def test_1d_value(self, qm_1d):
        """1D vector value."""
        expected = jnp.array([1, 2, 3])
        assert jnp.array_equal(qm_1d.value, expected)

    def test_1d_from_strings(self):
        """1D vector from unit strings."""
        qm = QMat(jnp.array([7, 8]), unit=("m", "s"))
        assert isinstance(qm.unit, UnitsMatrix)
        assert qm.unit[0] == _m
        assert qm.unit[1] == _s

    def test_1d_batch_dims(self):
        """1D vector with batch dimensions."""
        qm = QMat(jnp.ones((5, 3)), unit=(_m, _s, _kg))
        assert qm.ndim == 1
        assert qm.shape == (5, 3)

    def test_ndim_property_1d(self, qm_1d):
        """Ndim property returns 1 for vectors."""
        assert qm_1d.ndim == 1

    def test_ndim_property_2d(self, qm_2x2):
        """Ndim property returns 2 for matrices."""
        assert qm_2x2.ndim == 2

    def test_repr(self, qm_2x2):
        """``repr(QuantityMatrix(...))`` succeeds and contains key info."""
        r = repr(qm_2x2)
        assert "QuantityMatrix" in r
        assert "((m, s), (kg, rad))" in r

    def test_repr_1x1(self):
        """Repr for a 1x1 matrix includes trailing-comma tuple syntax."""
        qm = QMat(jnp.array([[42]]), unit=((_m,),))
        r = repr(qm)
        assert "QuantityMatrix" in r
        assert "((m,),)" in r


# ---------------------------------------------------------------------------
# UnitsMatrix
# ---------------------------------------------------------------------------


class TestUnitsMatrix:
    """Tests for the ``UnitsMatrix`` object-array-backed unit structure."""

    def test_bare_string_rejected(self):
        """A bare unit string is rejected (else it splits into per-char units)."""
        with pytest.raises(TypeError, match="bare unit string"):
            UnitsMatrix("ms")
        with pytest.raises(TypeError, match="bare unit string"):
            UnitsMatrix("m")
        # The intended single-unit form works.
        assert UnitsMatrix(("ms",)).shape == (1,)

    def test_repr_roundtrips(self):
        """The structural repr / to_string form is accepted back as a constructor."""
        for m in (
            UnitsMatrix((_m, _s, _kg)),
            UnitsMatrix(((_m, _s), (_kg, _m))),
            UnitsMatrix((_m,)),
        ):
            assert UnitsMatrix(m.to_string()) == m
            assert eval(repr(m)) == m  # noqa: S307

    def test_unbalanced_parens_rejected(self):
        """A structural string with unbalanced parentheses raises ValueError."""
        with pytest.raises(ValueError, match="Unbalanced"):
            UnitsMatrix("(a, (b)")  # unmatched '('
        with pytest.raises(ValueError, match="Unbalanced"):
            UnitsMatrix("(), )")  # unmatched ')'

    def test_not_isinstance_structured_unit(self):
        """UnitsMatrix is NOT a StructuredUnit — it is a standalone class."""
        units = UnitsMatrix(((_m, _s),))
        assert not isinstance(units, apu.StructuredUnit)

    def test_repr(self):
        """repr() identifies the type and format."""
        units = UnitsMatrix((_m, _s, _kg))
        assert repr(units) == 'UnitsMatrix("(m, s, kg)")'

    def test_repr_2d(self):
        """repr() for a 2D UnitsMatrix."""
        units = UnitsMatrix(((_m, _s), (_kg, _rad)))
        assert repr(units) == 'UnitsMatrix("((m, s), (kg, rad))")'

    def test_unit_from_object_array_1d(self):
        """u.unit() accepts a 1-D numpy object array of AbstractUnit."""
        arr = np.array([_m, _s, _kg], dtype=object)
        result = u.unit(arr)
        assert isinstance(result, UnitsMatrix)
        assert result.shape == (3,)
        assert result[0] == _m
        assert result[2] == _kg

    def test_unit_from_object_array_2d(self):
        """u.unit() accepts a 2-D numpy object array of AbstractUnit."""
        arr = np.array([[_m, _s], [_kg, _rad]], dtype=object)
        result = u.unit(arr)
        assert isinstance(result, UnitsMatrix)
        assert result.shape == (2, 2)
        assert result[0, 0] == _m
        assert result[1, 1] == _rad

    def test_hashable(self):
        """UnitsMatrix is hashable (required for use as static eqx field)."""
        units = UnitsMatrix((_m, _s, _kg))
        h = hash(units)
        assert isinstance(h, int)

    def test_hash_stable(self):
        """Equal UnitsMatrix objects have the same hash."""
        a = UnitsMatrix((_m, _s, _kg))
        b = UnitsMatrix((_m, _s, _kg))
        assert hash(a) == hash(b)

    def test_to_string_2x2(self):
        units = UnitsMatrix(((_m, _s), (_kg, _rad)))
        assert units.to_string() == "((m, s), (kg, rad))"

    def test_to_string_1x1(self):
        units = UnitsMatrix(((_m,),))
        assert units.to_string() == "((m,),)"

    def test_to_string_1x2(self):
        units = UnitsMatrix(((_m, _s),))
        assert units.to_string() == "((m, s),)"

    def test_to_string_2x1(self):
        units = UnitsMatrix(((_m,), (_s,)))
        assert units.to_string() == "((m,), (s,))"

    def test_to_string_1d(self):
        """1D units to_string."""
        units = UnitsMatrix((_m, _s, _kg))
        assert units.to_string() == "(m, s, kg)"

    def test_to_string_1d_single(self):
        """Single element 1D."""
        units = UnitsMatrix((_m,))
        assert units.to_string() == "(m,)"

    def test_from_strings_2d(self):
        """``UnitsMatrix`` accepts unit strings via the ``u.unit`` converter."""
        units = UnitsMatrix((("m", "s"), ("kg", "rad")))
        assert units[0, 0] == _m
        assert units[1, 1] == _rad

    def test_from_strings_1d(self):
        """1D from strings."""
        units = UnitsMatrix(("m", "s", "kg"))
        assert units[0] == _m
        assert units[2] == _kg

    def test_idempotent_2d(self):
        """Constructing from an existing ``UnitsMatrix`` returns an equal copy."""
        orig = UnitsMatrix(((_m, _s), (_kg, _rad)))
        copy = UnitsMatrix(orig)
        assert copy == orig
        assert isinstance(copy, UnitsMatrix)

    def test_idempotent_1d(self):
        """1D idempotent."""
        orig = UnitsMatrix((_m, _s, _kg))
        copy = UnitsMatrix(orig)
        assert copy == orig
        assert isinstance(copy, UnitsMatrix)

    def test_shape_1d(self):
        """1D shape."""
        units = UnitsMatrix((_m, _s, _kg))
        assert units.shape == (3,)

    def test_shape_2d(self):
        """2D shape."""
        units = UnitsMatrix(((_m, _s), (_kg, _rad)))
        assert units.shape == (2, 2)

    def test_ndim_1d(self):
        """1D ndim."""
        units = UnitsMatrix((_m, _s))
        assert units.ndim == 1

    def test_ndim_2d(self):
        """2D ndim."""
        units = UnitsMatrix(((_m, _s), (_kg, _rad)))
        assert units.ndim == 2

    def test_indexing_1d(self):
        """1D indexing."""
        units = UnitsMatrix((_m, _s, _kg))
        assert units[0] == _m
        assert units[1] == _s
        assert units[2] == _kg

    def test_indexing_2d_tuple(self):
        """2D tuple indexing."""
        units = UnitsMatrix(((_m, _s), (_kg, _rad)))
        assert units[0, 0] == _m
        assert units[0, 1] == _s
        assert units[1, 0] == _kg
        assert units[1, 1] == _rad

    def test_indexing_2d_single(self):
        """2D single-index returns a row."""
        units = UnitsMatrix(((_m, _s), (_kg, _rad)))
        row = units[0]
        assert isinstance(row, UnitsMatrix)
        assert row.shape == (2,)
        assert row[0] == _m
        assert row[1] == _s

    def test_rejects_ragged_2d_structure(self):
        """Nested tuples must be rectangular, not ragged."""
        with pytest.raises(ValueError, match="ragged structure"):
            UnitsMatrix(((_m, _s), (_kg,)))

    def test_rejects_mixed_nested_and_leaf_structure(self):
        """Mixed leaf/nested rows are rejected as ragged structures."""
        with pytest.raises(ValueError, match="ragged structure"):
            UnitsMatrix(((_m, _s), _kg))

    def test_rejects_single_unit(self):
        """A bare unit is not a valid matrix/vector unit structure."""
        with pytest.raises((TypeError, ValueError)):
            UnitsMatrix(_m)

    def test_to_tuple_1d(self):
        units = UnitsMatrix((_m, _s, _kg))
        assert units.to_tuple() == (_m, _s, _kg)

    def test_to_tuple_2d(self):
        units = UnitsMatrix(((_m, _s), (_kg, _rad)))
        assert units.to_tuple() == ((_m, _s), (_kg, _rad))

    def test_to_tuple_is_public(self):
        """to_tuple() is the canonical round-trip — no private ._units access."""
        units = UnitsMatrix(((_km, _deg), (_m, _rad)))
        assert units.to_tuple() == ((_km, _deg), (_m, _rad))


# ---------------------------------------------------------------------------
# _convert_value_matrix / _convert_value_vector  (internal helpers)
# ---------------------------------------------------------------------------


class TestConvertValueMatrix:
    """Tests for the element-wise unit conversion helper (2D)."""

    def test_noop_same_units(self, unit_2x2):
        """If from_units == to_units no conversion happens."""
        val = jnp.array([[7, 8], [9, 10]])
        out = _convert_value_matrix(val, unit_2x2, unit_2x2)
        assert jnp.array_equal(out, val)

    def test_km_to_m(self):
        """1 km → 1000 m."""
        out = _convert_value_matrix(jnp.array([[3]]), ((_km,),), ((_m,),))
        assert jnp.isclose(out[0, 0], 3000)

    def test_mixed_conversion(self, unit_2x2, unit_2x2_alt):
        """Convert from (km, ms, g, deg) → (m, s, kg, rad)."""
        val = jnp.array([[1, 1000], [3000, 180]])
        out = _convert_value_matrix(val, unit_2x2_alt, unit_2x2)
        assert jnp.isclose(out[0, 0], 1000)  # 1 km -> 1000 m
        assert jnp.isclose(out[0, 1], 1)  # 1000 ms -> 1 s
        assert jnp.isclose(out[1, 0], 3)  # 3000 g -> 3 kg
        assert jnp.isclose(out[1, 1], math.pi, atol=1e-4)  # 180 deg -> pi rad

    def test_preserves_batch(self):
        """Batch dimensions are preserved."""
        val = jnp.array([[[2]], [[5]]])  # (2, 1, 1)
        out = _convert_value_matrix(val, ((_km,),), ((_m,),))
        assert out.shape == (2, 1, 1)
        assert jnp.isclose(out[0, 0, 0], 2000)
        assert jnp.isclose(out[1, 0, 0], 5000)


class TestConvertValuePoint:
    """Tests for the element-wise unit conversion helper (1D)."""

    def test_noop_same_units(self, unit_1d):
        """If from_units == to_units no conversion happens."""
        val = jnp.array([7, 8, 9])
        out = _convert_value_vector(val, unit_1d, unit_1d)
        assert jnp.array_equal(out, val)

    def test_km_to_m(self):
        """1 km → 1000 m."""
        out = _convert_value_vector(jnp.array([3]), (_km,), (_m,))
        assert jnp.isclose(out[0], 3000)

    def test_mixed_conversion(self, unit_1d, unit_1d_alt):
        """Convert from (km, ms, g) → (m, s, kg)."""
        val = jnp.array([1, 1000, 3000])
        out = _convert_value_vector(val, unit_1d_alt, unit_1d)
        assert jnp.isclose(out[0], 1000)  # 1 km -> 1000 m
        assert jnp.isclose(out[1], 1)  # 1000 ms -> 1 s
        assert jnp.isclose(out[2], 3)  # 3000 g -> 3 kg

    def test_preserves_batch(self):
        """Batch dimensions are preserved."""
        val = jnp.array([[2], [5]])  # (2, 1)
        out = _convert_value_vector(val, (_km,), (_m,))
        assert out.shape == (2, 1)
        assert jnp.isclose(out[0, 0], 2000)
        assert jnp.isclose(out[1, 0], 5000)


# ---------------------------------------------------------------------------
# Addition  (quax lax.add_p)
# ---------------------------------------------------------------------------


@quax.quaxify
def _add(a, b):
    return a + b


class TestAddition:
    """Tests for QuantityMatrix + QuantityMatrix."""

    def test_same_units(self, qm_2x2, unit_2x2):
        """Simple add, same units."""
        other = QMat(value=jnp.array([[10, 20], [30, 40]]), unit=unit_2x2)
        result = _add(qm_2x2, other)
        expected = jnp.array([[11, 22], [33, 44]])
        assert jnp.allclose(result.value, expected)
        assert result.unit == unit_2x2

    def test_result_keeps_lhs_units(self, qm_2x2, unit_2x2, unit_2x2_alt):
        """Result units come from the LHS."""
        other = QMat(value=jnp.array([[1, 1000], [3000, 180]]), unit=unit_2x2_alt)
        result = _add(qm_2x2, other)
        assert result.unit == unit_2x2

    def test_mixed_unit_values(self, qm_2x2, unit_2x2_alt):
        """Values are correctly converted before addition."""
        other = QMat(value=jnp.array([[1, 1000], [3000, 180]]), unit=unit_2x2_alt)
        res_val = _add(qm_2x2, other).value
        assert jnp.isclose(res_val[0, 0], 1001)  # 1 + 1000 m = 1001
        assert jnp.isclose(res_val[0, 1], 3)  # 2 + 1.0 s = 3
        assert jnp.isclose(res_val[1, 0], 6)  # 3 + 3.0 kg = 6
        assert jnp.isclose(res_val[1, 1], 4 + math.pi, atol=1e-4)  # 4+pi rad≈7.14159

    def test_add_zeros(self, qm_2x2, unit_2x2):
        """Adding zeros gives original values."""
        zeros = QMat(jnp.zeros((2, 2)), unit=unit_2x2)
        result = _add(qm_2x2, zeros)
        assert jnp.allclose(result.value, qm_2x2.value)

    def test_commutativity_same_units(self, unit_2x2):
        """A + b == b + a when units are the same."""
        a = QMat(jnp.array([[1, 2], [3, 4]]), unit=unit_2x2)
        b = QMat(jnp.array([[5, 6], [7, 8]]), unit=unit_2x2)
        r1 = _add(a, b)
        r2 = _add(b, a)
        assert jnp.allclose(r1.value, r2.value)

    def test_batch_addition(self, unit_2x2):
        """Batch dimensions are supported."""
        a = QMat(jnp.ones((3, 2, 2)), unit=unit_2x2)
        b = QMat(2 * jnp.ones((3, 2, 2)), unit=unit_2x2)
        result = _add(a, b)
        assert result.shape == (3, 2, 2)
        assert jnp.allclose(result.value, 3 * jnp.ones((3, 2, 2)))

    def test_1x1(self):
        """1x1 addition."""
        a = QMat(jnp.array([[3]]), unit=((_m,),))
        b = QMat(jnp.array([[7]]), unit=((_m,),))
        result = _add(a, b)
        assert jnp.isclose(result.value[0, 0], 10)

    def test_1d_addition_same_units(self, qm_1d, unit_1d):
        """1D vector addition, same units."""
        other = QMat(jnp.array([10, 20, 30]), unit=unit_1d)
        result = _add(qm_1d, other)
        expected = jnp.array([11, 22, 33])
        assert jnp.allclose(result.value, expected)
        assert result.unit == unit_1d

    def test_1d_addition_mixed_units(self, qm_1d, unit_1d_alt):
        """1D vector addition with unit conversion."""
        other = QMat(jnp.array([1.0, 1000.0, 3000.0]), unit=unit_1d_alt)
        result = _add(qm_1d, other)
        assert jnp.isclose(result.value[0], 1001)  # 1 + 1000 m
        assert jnp.isclose(result.value[1], 3)  # 2 + 1.0 s
        assert jnp.isclose(result.value[2], 6)  # 3 + 3.0 kg
        assert result.unit == qm_1d.unit

    def test_1d_batch_addition(self, unit_1d):
        """1D batch addition."""
        a = QMat(jnp.ones((3, 3)), unit=unit_1d)
        b = QMat(2 * jnp.ones((3, 3)), unit=unit_1d)
        result = _add(a, b)
        assert result.shape == (3, 3)
        assert jnp.allclose(result.value, 3 * jnp.ones((3, 3)))

    def test_direct_operator_mixed_units(self):
        """Direct ``+`` supports mixed-but-compatible units."""
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        q1 = QMat(x, (("km", "deg"), ("km", "deg")))
        q2 = QMat(x, (("m", "rad"), ("m", "rad")))
        result = q1 + q2

        assert result.unit.to_string() == "((km, deg), (km, deg))"
        assert jnp.isclose(result.value[0, 0], 1.001)
        assert jnp.isclose(result.value[1, 0], 3.003)
        assert jnp.isclose(result.value[0, 1], 2 + 2 * (180 / jnp.pi), atol=1e-8)
        assert jnp.isclose(result.value[1, 1], 4 + 720 / jnp.pi, atol=1e-8)

    def test_per_element_conversion_with_logarithmic_unit(self):
        """Conversion is per-element (`uconvert_value`), not one global scale.

        A logarithmic unit (``mag``) converts as identity while a neighbouring
        length element scales by 1000 — a single shared scale factor could not
        produce both. Guards the ``_convert_value`` per-element contract that
        add/sub rely on for non-scale (log/affine) units.
        """
        mag = u.unit("mag")
        x = QMat(jnp.array([1.0, 1.0]), unit=(mag, _m))
        y = QMat(jnp.array([2.0, 2.0]), unit=(mag, _km))  # 2 mag, 2 km
        result = _add(x, y)
        # mag: 1 + 2 = 3 (identity conversion); m: 1 + 2000 = 2001 (km→m).
        assert jnp.allclose(result.value, jnp.array([3.0, 2001.0]))
        assert result.unit == (mag, _m)


# ---------------------------------------------------------------------------
# Subtraction  (quax lax.sub_p)
# ---------------------------------------------------------------------------


@quax.quaxify
def _sub(a, b):
    return a - b


class TestSubtraction:
    """Tests for QuantityMatrix - QuantityMatrix."""

    def test_same_units(self, qm_2x2, unit_2x2):
        """Simple sub, same units."""
        other = QMat(value=jnp.array([[10, 20], [30, 40]]), unit=unit_2x2)
        result = _sub(other, qm_2x2)
        expected = jnp.array([[9, 18], [27, 36]])
        assert jnp.allclose(result.value, expected)
        assert result.unit == unit_2x2

    def test_result_keeps_lhs_units(self, qm_2x2, unit_2x2, unit_2x2_alt):
        """Result units come from the LHS."""
        other = QMat(value=jnp.array([[1, 1000], [3000, 180]]), unit=unit_2x2_alt)
        result = _sub(qm_2x2, other)
        assert result.unit == unit_2x2

    def test_mixed_unit_values(self, qm_2x2, unit_2x2_alt):
        """Values are correctly converted before subtraction."""
        other = QMat(value=jnp.array([[1, 1000], [3000, 180]]), unit=unit_2x2_alt)
        res_val = _sub(qm_2x2, other).value
        assert jnp.isclose(res_val[0, 0], -999)  # 1 - 1000 m
        assert jnp.isclose(res_val[0, 1], 1)  # 2 - 1.0 s
        assert jnp.isclose(res_val[1, 0], 0, atol=1e-4)  # 3 - 3.0 kg
        assert jnp.isclose(res_val[1, 1], 4 - math.pi, atol=1e-4)  # 4-pi rad

    def test_sub_zeros(self, qm_2x2, unit_2x2):
        """Subtracting zeros gives original values."""
        zeros = QMat(jnp.zeros((2, 2)), unit=unit_2x2)
        result = _sub(qm_2x2, zeros)
        assert jnp.allclose(result.value, qm_2x2.value)

    def test_self_subtraction(self, qm_2x2, unit_2x2):
        """A - a == 0."""
        result = _sub(qm_2x2, qm_2x2)
        assert jnp.allclose(result.value, jnp.zeros((2, 2)))
        assert result.unit == unit_2x2

    def test_anticommutativity_same_units(self, unit_2x2):
        """A - b == -(b - a) when units are the same."""
        a = QMat(jnp.array([[1, 2], [3, 4]]), unit=unit_2x2)
        b = QMat(jnp.array([[5, 6], [7, 8]]), unit=unit_2x2)
        r1 = _sub(a, b)
        r2 = _sub(b, a)
        assert jnp.allclose(r1.value, -r2.value)

    def test_batch_subtraction(self, unit_2x2):
        """Batch dimensions are supported."""
        a = QMat(3 * jnp.ones((3, 2, 2)), unit=unit_2x2)
        b = QMat(jnp.ones((3, 2, 2)), unit=unit_2x2)
        result = _sub(a, b)
        assert result.shape == (3, 2, 2)
        assert jnp.allclose(result.value, 2 * jnp.ones((3, 2, 2)))

    def test_1x1(self):
        """1x1 subtraction."""
        a = QMat(jnp.array([[7]]), unit=((_m,),))
        b = QMat(jnp.array([[3]]), unit=((_m,),))
        result = _sub(a, b)
        assert jnp.isclose(result.value[0, 0], 4)

    def test_1d_subtraction_same_units(self, qm_1d, unit_1d):
        """1D vector subtraction, same units."""
        other = QMat(jnp.array([10, 20, 30]), unit=unit_1d)
        result = _sub(other, qm_1d)
        expected = jnp.array([9, 18, 27])
        assert jnp.allclose(result.value, expected)
        assert result.unit == unit_1d

    def test_1d_subtraction_mixed_units(self, qm_1d, unit_1d_alt):
        """1D vector subtraction with unit conversion."""
        other = QMat(jnp.array([1.0, 1000.0, 3000.0]), unit=unit_1d_alt)
        result = _sub(qm_1d, other)
        assert jnp.isclose(result.value[0], -999)  # 1 - 1000 m
        assert jnp.isclose(result.value[1], 1)  # 2 - 1.0 s
        assert jnp.isclose(result.value[2], 0, atol=1e-4)  # 3 - 3.0 kg
        assert result.unit == qm_1d.unit


# ---------------------------------------------------------------------------
# Element-wise multiply / divide  (quax lax.mul_p / div_p)
# ---------------------------------------------------------------------------


@quax.quaxify
def _mul(a, b):
    return a * b


@quax.quaxify
def _div(a, b):
    return a / b


class TestElementwiseMulDiv:
    """Element-wise mul/div keep the QuantityMatrix type and compose units."""

    def test_qm_times_quantity_both_orders(self):
        qm = QMat(jnp.array([[1.0, 2.0], [3.0, 4.0]]), unit=((_m, _s), (_kg, _rad)))
        for r in (_mul(qm, u.Q(2.0, "s")), _mul(u.Q(2.0, "s"), qm)):  # commutes
            assert isinstance(r, QMat)
            assert jnp.allclose(r.value, jnp.array([[2.0, 4.0], [6.0, 8.0]]))
            assert r.unit == ((_m * _s, _s * _s), (_kg * _s, _rad * _s))

    def test_qm_times_qm_hadamard(self):
        qm = QMat(jnp.array([[1.0, 2.0], [3.0, 4.0]]), unit=((_m, _s), (_kg, _rad)))
        r = _mul(qm, qm)
        assert isinstance(r, QMat)
        assert jnp.allclose(r.value, jnp.array([[1.0, 4.0], [9.0, 16.0]]))
        assert r.unit == ((_m * _m, _s * _s), (_kg * _kg, _rad * _rad))

    def test_scalar_leaves_units(self):
        qv = QMat(jnp.array([1.0, 2.0, 3.0]), unit=(_m, _s, _kg))
        r = _mul(qv, 2.0)
        assert isinstance(r, QMat)
        assert r.unit == (_m, _s, _kg)

    def test_division(self):
        qm = QMat(jnp.array([[2.0, 4.0], [6.0, 8.0]]), unit=((_m, _s), (_kg, _rad)))
        r = _div(qm, u.Q(2.0, "s"))
        assert isinstance(r, QMat)
        assert jnp.allclose(r.value, jnp.array([[1.0, 2.0], [3.0, 4.0]]))
        assert r.unit == ((_m / _s, _s / _s), (_kg / _s, _rad / _s))
        # dimensionless / qm inverts units
        r2 = _div(1.0, QMat(jnp.array([2.0, 4.0]), unit=(_m, _s)))
        assert isinstance(r2, QMat)
        assert r2.unit == (_m**-1, _s**-1)

    def test_quantity_times_qm_is_usable_downstream(self):
        """A quantity-on-left product is a real QuantityMatrix, not a Quantity."""
        qm = QMat(jnp.ones((2, 2)), unit=((_m, _s), (_kg, _rad)))
        r = _mul(u.Q(2.0, "s"), qm)
        r2 = _add(r, r)  # would raise if r were a plain Quantity holding a UnitsMatrix
        assert jnp.allclose(r2.value, 2 * r.value)


class TestReduceSum:
    """lax.reduce_sum on a QuantityMatrix respects leading batch axes."""

    def test_batch_axis_reduction_keeps_units(self):
        qm = QMat(jnp.ones((3, 2, 2)), unit=((_m, _s), (_kg, _rad)))
        r = quax.quaxify(lambda a: jnp.sum(a, axis=0))(qm)  # sum over batch
        assert r.value.shape == (2, 2)
        assert r.unit == ((_m, _s), (_kg, _rad))  # unchanged

    def test_logical_row_and_col_reduction(self):
        # Units must be dimensionally uniform along the *summed* axis. For row
        # reduction (axis 1) each column is uniform-dimension; the output takes
        # the first row's units.
        qm = QMat(jnp.ones((3, 2, 2)), unit=((_m, _s), (_km, _s)))
        row = quax.quaxify(lambda a: jnp.sum(a, axis=1))(qm)  # logical rows
        assert row.value.shape == (3, 2)
        assert row.unit == (_m, _s)
        # For column reduction (axis 2) each row is uniform-dimension; the
        # output takes the first column's units.
        qm2 = QMat(jnp.ones((3, 2, 2)), unit=((_m, _km), (_s, _s)))
        col = quax.quaxify(lambda a: jnp.sum(a, axis=2))(qm2)  # logical cols
        assert col.value.shape == (3, 2)
        assert col.unit == (_m, _s)

    def test_row_reduction_converts_compatible_units(self):
        """A column of compatible-but-different units is converted, not relabelled."""
        # column 0 has units [m, km]; summing must give 1 m + 1 km = 1001 m.
        qm = QMat(jnp.array([[1.0, 1.0], [1.0, 1.0]]), unit=((_m, _s), (_km, _s)))
        row = quax.quaxify(lambda a: jnp.sum(a, axis=0))(qm)
        assert row.unit == (_m, _s)
        assert jnp.allclose(row.value, jnp.array([1001.0, 2.0]))

    def test_col_reduction_converts_compatible_units(self):
        """A row of compatible-but-different units is converted before summing."""
        # row 0 has units [km, m]; summing must give 1 km + 1000 m = 2 km.
        qm = QMat(jnp.array([[1.0, 1000.0], [1.0, 1.0]]), unit=((_km, _m), (_km, _m)))
        col = quax.quaxify(lambda a: jnp.sum(a, axis=1))(qm)
        assert col.unit == (_km, _km)
        assert jnp.allclose(col.value, jnp.array([2.0, 1.001]))

    def test_reduction_incompatible_units_raises(self):
        """Summing a column of dimensionally incompatible units raises."""
        qm = QMat(jnp.array([[1.0, 1.0], [1.0, 1.0]]), unit=((_m, _s), (_kg, _s)))
        with pytest.raises(Exception, match=r"(?i)convert|unit"):
            quax.quaxify(lambda a: jnp.sum(a, axis=0))(qm)


# ---------------------------------------------------------------------------
# Dot product / matmul  (quax lax.dot_general_p)
# ---------------------------------------------------------------------------


@quax.quaxify
def _matmul(a, b):
    return a @ b


class TestDotProduct:
    """Tests for QuantityMatrix @ QuantityMatrix."""

    def test_simple_matmul_uniform_units(self):
        """2x2 @ 2x1 with uniform units along contraction axis."""
        a = QMat(jnp.array([[2, 3], [4, 5]]), unit=((_m, _m), (_kg, _kg)))
        b = QMat(jnp.array([[10], [20]]), unit=((_s,), (_s,)))
        result = _matmul(a, b)
        # C[0,0] = 2*10 + 3*20 = 80 in m*s
        # C[1,0] = 4*10 + 5*20 = 140 in kg*s
        assert jnp.isclose(result.value[0, 0], 80)
        assert jnp.isclose(result.value[1, 0], 140)
        assert result.unit == ((_m * _s,), (_kg * _s,))
        assert result.shape[-2] == 2
        assert result.shape[-1] == 1

    def test_matmul_with_unit_conversion(self):
        """Contraction axis has mixed units that need conversion."""
        # A is 2x2 with units [[m, km], [kg, kg]]
        # B is 2x1 with units [[s], [s]]
        # C[0,0] = m*s + km*s -> converted to m*s (ref is j=0)
        #        = 2*10 + (3 km = 3000 m)*20 s = 20 + 60000 = 60020
        a = QMat(jnp.array([[2, 3], [4, 5]]), unit=((_m, _km), (_kg, _kg)))
        b = QMat(jnp.array([[10], [20]]), unit=((_s,), (_s,)))
        result = _matmul(a, b)
        assert jnp.isclose(result.value[0, 0], 60020)
        # C[1,0] = 4*10 + 5*20 = 140 (uniform kg*s, no conversion)
        assert jnp.isclose(result.value[1, 0], 140)

    def test_matmul_2x2_by_2x2(self):
        """Square 2x2 @ 2x2 matmul."""
        a = QMat(jnp.array([[1, 2], [3, 4]]), unit=((_m, _m), (_m, _m)))
        b = QMat(jnp.array([[5, 6], [7, 8]]), unit=((_s, _s), (_s, _s)))
        result = _matmul(a, b)
        # Standard matmul: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        #                = [[19, 22], [43, 50]]
        expected = jnp.array([[19, 22], [43, 50]])
        assert jnp.allclose(result.value, expected)
        ms = _m * _s
        assert result.unit == ((ms, ms), (ms, ms))

    def test_matmul_identity(self):
        """Multiply by identity matrix."""
        a = QMat(jnp.array([[3, 7], [11, 13]]), unit=((_m, _m), (_m, _m)))
        identity = QMat(jnp.eye(2), unit=((_dimless, _dimless), (_dimless, _dimless)))
        result = _matmul(a, identity)
        assert jnp.allclose(result.value, a.value)

    def test_matmul_output_units(self):
        """Output unit[i][k] = lhs.unit[i][0] * rhs.unit[0][k]."""
        a = QMat(jnp.array([[1, 1]]), unit=((_m, _m),))
        b = QMat(jnp.array([[2, 3], [4, 5]]), unit=((_s, _kg), (_s, _kg)))
        result = _matmul(a, b)
        # Output shape: 1x2
        assert result.shape[-2] == 1
        assert result.shape[-1] == 2
        # Output units: row 0 from A.unit[0][0]=m, col 0 from B.unit[0][0]=s,
        # col 1 from B.unit[0][1]=kg
        assert result.unit[0][0] == _m * _s
        assert result.unit[0][1] == _m * _kg

    def test_matmul_1x1(self):
        """1x1 @ 1x1 is scalar product."""
        a = QMat(jnp.array([[3]]), unit=((_m,),))
        b = QMat(jnp.array([[7]]), unit=((_s,),))
        result = _matmul(a, b)
        assert jnp.isclose(result.value[0, 0], 21)
        assert result.unit == ((_m * _s,),)

    def test_matmul_quantity_left_quantitymatrix_right(self):
        """`Quantity @ QuantityMatrix` yields a well-formed QuantityMatrix.

        Regression: with a plain Quantity on the left there was no
        ``(Quantity, QuantityMatrix)`` dot_general rule, so unxt's generic
        ``AbstractQuantity`` rule won and built a `Quantity` whose ``.unit`` was
        a `UnitsMatrix` -- a malformed object. Both operand orders must give a
        `QuantityMatrix` with a `UnitsMatrix` unit.
        """
        # Uniform units along the contraction so the dot product is
        # dimensionally valid; the bug was about the result *type*.
        q_mat = u.Q(jnp.eye(2), "kg")  # plain Quantity, single unit
        vec = QMat(jnp.array([2.0, 3.0]), unit=(_m, _m))

        got = _matmul(q_mat, vec)
        assert isinstance(got, QMat)
        assert isinstance(got.unit, UnitsMatrix)
        # Every element's unit, not just the first, so a partial/batched
        # unit-propagation bug can't slip through.
        assert tuple(got.unit) == (_kg * _m, _kg * _m)
        assert jnp.allclose(got.value, jnp.array([2.0, 3.0]))

        # Consistent with the already-working reverse order.
        mat = QMat(jnp.eye(2), unit=((_kg, _kg), (_kg, _kg)))
        rev = _matmul(mat, u.Q(jnp.array([2.0, 3.0]), "s"))
        assert isinstance(rev, QMat)
        assert isinstance(rev.unit, UnitsMatrix)
        assert tuple(rev.unit) == (_kg * _s, _kg * _s)
        assert jnp.allclose(rev.value, jnp.array([2.0, 3.0]))

    def test_batched_matmul_2x2(self):
        """Leading batch axis: (B, 2, 2) @ (B, 2, 2) contracts per batch element.

        ``jnp.matmul`` lowers this to a ``dot_general`` with a non-empty batch
        dimension; the handler must accept the standard (batched) matmul pattern
        and broadcast the per-element units over the leading batch axis.
        """
        a_val = jnp.array(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
        )  # (2, 2, 2)
        b_val = jnp.broadcast_to(jnp.eye(2), (2, 2, 2))
        a = QMat(a_val, unit=((_m, _m), (_m, _m)))
        b = QMat(b_val, unit=((_s, _s), (_s, _s)))
        result = _matmul(a, b)
        assert result.shape == (2, 2, 2)
        # Multiplying each batch element by the identity leaves values unchanged.
        assert jnp.allclose(result.value, a_val)
        ms = _m * _s
        assert result.unit == ((ms, ms), (ms, ms))

    def test_matmul_rhs_unit_conversion(self):
        """Unit conversion needed on the RHS contraction axis."""
        # A: 1x2, units [[m, m]]
        # B: 2x1, units [[s], [min]]
        # C[0,0] = m*s + m*min -> ref = m*s
        #        = 1*1 + 1*1 min -> 1*1 + 1*60 s = 1 + 60 = 61 in m*s
        a = QMat(jnp.array([[1, 1]]), ((_m, _m),))
        b = QMat(jnp.array([[1], [1]]), ((_s,), (_min,)))
        result = _matmul(a, b)
        assert jnp.isclose(result.value[0, 0], 61)
        assert result.unit == ((_m * _s,),)

    def test_1d_dot_product_uniform_units(self):
        """1D @ 1D vector dot product with uniform units."""
        a = QMat(jnp.array([2, 3]), unit=(_m, _m))
        b = QMat(jnp.array([4, 5]), unit=(_s, _s))
        result = _matmul(a, b)
        # Result should be a scalar Quantity, not a QuantityMatrix
        # 2*4 + 3*5 = 8 + 15 = 23 in m*s
        assert isinstance(result, u.Q)
        assert jnp.isclose(result.value, 23)
        assert result.unit == _m * _s

    def test_1d_dot_product_mixed_units(self):
        """1D @ 1D with mixed units requiring conversion."""
        # a: [1 m, 1 km], b: [1 s, 1 s]
        # Result = 1*1 + 1000*1 = 1 + 1000 = 1001 in m*s
        a = QMat(jnp.array([1, 1]), unit=(_m, _km))
        b = QMat(jnp.array([1, 1]), unit=(_s, _s))
        result = _matmul(a, b)
        assert isinstance(result, u.Q)
        assert jnp.isclose(result.value, 1001)
        assert result.unit == _m * _s

    def test_1d_dot_product_batch(self):
        """1D @ 1D with batch dimensions."""
        # Batch of 3 vectors, each length 2
        a = QMat(jnp.array([[1, 2], [3, 4], [5, 6]]), unit=(_m, _m))
        b = QMat(jnp.array([[7, 8], [9, 10], [11, 12]]), unit=(_s, _s))

        @quax.quaxify
        def dot_batched(x, y):
            return x @ y

        result = jax.vmap(dot_batched)(a, b)
        # [1*7 + 2*8, 3*9 + 4*10, 5*11 + 6*12] = [23, 67, 127]
        assert jnp.isclose(result.value[0], 23)
        assert jnp.isclose(result.value[1], 67)
        assert jnp.isclose(result.value[2], 127)


# ---------------------------------------------------------------------------
# Public products: matmul / matvec / vecmat / vecdot
# ---------------------------------------------------------------------------


class TestProducts:
    """The batch-safe `unxts.linalg` matrix/vector product wrappers."""

    def test_matmul_2d_2d(self):
        """matmul(matrix, matrix) → matrix, with per-element units."""
        a = QMat(jnp.array([[1.0, 2.0], [3.0, 4.0]]), unit=((_m, _m), (_m, _m)))
        b = QMat(jnp.eye(2), unit=((_s, _s), (_s, _s)))
        r = matmul(a, b)
        assert jnp.allclose(r.value, a.value)  # times identity
        ms = _m * _s
        assert r.unit == ((ms, ms), (ms, ms))

    def test_matvec_unbatched(self):
        """matvec(matrix, vector) → vector."""
        a = QMat(jnp.array([[1.0, 2.0], [3.0, 4.0]]), unit=((_m, _m), (_m, _m)))
        v = QMat(jnp.array([1.0, 1.0]), unit=(_s, _s))
        w = matvec(a, v)
        assert w.ndim == 1
        assert jnp.allclose(w.value, jnp.array([3.0, 7.0]))
        assert w.unit == (_m * _s, _m * _s)

    def test_matvec_batched(self):
        """A batch of matrices applied to a batch of vectors (per batch element)."""
        a_val = jnp.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        a = QMat(a_val, unit=((_m, _m), (_m, _m)))
        v = QMat(jnp.array([[1.0, 1.0], [1.0, 1.0]]), unit=(_s, _s))
        w = matvec(a, v)
        assert w.shape == (2, 2)
        # b0: [1+2, 3+4]=[3,7]; b1: [5+6, 7+8]=[11,15]
        assert jnp.allclose(w.value, jnp.array([[3.0, 7.0], [11.0, 15.0]]))
        assert w.unit == (_m * _s, _m * _s)

    def test_matmul_rejects_batched_matvec(self):
        """Matmul rejects a batched vector operand with a pointer to matvec.

        A batched vector's value ``(B, K)`` is indistinguishable from a matrix.
        Without the guard, matmul silently returns the matvec answer for square
        batches (N==K==B) but raises for non-square shapes — an inconsistent
        trap. The guard makes it consistently error in both cases.
        """
        # Square batch (N==K==B) — this used to *silently succeed*.
        a_sq = QMat(jnp.ones((2, 2, 2)), unit=((_m, _m), (_m, _m)))
        v_sq = QMat(jnp.ones((2, 2)), unit=(_s, _s))
        with pytest.raises(ValueError, match="matvec"):
            matmul(a_sq, v_sq)
        # Non-square batch.
        a_ns = QMat(jnp.ones((2, 2, 3)), unit=((_m, _m, _m), (_m, _m, _m)))
        v_ns = QMat(jnp.ones((2, 3)), unit=(_s, _s, _s))
        with pytest.raises(ValueError, match="matvec"):
            matmul(a_ns, v_ns)
        # matvec handles both.
        assert matvec(a_sq, v_sq).shape == (2, 2)
        assert matvec(a_ns, v_ns).shape == (2, 2)

    def test_matmul_guard_is_narrow(self):
        """The guard only rejects *batched* vectors, not the valid matmul cases."""
        # Unbatched matrix @ vector: matmul promotes the 1-D vector — allowed.
        A = QMat(jnp.array([[1.0, 2.0], [3.0, 4.0]]), unit=((_m, _m), (_m, _m)))
        v = QMat(jnp.array([1.0, 1.0]), unit=(_s, _s))
        assert matmul(A, v).value.shape == (2,)
        # Batched matrix @ matrix — allowed.
        Ab = QMat(jnp.ones((3, 2, 2)), unit=((_m, _m), (_m, _m)))
        Bb = QMat(jnp.ones((3, 2, 2)), unit=((_s, _s), (_s, _s)))
        assert matmul(Ab, Bb).value.shape == (3, 2, 2)

    def test_vecmat_unbatched(self):
        """vecmat(vector, matrix) → vector (transpose of matvec)."""
        v = QMat(jnp.array([1.0, 1.0]), unit=(_s, _s))
        a = QMat(jnp.array([[1.0, 2.0], [3.0, 4.0]]), unit=((_m, _km), (_m, _km)))
        w = vecmat(v, a)
        assert w.ndim == 1
        assert jnp.allclose(w.value, jnp.array([4.0, 6.0]))
        assert w.unit == (_s * _m, _s * _km)

    def test_vecmat_batched(self):
        """A batch of vectors times a batch of matrices."""
        v = QMat(jnp.ones((2, 2)), unit=(_s, _s))
        a_val = jnp.broadcast_to(jnp.array([[1.0, 2.0], [3.0, 4.0]]), (2, 2, 2))
        a = QMat(a_val, unit=((_m, _km), (_m, _km)))
        w = vecmat(v, a)
        assert w.shape == (2, 2)
        assert jnp.allclose(w.value, jnp.array([[4.0, 6.0], [4.0, 6.0]]))

    def test_vecdot_unbatched(self):
        """vecdot(vector, vector) → scalar Quantity."""
        a = QMat(jnp.array([1.0, 2.0]), unit=(_m, _km))
        b = QMat(jnp.array([3.0, 4.0]), unit=(_s, _s))
        d = vecdot(a, b)
        # 1*3 m*s + 2*4 km*s = 3 + 8000 = 8003 m*s
        assert jnp.isclose(d.value, 8003.0)
        assert d.unit == _m * _s

    def test_vecdot_batched(self):
        """Batched vector dot product."""
        a = QMat(jnp.array([[1.0, 2.0], [1.0, 2.0]]), unit=(_m, _km))
        b = QMat(jnp.array([[3.0, 4.0], [3.0, 4.0]]), unit=(_s, _s))
        d = vecdot(a, b)
        assert d.value.shape == (2,)
        assert jnp.allclose(d.value, jnp.array([8003.0, 8003.0]))

    def test_check_contract_raises_on_mismatch(self):
        """Contraction validation raises ValueError (not a strippable assert)."""
        _check_contract(3, 3)  # ok
        with pytest.raises(ValueError, match="contraction mismatch"):
            _check_contract(2, 3)

    def test_wrap_operand_rejects_unsupported_rank(self):
        """The operand-wrapper raises a clear error for non vector/matrix ranks."""
        # logical ndim 3 (no batch axes) -> not a vector or matrix
        with pytest.raises(NotImplementedError, match=r"vector .* or matrix"):
            _wrap_operand(jnp.ones((2, 2, 2)), _dimless, ())

    def test_nonstandard_dot_general_rejected(self):
        """A non-matmul dot_general contraction is rejected, not mis-propagated."""
        A = QMat(jnp.array([[1.0, 2.0], [3.0, 4.0]]), unit=((_m, _m), (_m, _m)))
        B = QMat(jnp.array([[1.0, 0.0], [0.0, 1.0]]), unit=((_s, _s), (_s, _s)))
        # Contract the *first* axes (not the standard last/second-last).
        bad = quax.quaxify(lambda a, b: jnp.tensordot(a, b, axes=([0], [0])))
        with pytest.raises(NotImplementedError, match="standard"):
            bad(A, B)

    def test_vecmat_unit_conversion(self):
        """Vecmat converts mixed units within each output column."""
        v = QMat(jnp.array([1.0, 1.0]), unit=(_kg, _kg))
        # Column 0 mixes m (row 0) and km (row 1) -> km converted to m.
        a = QMat(jnp.array([[1.0, 2.0], [3.0, 4.0]]), unit=((_m, _s), (_km, _s)))
        w = vecmat(v, a)
        # col 0: 1*1 kg*m + 1*3 kg*km -> 1 + 3000 = 3001 (in kg*m)
        # col 1: 1*2 kg*s + 1*4 kg*s = 6 (in kg*s)
        assert jnp.allclose(w.value, jnp.array([3001.0, 6.0]))
        assert w.unit == (_kg * _m, _kg * _s)

    def test_matvec_under_transforms(self):
        """Matvec composes with jit / grad / vmap (values + units preserved)."""
        A = QMat(jnp.array([[1.0, 2.0], [3.0, 4.0]]), unit=((_m, _m), (_m, _m)))
        v = QMat(jnp.array([1.0, 1.0]), unit=(_s, _s))

        r = jax.jit(matvec)(A, v)
        assert jnp.allclose(r.value, jnp.array([3.0, 7.0]))
        assert r.unit == (_m * _s, _m * _s)

        def loss(mat_val):
            AA = QMat(mat_val, unit=((_m, _m), (_m, _m)))
            return jnp.sum(matvec(AA, v).value)

        # d/dA_ij of sum(A @ [1, 1]) is 1 everywhere.
        assert jnp.allclose(jax.grad(loss)(A.value), jnp.ones((2, 2)))

        Ab = QMat(jnp.stack([A.value, 2 * A.value]), unit=((_m, _m), (_m, _m)))
        vb = QMat(jnp.stack([v.value, v.value]), unit=(_s, _s))
        rv = jax.vmap(matvec)(Ab, vb)
        assert jnp.allclose(rv.value, jnp.array([[3.0, 7.0], [6.0, 14.0]]))

    def test_matmul_under_transforms(self):
        """Matmul composes with jit / grad / vmap (values + units preserved)."""
        A = QMat(jnp.array([[1.0, 2.0], [3.0, 4.0]]), unit=((_m, _m), (_m, _m)))
        B = QMat(jnp.array([[5.0, 6.0], [7.0, 8.0]]), unit=((_s, _s), (_s, _s)))

        r = jax.jit(matmul)(A, B)
        assert jnp.allclose(r.value, jnp.array([[19.0, 22.0], [43.0, 50.0]]))
        assert r.unit == ((_m * _s, _m * _s), (_m * _s, _m * _s))

        def loss(mat_val):
            return jnp.sum(matmul(QMat(mat_val, unit=((_m, _m), (_m, _m))), B).value)

        # d/dA_ik of sum(A @ B) is the row-sum of B over k: B row 0 = 11, row 1 = 15.
        assert jnp.allclose(
            jax.grad(loss)(A.value), jnp.array([[11.0, 15.0], [11.0, 15.0]])
        )

        Ab = QMat(jnp.stack([A.value, 2 * A.value]), unit=((_m, _m), (_m, _m)))
        Bb = QMat(jnp.stack([B.value, B.value]), unit=((_s, _s), (_s, _s)))
        assert jax.vmap(matmul)(Ab, Bb).value.shape == (2, 2, 2)

    def test_vecmat_under_transforms(self):
        """Vecmat composes with jit / grad / vmap."""
        v = QMat(jnp.array([1.0, 1.0]), unit=(_s, _s))
        a = QMat(jnp.array([[1.0, 2.0], [3.0, 4.0]]), unit=((_m, _m), (_m, _m)))

        r = jax.jit(vecmat)(v, a)
        assert jnp.allclose(r.value, jnp.array([4.0, 6.0]))
        assert r.unit == (_s * _m, _s * _m)

        def loss(vec_val):
            return jnp.sum(vecmat(QMat(vec_val, unit=(_s, _s)), a).value)

        # d/dv_i of sum(v @ a) is the row sums of a: [3, 7].
        assert jnp.allclose(jax.grad(loss)(v.value), jnp.array([3.0, 7.0]))

        vb = QMat(jnp.stack([v.value, v.value]), unit=(_s, _s))
        ab = QMat(jnp.stack([a.value, a.value]), unit=((_m, _m), (_m, _m)))
        assert jax.vmap(vecmat)(vb, ab).value.shape == (2, 2)

    def test_vecdot_under_transforms(self):
        """Vecdot composes with jit / grad / vmap (unit conversion preserved)."""
        a = QMat(jnp.array([1.0, 2.0]), unit=(_m, _km))
        b = QMat(jnp.array([3.0, 4.0]), unit=(_s, _s))

        r = jax.jit(vecdot)(a, b)
        assert jnp.isclose(r.value, 8003.0)
        assert r.unit == _m * _s

        # d/da_i of (a·b) carries the km→m conversion: [3, 4000] m·s.
        grad_a = jax.grad(lambda av: vecdot(QMat(av, unit=(_m, _km)), b).value)(a.value)
        assert jnp.allclose(grad_a, jnp.array([3.0, 4000.0]))

        ab = QMat(jnp.stack([a.value, a.value]), unit=(_m, _km))
        bb = QMat(jnp.stack([b.value, b.value]), unit=(_s, _s))
        assert jax.vmap(vecdot)(ab, bb).value.shape == (2,)

    def test_products_preserve_operand_dtype(self):
        """Unit-conversion scale factors must not upcast the result dtype.

        ``uconvert_value`` returns Python floats, so a naive ``jnp.array`` of
        the scales is float64 under ``jax_enable_x64``; casting the scales to
        the operand result dtype avoids surprise promotion. Tested here with
        float16 inputs (the same upcast mechanism, but visible under x32).
        """
        f16 = lambda x: jnp.asarray(x, jnp.float16)
        a = QMat(f16([1.0, 2.0]), unit=(_m, _km))
        b = QMat(f16([3.0, 4.0]), unit=(_s, _s))
        A = QMat(f16([[1.0, 2.0], [3.0, 4.0]]), unit=((_m, _km), (_m, _km)))
        v = QMat(f16([1.0, 1.0]), unit=(_s, _s))
        B = QMat(f16([[1.0, 2.0], [3.0, 4.0]]), unit=((_s, _s), (_s, _s)))
        assert vecdot(a, b).value.dtype == jnp.float16
        assert matvec(A, v).value.dtype == jnp.float16
        assert vecmat(v, A).value.dtype == jnp.float16
        assert matmul(A, B).value.dtype == jnp.float16

    def test_batched_plain_and_quantity_operands(self):
        """A *batched* plain / Quantity vector is not mis-classified as a matrix."""
        A = QMat(jnp.ones((2, 2, 2)), unit=((_m, _m), (_m, _m)))
        vvals = jnp.array([[0.0, 1.0], [2.0, 3.0]])  # (B=2, K=2) batched vector
        ref = matvec(A, QMat(vvals, unit=(_dimless, _dimless))).value

        # plain array on the right (dot_general_qm_arr)
        assert jnp.allclose(matvec(A, vvals).value, ref)
        # Quantity on the right (dot_general_qm_qty)
        w_qty = matvec(A, u.Q(vvals, "kg"))
        assert w_qty.value.shape == (2, 2)
        assert w_qty.unit == (_m * _kg, _m * _kg)
        # plain array on the left, vecmat (dot_general_arr_qm) — was silently (2,2,2)
        assert vecmat(jnp.ones((2, 2)), A).value.shape == (2, 2)


# ---------------------------------------------------------------------------
# The `@` operator (QuantityMatrix.__matmul__)
# ---------------------------------------------------------------------------


class TestMatmulOperator:
    """`@` dispatches on logical rank (NumPy semantics) and is batch-safe."""

    def test_at_batched_matvec(self):
        """A @ v with leading batch axes does matvec — matmul cannot."""
        A = QMat(
            jnp.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
            unit=((_m, _m), (_m, _m)),
        )
        v = QMat(jnp.ones((2, 2)), unit=(_s, _s))
        w = A @ v  # eager, no explicit quaxify needed
        assert jnp.allclose(w.value, jnp.array([[3.0, 7.0], [11.0, 15.0]]))
        assert w.unit == (_m * _s, _m * _s)
        assert jnp.allclose(w.value, matvec(A, v).value)

    def test_at_nonsquare_batched_matvec(self):
        """The non-square batch that ``matmul`` rejects works via ``@``."""
        A = QMat(jnp.ones((2, 2, 3)), unit=((_m, _m, _m), (_m, _m, _m)))
        v = QMat(jnp.ones((2, 3)), unit=(_s, _s, _s))
        assert (A @ v).shape == (2, 2)

    def test_at_dispatch_by_rank(self):
        """2D@2D→matmul, 1D@2D→vecmat, 1D@1D→vecdot (scalar)."""
        A = QMat(jnp.array([[1.0, 2.0], [3.0, 4.0]]), unit=((_m, _m), (_m, _m)))
        B = QMat(jnp.eye(2), unit=((_s, _s), (_s, _s)))
        assert (A @ B).ndim == 2  # matmul
        v = QMat(jnp.array([1.0, 1.0]), unit=(_s, _s))
        assert (v @ A).ndim == 1  # vecmat
        a = QMat(jnp.array([1.0, 2.0]), unit=(_m, _km))
        b = QMat(jnp.array([3.0, 4.0]), unit=(_s, _s))
        assert jnp.isclose((a @ b).value, 8003.0)  # vecdot → scalar Quantity

    def test_at_fallback_plain_array(self):
        """QM @ plain array falls back to default handling (still works)."""
        A = QMat(jnp.array([[1.0, 2.0], [3.0, 4.0]]), unit=((_m, _m), (_m, _m)))
        assert (A @ jnp.array([1.0, 1.0])).value.shape == (2,)


# ---------------------------------------------------------------------------
# JAX integration
# ---------------------------------------------------------------------------


class TestJaxIntegration:
    """QuantityMatrix works with JAX transformations."""

    def test_jit_add(self, unit_2x2):
        """jit-compiled addition."""
        a = QMat(jnp.array([[1, 2], [3, 4]]), unit=unit_2x2)
        b = QMat(jnp.array([[5, 6], [7, 8]]), unit=unit_2x2)
        result = jax.jit(_add)(a, b)
        expected = jnp.array([[6, 8], [10, 12]])
        assert jnp.allclose(result.value, expected)

    def test_jit_matmul(self):
        """jit-compiled matmul."""
        a = QMat(jnp.array([[2, 3]]), unit=((_m, _m),))
        b = QMat(jnp.array([[4], [5]]), unit=((_s,), (_s,)))
        result = jax.jit(_matmul)(a, b)
        assert jnp.isclose(result.value[0, 0], 23)

    def test_pytree_flatten_unflatten(self, qm_2x2, unit_2x2):
        """QuantityMatrix is a proper PyTree."""
        leaves, treedef = jax.tree.flatten(qm_2x2)
        restored = jax.tree.unflatten(treedef, leaves)
        assert jnp.array_equal(restored.value, qm_2x2.value)
        assert restored.unit == unit_2x2

    def test_vmap_add(self, unit_2x2):
        """Vmap over batch dimension for addition."""
        a = QMat(jnp.ones((4, 2, 2)), unit=unit_2x2)
        b = QMat(2 * jnp.ones((4, 2, 2)), unit=unit_2x2)

        @quax.quaxify
        def add_batched(x, y):
            return x + y

        result = jax.vmap(add_batched)(a, b)
        assert result.shape == (4, 2, 2)
        assert jnp.allclose(result.value, 3 * jnp.ones((4, 2, 2)))


# ---------------------------------------------------------------------------
# plum.convert registration
# ---------------------------------------------------------------------------


class TestPlumConversion:
    """Tests for ``plum.convert`` registrations involving ``QuantityMatrix``."""

    def test_QuantityMatrix_to_quantity_uniform_1d(self):
        """1D uniform-unit ``QuantityMatrix`` converts to ``u.Q``."""
        qm = QMat(value=jnp.array([1, 2, 3]), unit=(_m, _m, _m))

        result = plum.convert(qm, u.Q)

        assert isinstance(result, u.Q)
        assert result.unit == _m
        assert jnp.array_equal(result.value, qm.value)

    def test_QuantityMatrix_to_quantity_uniform_2d(self):
        """2D uniform-unit ``QuantityMatrix`` converts to ``u.Q``."""
        qm = QMat(value=jnp.array([[1, 2], [3, 4]]), unit=((_s, _s), (_s, _s)))

        result = plum.convert(qm, u.Q)

        assert isinstance(result, u.Q)
        assert result.unit == _s
        assert result.shape == (2, 2)
        assert jnp.array_equal(result.value, qm.value)

    def test_QuantityMatrix_to_quantity_heterogeneous_units_raises(self):
        """Mixed units cannot be converted to a single ``u.Q``."""
        qm = QMat(value=jnp.array([1.0, 2.0]), unit=(_m, _s))

        with pytest.raises(ValueError, match="all units are identical"):
            plum.convert(qm, u.Q)


# ---------------------------------------------------------------------------
# Affine / logarithmic product-unit guards
# ---------------------------------------------------------------------------


class TestAffineProductUnitsRejected:
    """Verify that astropy rejects product conversions for affine units.

    The ``dot_general_qm_qm`` implementation uses a multiplicative scale
    factor (``scale_3d``).  This is correct because affine units (°C, °F)
    are the only units where a multiplicative scale would be wrong (they
    have an additive offset), and astropy rejects product conversions
    involving them.  Logarithmic units (dex, mag) in products become
    plain ``CompositeUnit`` objects whose conversion IS multiplicative.

    If astropy ever starts accepting affine product conversions, these
    tests will *fail* — that's intentional: it means the assumption
    behind ``scale_3d`` must be revisited.

    The astropy-level checks below pin the *premise*; ``test_real_matmul_*``
    pins the *conclusion* by driving a real ``QuantityMatrix @ QuantityMatrix``
    through the actual dot-general code path.
    """

    def test_real_matmul_affine_units_raises(self):
        """A real QM @ QM whose contraction needs an affine conversion raises.

        The contraction reference unit is ``°C·s`` (k=0); the ``°F·s`` term
        (k=1) must convert to it, which astropy rejects — so the whole
        product raises rather than silently mis-scaling.
        """
        degC, degF = apu.deg_C, imperial.deg_F
        a = QMat(jnp.array([[1.0, 1.0]]), unit=((degC, degF),))
        b = QMat(jnp.array([[1.0], [1.0]]), unit=((_s,), (_s,)))
        with pytest.raises(apu.UnitConversionError, match="not convertible"):
            _matmul(a, b)

    def test_degC_times_s_not_convertible(self):
        """°C·s → °F·s must fail (affine offset is undefined for products)."""
        with pytest.raises(apu.UnitConversionError, match="not convertible"):
            (apu.deg_C * apu.s).to(imperial.deg_F * apu.s, 1.0)

    def test_degF_times_s_not_convertible(self):
        """°F·s → °C·s must fail (symmetric check)."""
        with pytest.raises(apu.UnitConversionError, match="not convertible"):
            (imperial.deg_F * apu.s).to(apu.deg_C * apu.s, 1.0)

    def test_degC_times_m_not_convertible(self):
        """°C·m → °F·m must fail."""
        with pytest.raises(apu.UnitConversionError, match="not convertible"):
            (apu.deg_C * apu.m).to(imperial.deg_F * apu.m, 1.0)

    def test_dex_times_s_is_convertible(self):
        """dex·s → dex·ms succeeds.

        dex in a product is a plain CompositeUnit; only the s → ms part
        converts, multiplicatively.
        """
        result = (apu.dex() * apu.s).to(apu.dex() * apu.ms, 1.0)
        assert math.isclose(result, 1000.0, rel_tol=1e-12)

    def test_mag_times_s_is_convertible(self):
        """mag·s → mag·ms succeeds (same reasoning as dex)."""
        result = (apu.mag() * apu.s).to(apu.mag() * apu.ms, 1.0)
        assert math.isclose(result, 1000.0, rel_tol=1e-12)

    def test_kelvin_times_s_is_convertible(self):
        """K·s → K·ms MUST succeed (Kelvin is absolute / linear).

        This is the *positive* control: linear temperature products work
        fine and the multiplicative scale is correct.
        """
        result = (apu.K * apu.s).to(apu.K * apu.ms, 1.0)
        assert math.isclose(result, 1000.0, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# Matrix-vector multiply  (_dot_general_2d_1d)
# ---------------------------------------------------------------------------


class TestMatVec:
    """Tests for 2D `QuantityMatrix` @ 1D `QuantityMatrix`."""

    def test_identity_uniform_units(self):
        """Identity 3x3 @ uniform-unit vector → same vector."""
        A = QMat(jnp.eye(3), unit=((_dimless, _dimless, _dimless),) * 3)
        v = QMat(jnp.array([1, 2, 3]), unit=(_m, _m, _m))
        w = _matmul(A, v)
        assert isinstance(w, QMat)
        assert w.ndim == 1
        assert jnp.allclose(w.value, jnp.array([1, 2, 3]))

    def test_uniform_units_values(self):
        """2x2 @ 2: correct values with uniform units."""
        A = QMat(jnp.array([[2, 3], [4, 5]]), unit=((_m, _m), (_m, _m)))
        v = QMat(jnp.array([10, 20]), unit=(_s, _s))
        w = _matmul(A, v)
        # [2*10 + 3*20, 4*10 + 5*20] = [80, 140]
        assert jnp.isclose(w.value[0], 80)
        assert jnp.isclose(w.value[1], 140)

    def test_output_is_1d_QuantityMatrix(self):
        """Result of 2D @ 1D is a 1D ``QuantityMatrix``, not a 2D one."""
        A = QMat(jnp.array([[1, 0], [0, 1]]), unit=((_m, _m), (_m, _m)))
        v = QMat(jnp.array([3, 7]), unit=(_s, _s))
        w = _matmul(A, v)
        assert isinstance(w, QMat)
        assert w.ndim == 1
        assert w.shape == (2,)

    def test_output_units_are_product(self):
        """Output unit[i] == lhs.unit[i][0] * rhs.unit[0]."""
        A = QMat(jnp.array([[1, 1], [1, 1]]), unit=((_m, _m), (_kg, _kg)))
        v = QMat(jnp.array([1, 1]), unit=(_s, _s))
        w = _matmul(A, v)
        assert w.unit[0] == _m * _s
        assert w.unit[1] == _kg * _s

    def test_lhs_unit_conversion_on_contraction_axis(self):
        """Km column in A is converted to m before summing."""
        # A: 2x2 with top-row units [[m, km]], bottom [[m, km]]
        # v: [1, 1] in [s, s]
        # ref[i] = m*s, scale[i,1] = 1000 (km*s → m*s)
        # w[0] = 1*1 + 1000*1 = 1001, w[1] = 1001
        A = QMat(jnp.array([[1, 1], [1, 1]]), unit=((_m, _km), (_m, _km)))
        v = QMat(jnp.array([1, 1]), unit=(_s, _s))
        w = _matmul(A, v)
        assert jnp.isclose(w.value[0], 1001)
        assert jnp.isclose(w.value[1], 1001)
        assert w.unit[0] == _m * _s
        assert w.unit[1] == _m * _s

    def test_rhs_unit_conversion_on_contraction_axis(self):
        """Ms column in v is converted to s before summing."""
        # A: 2x2 with units [[m, m], [m, m]]
        # v: [1, 1] in [s, ms]
        # ref[i] = m*s, scale[i,1] = uconvert_value(m*s, m*ms, 1.0) = 0.001
        # w[0] = 1*1 + 0.001*1 = 1.001, w[1] = 1.001
        A = QMat(jnp.array([[1, 1], [1, 1]]), unit=((_m, _m), (_m, _m)))
        v = QMat(jnp.array([1, 1]), unit=(_s, _ms))
        w = _matmul(A, v)
        assert jnp.isclose(w.value[0], 1.001)
        assert jnp.isclose(w.value[1], 1.001)
        assert w.unit[0] == _m * _s

    def test_non_square_3x2_at_2(self):
        """Non-square 3x2 @ 2 → 3."""
        A = QMat(
            jnp.array([[1, 2], [3, 4], [5, 6]]), unit=((_m, _km), (_m, _km), (_m, _km))
        )
        v = QMat(jnp.array([1, 1]), unit=(_s, _s))
        w = _matmul(A, v)
        # scale[i,1] = 1000 (km*s → m*s)
        # w[0] = 1*1 + 1000*2*1 = 2001
        # w[1] = 1*3 + 1000*4*1 = 4003
        # w[2] = 1*5 + 1000*6*1 = 6005
        assert w.shape == (3,)
        assert jnp.isclose(w.value[0], 2001)
        assert jnp.isclose(w.value[1], 4003)
        assert jnp.isclose(w.value[2], 6005)
        assert all(w.unit[i] == _m * _s for i in range(3))

    def test_jit_compatible(self):
        """jit-compiled matrix-vector multiply works."""
        A = QMat(jnp.array([[2, 3], [4, 5]]), unit=((_m, _m), (_m, _m)))
        v = QMat(jnp.array([10, 20]), unit=(_s, _s))
        w = jax.jit(_matmul)(A, v)
        assert jnp.isclose(w.value[0], 80)
        assert jnp.isclose(w.value[1], 140)

    def test_batch_dimensions(self):
        """Leading batch dimensions are preserved."""
        # A: (3, 2, 2), v: (3, 2) — vmapped over batch dim
        A = QMat(jnp.ones((3, 2, 2)), unit=((_m, _m), (_m, _m)))
        v = QMat(jnp.ones((3, 2)), unit=(_s, _s))

        @quax.quaxify
        def mv(a, b):
            return a @ b

        w = jax.vmap(mv)(A, v)
        # Each 2x2 ones @ [1, 1] = [2, 2]
        assert w.shape == (3, 2)
        assert jnp.allclose(w.value, 2 * jnp.ones((3, 2)))

    def test_different_per_row_output_units(self):
        """Each output row can have a different unit."""
        # A: row 0 in m, row 1 in kg; v in s
        A = QMat(jnp.array([[1, 1], [1, 1]]), unit=((_m, _m), (_kg, _kg)))
        v = QMat(jnp.array([1, 1]), unit=(_s, _s))
        w = _matmul(A, v)
        assert w.unit[0] == _m * _s
        assert w.unit[1] == _kg * _s
        assert jnp.isclose(w.value[0], 2)
        assert jnp.isclose(w.value[1], 2)


# ---------------------------------------------------------------------------
# Gather (jnp.diag)
# ---------------------------------------------------------------------------


@quax.quaxify
def _diag(a):
    return jnp.diag(a)


class TestDiagAndGather:
    """Tests for gather_qm: covers jnp.diag and element-selection indexing."""

    def test_diag_values(self):
        """jnp.diag extracts the correct diagonal values."""
        A = QMat(
            jnp.diag(jnp.array([1, 4, 9])),
            unit=(("m", "m", "m"), ("m", "m", "m"), ("m", "m", "m")),
        )
        d = _diag(A)
        assert isinstance(d, QMat)
        assert jnp.allclose(d.value, jnp.array([1, 4, 9]))

    def test_diag_unit_is_1d(self):
        """jnp.diag result has a 1-D UnitsMatrix, not 2-D."""
        A = QMat(
            jnp.diag(jnp.array([1, 4, 9])),
            unit=(("m", "m", "m"), ("m", "m", "m"), ("m", "m", "m")),
        )
        d = _diag(A)
        assert d.unit.ndim == 1
        assert d.unit.shape == (3,)

    def test_diag_uniform_units(self):
        """Diagonal of uniform-unit matrix keeps that unit."""
        A = QMat(
            jnp.diag(jnp.array([1, 2, 3])),
            unit=((_m, _m, _m), (_m, _m, _m), (_m, _m, _m)),
        )
        d = _diag(A)
        assert all(d.unit[i] == _m for i in range(3))

    def test_diag_heterogeneous_units_picks_diagonal(self):
        """gather_p with concrete indices extracts correct units from heterogeneous QM.

        jnp.diag is internally @jit-decorated, which traces indices abstract under
        quax.quaxify.  Direct fancy indexing (without the @jit layer) demonstrates
        that our gather_p handler correctly extracts per-element units when indices
        are concrete Python/NumPy values.
        """
        A = QMat(
            jnp.diag(jnp.array([1, 2, 3])),
            unit=((_m, _s, _kg), (_m, _s, _kg), (_m, _s, _kg)),
        )

        @quax.quaxify
        def _fancy_diag(mat):
            # Direct fancy indexing: indices are concrete (no @jit layer)
            return mat[jnp.array([0, 1, 2]), jnp.array([0, 1, 2])]

        d = _fancy_diag(A)
        assert d.unit[0] == _m  # unit[0, 0]
        assert d.unit[1] == _s  # unit[1, 1]
        assert d.unit[2] == _kg  # unit[2, 2]

    def test_multidim_advanced_indexing_units(self):
        """2-D index arrays give a 2-D output whose units match element-by-element.

        JAX packs the indices as ``(*index_shape, index_vector_dim)``; the unit
        lookup must index the last axis (``idx[..., k]``) and carry the full
        ``index_shape``, not just its first entry.
        """
        A = QMat(jnp.arange(4.0).reshape(2, 2), unit=((_m, _s), (_kg, _rad)))
        ii = jnp.array([[0, 1], [1, 0]])
        jj = jnp.array([[0, 0], [1, 1]])

        r = quax.quaxify(lambda a: a[ii, jj])(A)
        # value[a,b] = A.value[ii[a,b], jj[a,b]]
        assert jnp.allclose(r.value, jnp.array([[0.0, 2.0], [3.0, 1.0]]))
        # unit[a,b] = A.unit[ii[a,b], jj[a,b]]
        assert r.unit.shape == (2, 2)
        assert r.unit[0, 0] == _m  # (0,0)
        assert r.unit[0, 1] == _kg  # (1,0)
        assert r.unit[1, 0] == _rad  # (1,1)
        assert r.unit[1, 1] == _s  # (0,1)

    def test_2d_index_into_1d_quantity_matrix(self):
        """A 2-D index array on a 1-D QuantityMatrix yields a 2-D unit structure."""
        v = QMat(jnp.arange(4.0), unit=(_m, _s, _kg, _rad))
        r = quax.quaxify(lambda a: a[jnp.array([[0, 1], [2, 3]])])(v)
        assert jnp.allclose(r.value, jnp.array([[0.0, 1.0], [2.0, 3.0]]))
        assert r.unit.shape == (2, 2)
        assert r.unit[0, 0] == _m
        assert r.unit[1, 1] == _rad

    def test_scalar_output_gather_returns_quantity(self):
        """A scalar-output gather (index_shape == ()) returns a scalar Quantity.

        A single element has a single unit; UnitsMatrix only represents 1-D/2-D
        structures, so the handler returns a plain Quantity for scalar output.
        """
        A = QMat(jnp.arange(4.0).reshape(2, 2), unit=((_m, _s), (_kg, _rad)))
        dnums = lax.GatherDimensionNumbers(
            offset_dims=(), collapsed_slice_dims=(0, 1), start_index_map=(0, 1)
        )
        # select element (1, 0) -> value 2.0, unit kg
        r = gather_qm(A, jnp.array([1, 0]), dimension_numbers=dnums, slice_sizes=(1, 1))
        assert isinstance(r, u.AbstractQuantity)
        assert not isinstance(r, QMat)
        assert jnp.allclose(r.value, 2.0)
        assert r.unit == _kg
        # 1-D QuantityMatrix scalar select
        v = QMat(jnp.arange(3.0), unit=(_m, _s, _kg))
        dn1 = lax.GatherDimensionNumbers(
            offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)
        )
        r1 = gather_qm(v, jnp.array([2]), dimension_numbers=dn1, slice_sizes=(1,))
        assert isinstance(r1, u.AbstractQuantity)
        assert not isinstance(r1, QMat)
        assert jnp.allclose(r1.value, 2.0)
        assert r1.unit == _kg

    def test_diag_dimensionless_unit_string(self):
        """Dimensionless diagonal has '(, , )'-style repr, not '((, , ), ...)'."""
        A = QMat(
            jnp.diag(jnp.array([1, 4, 4])),
            unit=(("", "", ""), ("", "", ""), ("", "", "")),
        )
        d = _diag(A)
        assert d.unit.ndim == 1
        assert d.unit.shape == (3,)
        # The string should be 1D style like '(, , )' not nested '((, , ), ...)',
        # i.e. exactly one opening parenthesis.
        s = d.unit.to_string()
        assert s.count("(") == 1, f"Expected 1D unit string, got: {s!r}"

    def test_diag_under_jit_uniform_units(self):
        """jnp.diag under jit works for uniform-unit QuantityMatrix."""
        A = QMat(
            jnp.diag(jnp.array([1, 4, 9])),
            unit=((_m, _m, _m), (_m, _m, _m), (_m, _m, _m)),
        )
        d = jax.jit(_diag)(A)
        assert isinstance(d, QMat)
        assert jnp.allclose(d.value, jnp.array([1, 4, 9]))
        assert d.unit.ndim == 1
        assert all(d.unit[i] == _m for i in range(3))

    def test_diag_under_jit_heterogeneous_units_rejected(self):
        """qnp.diag under jit cannot resolve traced indices for mixed units."""
        A = QMat(jnp.eye(2), unit=((_m, _s), (_m, _s)))
        with pytest.raises(ValueError, match="units to be equal"):
            jax.jit(_diag)(A)

    def test_gather_non_element_selection_rejected(self):
        """A window (non element-selection) gather is rejected, not mislabelled."""
        qm = QMat(jnp.arange(8.0).reshape(4, 2), unit=(_m, _m))
        # A slice/window gather: offset_dims is non-empty (selects full rows).
        dnums = lax.GatherDimensionNumbers(
            offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,)
        )
        with pytest.raises(NotImplementedError, match="element-selection"):
            gather_qm(
                qm,
                jnp.array([[0], [2]]),
                dimension_numbers=dnums,
                slice_sizes=(1, 2),
            )


# ---------------------------------------------------------------------------
# QuantityMatrix.diag() method
# ---------------------------------------------------------------------------


class TestDiagMethod:
    """Tests for ``QuantityMatrix.diag()`` — the method that bypasses JAX gather."""

    def test_uniform_units_values(self):
        """Diagonal values are correct for a uniform-unit matrix."""
        A = QMat(
            jnp.diag(jnp.array([1, 4, 9])),
            unit=((_m, _m, _m), (_m, _m, _m), (_m, _m, _m)),
        )
        d = A.diag()
        assert isinstance(d, QMat)
        assert jnp.allclose(d.value, jnp.array([1, 4, 9]))

    def test_uniform_units_shape(self):
        """Result is 1-D with length equal to the diagonal."""
        A = QMat(
            jnp.diag(jnp.array([1, 4, 9])),
            unit=((_m, _m, _m), (_m, _m, _m), (_m, _m, _m)),
        )
        d = A.diag()
        assert d.ndim == 1
        assert d.shape == (3,)
        assert d.unit.ndim == 1
        assert d.unit.shape == (3,)

    def test_uniform_units_preserved(self):
        """Units on the diagonal are preserved unchanged."""
        A = QMat(
            jnp.diag(jnp.array([1, 2, 3])),
            unit=((_m, _m, _m), (_m, _m, _m), (_m, _m, _m)),
        )
        d = A.diag()
        assert all(d.unit[i] == _m for i in range(3))

    def test_heterogeneous_units_values(self):
        """Diagonal values are correct for a heterogeneous-unit matrix."""
        A = QMat(
            jnp.diag(jnp.array([1, 2, 3])),
            unit=((_m, _s, _kg), (_m, _s, _kg), (_m, _s, _kg)),
        )
        d = A.diag()
        assert jnp.allclose(d.value, jnp.array([1, 2, 3]))

    def test_heterogeneous_units_correct(self):
        """Each diagonal unit is taken from ``self.unit[i, i]``."""
        A = QMat(
            jnp.diag(jnp.array([1, 2, 3])),
            unit=((_m, _s, _kg), (_m, _s, _kg), (_m, _s, _kg)),
        )
        d = A.diag()
        assert d.unit[0] == _m
        assert d.unit[1] == _s
        assert d.unit[2] == _kg

    def test_heterogeneous_units_to_string(self):
        """1-D unit string has the correct format."""
        A = QMat(
            jnp.diag(jnp.array([1, 2, 3])),
            unit=((_m, _s, _kg), (_m, _s, _kg), (_m, _s, _kg)),
        )
        d = A.diag()
        assert d.unit.to_string() == "(m, s, kg)"

    def test_2x2_square(self):
        """2x2 matrix diagonal."""
        A = QMat(jnp.array([[5, 0], [0, 7]]), unit=((_m, _s), (_kg, _rad)))
        d = A.diag()
        assert d.shape == (2,)
        assert jnp.isclose(d.value[0], 5)
        assert jnp.isclose(d.value[1], 7)
        assert d.unit[0] == _m
        assert d.unit[1] == _rad

    def test_non_square_picks_min_dim(self):
        """For non-square matrices the diagonal length is min(rows, cols)."""
        # 2x3 matrix → diagonal of length 2
        A = QMat(jnp.arange(6).reshape(2, 3), unit=((_m, _s, _kg), (_rad, _km, _ms)))
        d = A.diag()
        assert d.shape == (2,)
        assert jnp.isclose(d.value[0], 0)  # A[0,0]
        assert jnp.isclose(d.value[1], 4)  # A[1,1]
        assert d.unit[0] == _m  # unit[0,0]
        assert d.unit[1] == _km  # unit[1,1]

    def test_1d_raises(self, qm_1d):
        """Calling .diag() on a 1-D QuantityMatrix raises ValueError."""
        with pytest.raises(ValueError, match="2D"):
            qm_1d.diag()

    def test_jit_uniform_units(self):
        """Works under jax.jit with uniform units."""
        A = QMat(
            jnp.diag(jnp.array([1, 4, 9])),
            unit=((_m, _m, _m), (_m, _m, _m), (_m, _m, _m)),
        )
        d = jax.jit(lambda x: x.diag())(A)
        assert isinstance(d, QMat)
        assert jnp.allclose(d.value, jnp.array([1, 4, 9]))
        assert d.unit[0] == _m

    def test_jit_heterogeneous_units(self):
        """Works under jax.jit with heterogeneous units (the key advantage)."""
        A = QMat(
            jnp.diag(jnp.array([1, 2, 3])),
            unit=((_m, _s, _kg), (_m, _s, _kg), (_m, _s, _kg)),
        )
        d = jax.jit(lambda x: x.diag())(A)
        assert d.unit[0] == _m
        assert d.unit[1] == _s
        assert d.unit[2] == _kg
        assert jnp.allclose(d.value, jnp.array([1, 2, 3]))

    def test_batch_dimensions(self):
        """Batch leading dimensions are preserved."""
        # (3, 2, 2) batched matrix
        base = jnp.diag(jnp.array([1, 4]))
        A = QMat(
            jnp.stack([base, 2 * base, 3 * base]),  # (3, 2, 2)
            unit=((_m, _m), (_s, _s)),
        )
        d = A.diag()
        assert d.shape == (3, 2)
        assert jnp.isclose(d.value[0, 0], 1)
        assert jnp.isclose(d.value[1, 0], 2)
        assert jnp.isclose(d.value[2, 1], 12)


# ---------------------------------------------------------------------------
# UnitsMatrix.inverse
# ---------------------------------------------------------------------------


class TestUnitsMatrixInverse:
    """Tests for ``UnitsMatrix.inverse`` — element-wise unit reciprocal."""

    def test_1d_values(self):
        """1-D: each unit is reciprocated."""
        um = UnitsMatrix((_m, _s))
        inv = um.inverse()
        assert inv[0] == _m ** (-1)
        assert inv[1] == _s ** (-1)

    def test_1d_shape_preserved(self):
        """1-D inverse has the same shape."""
        um = UnitsMatrix((_m, _s, _kg))
        assert um.inverse().shape == (3,)

    def test_1d_returns_unitsmatrix(self):
        """Result is a ``UnitsMatrix`` instance."""
        um = UnitsMatrix((_m, _s))
        assert isinstance(um.inverse(), UnitsMatrix)

    def test_1d_double_inverse_is_identity(self):
        """Two inversions return the original unit structure."""
        um = UnitsMatrix((_m, _s, _kg))
        assert um == um.inverse().inverse()

    def test_2d_uniform_values(self):
        """2-D uniform matrix: every entry is reciprocated."""
        um = UnitsMatrix(((_m, _m), (_m, _m)))
        inv = um.inverse()
        for i in range(2):
            for j in range(2):
                assert inv[i, j] == _m ** (-1)

    def test_2d_uniform_shape_preserved(self):
        """2-D uniform inverse has the same shape."""
        um = UnitsMatrix(((_m, _m), (_m, _m)))
        assert um.inverse().shape == (2, 2)

    def test_2d_mixed_values(self):
        """2-D mixed-unit matrix: element-wise reciprocal."""
        um = UnitsMatrix(((_m, _s), (_kg, _rad)))
        inv = um.inverse()
        assert inv[0, 0] == _m ** (-1)
        assert inv[0, 1] == _s ** (-1)
        assert inv[1, 0] == _kg ** (-1)
        assert inv[1, 1] == _rad ** (-1)

    def test_2d_returns_unitsmatrix(self):
        """Result is always a ``UnitsMatrix`` instance."""
        um = UnitsMatrix(((_m, _s), (_kg, _rad)))
        assert isinstance(um.inverse(), UnitsMatrix)

    def test_2d_double_inverse_is_identity(self):
        """Two inversions return the original unit structure."""
        um = UnitsMatrix(((_m, _s), (_kg, _rad)))
        assert um == um.inverse().inverse()


# ---------------------------------------------------------------------------
# UnitsMatrix.T
# ---------------------------------------------------------------------------


class TestUnitsMatrixTranspose:
    """Tests for ``UnitsMatrix.T`` — the unit-structure transpose."""

    def test_2d_square_values(self):
        """Transposing a 2x2 UnitsMatrix swaps rows and columns."""
        um = UnitsMatrix(((_m, _s), (_kg, _rad)))
        t = um.T
        assert t[0, 0] == _m
        assert t[0, 1] == _kg
        assert t[1, 0] == _s
        assert t[1, 1] == _rad

    def test_2d_square_shape(self):
        """Shape is preserved for a square transpose."""
        um = UnitsMatrix(((_m, _s), (_kg, _rad)))
        assert um.T.shape == (2, 2)

    def test_2d_nonsquare_shape(self):
        """Transposing a 2x3 UnitsMatrix gives a 3x2."""
        um = UnitsMatrix(((_m, _s, _kg), (_rad, _km, _ms)))
        t = um.T
        assert t.shape == (3, 2)

    def test_2d_nonsquare_values(self):
        """Each element [j, i] in the transpose equals [i, j] in the original."""
        um = UnitsMatrix(((_m, _s, _kg), (_rad, _km, _ms)))
        t = um.T
        # Original row 0: (m, s, kg); Original row 1: (rad, km, ms)
        # Transposed col 0 (row 0): (m, rad); col 1: (s, km); col 2: (kg, ms)
        assert t[0, 0] == _m
        assert t[0, 1] == _rad
        assert t[1, 0] == _s
        assert t[1, 1] == _km
        assert t[2, 0] == _kg
        assert t[2, 1] == _ms

    def test_2d_double_transpose_is_identity(self):
        """Two transposes return the original unit structure."""
        um = UnitsMatrix(((_m, _s), (_kg, _rad)))
        assert um == um.T.T

    def test_2d_returns_unitsmatrix(self):
        """Result is always a ``UnitsMatrix`` instance."""
        um = UnitsMatrix(((_m, _s), (_kg, _rad)))
        assert isinstance(um.T, UnitsMatrix)

    def test_1d_transpose_is_identity(self):
        """Transposing a 1-D UnitsMatrix is a no-op (numpy convention)."""
        um = UnitsMatrix((_m, _s, _kg))
        t = um.T
        assert t == um
        assert t.shape == (3,)
        assert t.ndim == 1


# ---------------------------------------------------------------------------
# QuantityMatrix.T
# ---------------------------------------------------------------------------


@quax.quaxify
def _transpose(x):
    return x.T


class TestQuantityMatrixTranspose:
    """Tests for ``QuantityMatrix.T`` — the matrix transpose property."""

    # -- Basic 2D values and units ----------------------------------------

    def test_2d_square_values(self, qm_2x2):
        """Value array is transposed correctly for a square matrix."""
        t = qm_2x2.T
        expected = jnp.array([[1, 3], [2, 4]])
        assert jnp.allclose(t.value, expected)

    def test_2d_square_units(self, qm_2x2):
        """Unit structure is transposed: unit[j][i] equals original unit[i][j]."""
        t = qm_2x2.T
        # Original: unit[0,0]=m, unit[0,1]=s, unit[1,0]=kg, unit[1,1]=rad
        assert t.unit[0, 0] == _m
        assert t.unit[0, 1] == _kg
        assert t.unit[1, 0] == _s
        assert t.unit[1, 1] == _rad

    def test_2d_square_unit_string(self, qm_2x2):
        """Transposed unit string matches expected layout."""
        assert qm_2x2.T.unit.to_string() == "((m, kg), (s, rad))"

    def test_2d_square_shape_preserved(self, qm_2x2):
        """Shape is unchanged for a square matrix."""
        assert qm_2x2.T.shape == (2, 2)

    def test_2d_square_returns_QuantityMatrix(self, qm_2x2):
        """Result is a ``QuantityMatrix`` instance."""
        assert isinstance(qm_2x2.T, QMat)

    def test_2d_nonsquare_values(self):
        """Transposing a 2x3 matrix gives a 3x2 with correct values."""
        a = QMat(jnp.arange(6).reshape(2, 3), unit=((_m, _s, _kg), (_rad, _km, _ms)))
        t = a.T
        expected = jnp.arange(6).reshape(2, 3).T
        assert jnp.allclose(t.value, expected)

    def test_2d_nonsquare_shape(self):
        """Shape is (3, 2) after transposing a (2, 3) matrix."""
        a = QMat(jnp.ones((2, 3)), unit=((_m, _s, _kg), (_rad, _km, _ms)))
        assert a.T.shape == (3, 2)

    def test_2d_nonsquare_units(self):
        """Unit element [j, i] of the transpose equals original [i, j]."""
        a = QMat(jnp.ones((2, 3)), unit=((_m, _s, _kg), (_rad, _km, _ms)))
        t = a.T
        # Original row 0: (m, s, kg); row 1: (rad, km, ms)
        assert t.unit[0, 0] == _m
        assert t.unit[0, 1] == _rad
        assert t.unit[1, 0] == _s
        assert t.unit[1, 1] == _km
        assert t.unit[2, 0] == _kg
        assert t.unit[2, 1] == _ms

    def test_2d_double_transpose_values(self, qm_2x2):
        """Transposing twice recovers the original values."""
        assert jnp.allclose(qm_2x2.T.T.value, qm_2x2.value)

    def test_2d_double_transpose_units(self, qm_2x2):
        """Transposing twice recovers the original unit structure."""
        assert qm_2x2.T.T.unit == qm_2x2.unit

    # -- Batch-dimension behavior -----------------------------------------

    def test_batch_shape(self):
        """For a batched ``(B, N, M)`` array, ``.T`` swaps only matrix axes.

        Result shape is ``(B, M, N)``.
        """
        a = QMat(jnp.ones((3, 2, 4)), unit=((_m, _s, _kg, _rad), (_km, _ms, _g, _deg)))
        t = a.T
        assert t.shape == (3, 4, 2)

    def test_batch_unit_structure_unchanged(self):
        """The unit structure (which is 2-D) is transposed in the usual way.

        Batch axes are preserved; only the last two (matrix) axes are swapped.
        The unit structure only has the two logical dimensions, so it is
        transposed as a normal 2-D matrix; the batch axis does not appear in
        the unit structure.
        """
        a = QMat(jnp.ones((3, 2, 2)), unit=((_m, _s), (_kg, _rad)))
        t = a.T
        # value shape: (3,2,2) → (3,2,2);  unit shape stays (2,2) but transposed
        assert t.shape == (3, 2, 2)
        assert t.unit.shape == (2, 2)
        assert t.unit[0, 0] == _m
        assert t.unit[0, 1] == _kg
        assert t.unit[1, 0] == _s
        assert t.unit[1, 1] == _rad

    def test_batch_matrix_transpose_via_quax(self):
        """``matrix_transpose`` on a batched ``(B, N, M)`` preserves batch axes."""
        a = QMat(jnp.arange(12).reshape(3, 2, 2), unit=((_m, _s), (_kg, _rad)))
        t = qnp.matrix_transpose(a)
        # Batch axis preserved; last two swapped: (3,2,2) → (3,2,2)
        assert t.shape == (3, 2, 2)
        # Check a few values: t[b, i, j] == a[b, j, i]
        for b in range(3):
            assert jnp.allclose(t.value[b], a.value[b].T)
        # Unit structure is the same transposed 2-D layout
        assert t.unit[0, 0] == _m
        assert t.unit[0, 1] == _kg
        assert t.unit[1, 0] == _s
        assert t.unit[1, 1] == _rad

    def test_vmap_transpose(self):
        """``jax.vmap`` over a batch of 2-D matrices gives the per-element transpose."""
        a = QMat(jnp.arange(12).reshape(3, 2, 2), unit=((_m, _s), (_kg, _rad)))

        @quax.quaxify
        def single_T(x):
            return x.T

        result = jax.vmap(single_T)(a)
        # Each (2,2) slice is transposed independently
        assert result.shape == (3, 2, 2)
        for i in range(3):
            expected = jnp.array(
                [
                    [a.value[i, 0, 0], a.value[i, 1, 0]],
                    [a.value[i, 0, 1], a.value[i, 1, 1]],
                ]
            )
            assert jnp.allclose(result.value[i], expected)

    # -- JIT compatibility ------------------------------------------------

    def test_jit_values(self, qm_2x2):
        """``jax.jit`` preserves transpose values."""
        t = jax.jit(_transpose)(qm_2x2)
        expected = jnp.array([[1, 3], [2, 4]])
        assert jnp.allclose(t.value, expected)

    def test_jit_units(self, qm_2x2):
        """``jax.jit`` preserves transposed unit structure."""
        t = jax.jit(_transpose)(qm_2x2)
        assert t.unit.to_string() == "((m, kg), (s, rad))"

    # -- 1-D error --------------------------------------------------------

    def test_1d_raises(self, qm_1d):
        """Accessing ``.T`` on a 1-D ``QuantityMatrix`` raises ``ValueError``.

        The ``.T`` property requires a 2-D unit structure to unpack ``(n, m)``.
        """
        with pytest.raises(ValueError, match="requires a 2-D matrix"):
            _ = qm_1d.T

    def test_nonstandard_permutation_rejected(self):
        """A transpose that reorders batch axes (not the last two) is rejected."""
        a = QMat(jnp.ones((3, 2, 2)), unit=((_m, _s), (_kg, _rad)))
        # swapaxes(0, 1) permutes a batch axis into the logical block.
        with pytest.raises(NotImplementedError, match="last two axes"):
            quax.quaxify(lambda x: jnp.swapaxes(x, 0, 1))(a)

    def test_identity_permutation_is_noop(self):
        """An identity permutation is a no-op that preserves value and units.

        Covers a 1-D vector (whose only permutation is ``(0,)``). ``jnp.transpose``
        elides such no-ops before they reach the handler, so drive the handler
        directly to confirm it does not raise on a valid identity transpose.
        """
        v = QMat(jnp.array([1.0, 2.0, 3.0]), unit=(_m, _s, _kg))
        r = transpose_qm(v, permutation=(0,))
        assert jnp.allclose(r.value, v.value)
        assert r.unit == (_m, _s, _kg)
        # jnp.transpose on a 1-D QuantityMatrix works end-to-end (no exception).
        r2 = quax.quaxify(lambda a: jnp.transpose(a))(v)
        assert jnp.allclose(r2.value, v.value)
        assert r2.unit == (_m, _s, _kg)


# ---------------------------------------------------------------------------
# det_p custom primitive + Quax dispatch for QuantityMatrix
# ---------------------------------------------------------------------------


class TestDetPrimitive:
    """Tests for the custom ``det_p`` JAX primitive on plain arrays."""

    def test_det_2x2_diagonal(self):
        """Det on a 2×2 diagonal matrix returns the product of the diagonals."""
        A = jnp.array([[2.0, 0.0], [0.0, 3.0]])
        assert jnp.allclose(qm_det(A), jnp.linalg.det(A))

    def test_det_3x3_identity(self):
        """det(I_3 * 2) == 8."""
        A = jnp.eye(3) * 2.0
        assert jnp.allclose(qm_det(A), 8.0)

    def test_det_matches_jnp_linalg_det(self):
        """det_p gives the same result as jnp.linalg.det for a generic matrix."""
        A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        assert jnp.allclose(qm_det(A), jnp.linalg.det(A))

    def test_det_jit(self):
        """det_p works under jax.jit."""
        A = jnp.array([[2.0, 0.0], [0.0, 3.0]])
        result = jax.jit(qm_det)(A)
        assert jnp.allclose(result, 6.0)

    def test_det_jvp(self):
        """Forward-mode derivative matches Jacobi's formula.

        d det(A)(dA) = det(A) * tr(A^-1 dA).
        """
        A = jnp.array([[2.0, 0.0], [0.0, 3.0]])
        dA = jnp.ones((2, 2))
        primal, tangent = jax.jvp(qm_det, (A,), (dA,))
        # det(A) = 6
        # tr(A⁻¹ dA) = tr([[0.5,0],[0,1/3]] @ [[1,1],[1,1]]) = 0.5 + 1/3 = 5/6
        # tangent = 6 * 5/6 = 5
        assert jnp.allclose(primal, 6.0)
        assert jnp.allclose(tangent, 5.0)

    def test_det_grad(self):
        """Reverse-mode gradient matches Jacobi's formula: ∂det(A)/∂A = det(A)·A⁻ᵀ."""
        A = jnp.array([[2.0, 0.0], [0.0, 3.0]])
        # grad_A = det(A) * A^{-T} = 6 * diag(0.5, 1/3) = diag(3, 2)
        grad_A = jax.grad(qm_det)(A)
        expected = jnp.array([[3.0, 0.0], [0.0, 2.0]])
        assert jnp.allclose(grad_A, expected)

    def test_det_jit_grad(self):
        """jit(grad(det)) works correctly."""
        A = jnp.array([[2.0, 0.0], [0.0, 3.0]])
        grad_A = jax.jit(jax.grad(qm_det))(A)
        expected = jnp.array([[3.0, 0.0], [0.0, 2.0]])
        assert jnp.allclose(grad_A, expected)

    def test_det_vmap(self):
        """det_p works under jax.vmap — maps over a batch of matrices."""
        A = jnp.stack(
            [jnp.diag(jnp.array([2.0, 3.0])), jnp.diag(jnp.array([4.0, 5.0]))]
        )
        results = jax.vmap(qm_det)(A)
        expected = jnp.array([6.0, 20.0])
        assert jnp.allclose(results, expected)

    def test_det_grad_batched(self):
        """Grad over a *batched* det sums matches jnp.linalg.det.

        The JVP traces the two matrix axes (not the leading batch axis); a
        regression guard for that trace axis choice.
        """
        A = jnp.stack(
            [jnp.array([[2.0, 1.0], [0.0, 3.0]]), jnp.array([[4.0, 0.0], [2.0, 5.0]])]
        )
        grad_custom = jax.grad(lambda mat: qm_det(mat).sum())(A)
        grad_ref = jax.grad(lambda mat: jnp.linalg.det(mat).sum())(A)
        assert jnp.allclose(grad_custom, grad_ref)

    def test_det_jit_vmap(self):
        """jit(vmap(det)) works correctly."""
        A = jnp.stack(
            [jnp.diag(jnp.array([2.0, 3.0])), jnp.diag(jnp.array([4.0, 5.0]))]
        )
        results = jax.jit(jax.vmap(qm_det))(A)
        expected = jnp.array([6.0, 20.0])
        assert jnp.allclose(results, expected)

    def test_det_batched_shape(self):
        """det_p on a (*batch, n, n) array returns shape (*batch,)."""
        A = jnp.ones((3, 4, 2, 2))
        # det of 2×2 ones matrix = 1*1 - 1*1 = 0
        result = qm_det(A)
        assert result.shape == (3, 4)


class TestDetQuantityMatrix:
    """Tests for det_p Quax dispatch on QuantityMatrix."""

    def test_returns_abstract_quantity(self):
        """Det of a 2×2 QuantityMatrix returns an AbstractQuantity."""
        A = QMat(jnp.array([[2, 0], [0, 3]]), unit=((_m, _m), (_m, _m)))
        result = quax.quaxify(qm_det)(A)
        assert isinstance(result, u.AbstractQuantity)

    def test_value_2x2_diagonal(self):
        """Numeric value equals jnp.linalg.det of the value array."""
        A = QMat(jnp.array([[2, 0], [0, 3]]), unit=((_m, _m), (_m, _m)))
        result = quax.quaxify(qm_det)(A)
        assert jnp.allclose(result.value, 6)

    def test_unit_product_of_diagonal(self):
        """Unit is the product of the main-diagonal units: m·m = m²."""
        A = QMat(jnp.array([[2, 0], [0, 3]]), unit=((_m, _m), (_m, _m)))
        result = quax.quaxify(qm_det)(A)
        assert result.unit == u.unit("m2")

    def test_unit_heterogeneous_diagonal(self):
        """Unit is u00 * u11 for a 2×2 matrix with mixed diagonal units."""
        A = QMat(jnp.eye(2), unit=((_m, _s), (_m, _s)))
        result = quax.quaxify(qm_det)(A)
        assert result.unit == _m * _s

    def test_unit_3x3_uniform(self):
        """Det of 3×3 identity with uniform unit m gives unit m³."""
        A = QMat(jnp.eye(3), unit=((_m, _m, _m),) * 3)
        result = quax.quaxify(qm_det)(A)
        assert jnp.allclose(result.value, 1)
        assert result.unit == u.unit("m3")

    def test_jit_QuantityMatrix(self):
        """Det of QuantityMatrix works under jax.jit."""
        A = QMat(jnp.array([[2.0, 0.0], [0.0, 3.0]]), unit=((_m, _m), (_m, _m)))
        result = jax.jit(quax.quaxify(qm_det))(A)
        assert jnp.allclose(result.value, 6)
        assert result.unit == u.unit("m2")

    def test_grad_QuantityMatrix(self):
        """Grad flows through the QuantityMatrix det dispatch, not just plain arrays."""

        def loss(mat_val):
            A = QMat(mat_val, unit=((_m, _m), (_m, _m)))
            return quax.quaxify(qm_det)(A).value

        base = jnp.array([[2.0, 0.0], [0.0, 3.0]])
        # ∂det/∂A = det(A)·A⁻ᵀ = diag(3, 2) for diag(2, 3).
        assert jnp.allclose(jax.grad(loss)(base), jnp.array([[3.0, 0.0], [0.0, 2.0]]))

    def test_vmap_QuantityMatrix(self):
        """Vmap maps det over a batch of QuantityMatrix values (unit preserved)."""
        vals = jnp.stack(
            [jnp.diag(jnp.array([2.0, 3.0])), jnp.diag(jnp.array([4.0, 5.0]))]
        )

        def one(mat_val):
            return quax.quaxify(qm_det)(QMat(mat_val, unit=((_m, _m), (_m, _m))))

        out = jax.vmap(one)(vals)
        assert jnp.allclose(out.value, jnp.array([6.0, 20.0]))
        assert out.unit == u.unit("m2")


# ---------------------------------------------------------------------------
# inv_p custom primitive + Quax dispatch for QuantityMatrix
# ---------------------------------------------------------------------------


class TestInvPrimitive:
    """Tests for the custom ``inv_p`` JAX primitive on plain arrays."""

    def test_inv_2x2_diagonal(self):
        """inv_p on a diagonal matrix returns the reciprocal diagonal."""
        A = jnp.array([[2.0, 0.0], [0.0, 4.0]])
        result = qm_inv(A)
        expected = jnp.linalg.inv(A)
        assert jnp.allclose(result, expected)

    def test_inv_3x3_identity(self):
        """inv(I) == I."""
        A = jnp.eye(3)
        assert jnp.allclose(qm_inv(A), A)

    def test_inv_matches_jnp_linalg_inv(self):
        """inv_p gives the same result as jnp.linalg.inv for a generic matrix."""
        A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        assert jnp.allclose(qm_inv(A), jnp.linalg.inv(A))

    def test_inv_jit(self):
        """inv_p works under jax.jit."""
        A = jnp.array([[2.0, 0.0], [0.0, 4.0]])
        result = jax.jit(qm_inv)(A)
        assert jnp.allclose(result, jnp.linalg.inv(A))

    def test_inv_jit_integer_dtype(self):
        """Integer input promotes to inexact under jit (abstract eval must match)."""
        A = jnp.array([[2, 0], [0, 4]], dtype=jnp.int32)
        result = jax.jit(qm_inv)(A)  # would fail lowering if abstract dtype disagreed
        assert jnp.issubdtype(result.dtype, jnp.inexact)
        assert jnp.allclose(result, jnp.array([[0.5, 0.0], [0.0, 0.25]]))

    def test_inv_jvp(self):
        """Forward-mode derivative of A^{-1}: d(A^{-1})(dA) = -A^{-1} dA A^{-1}."""
        A = jnp.array([[2.0, 0.0], [0.0, 4.0]])
        dA = jnp.ones((2, 2))
        primal, tangent = jax.jvp(qm_inv, (A,), (dA,))
        assert jnp.allclose(primal, jnp.linalg.inv(A))
        # expected tangent: -A^{-1} dA A^{-1}
        Ainv = jnp.linalg.inv(A)
        expected_tangent = -(Ainv @ dA @ Ainv)
        assert jnp.allclose(tangent, expected_tangent)

    def test_inv_grad(self):
        """Reverse-mode gradient via VJP derived from JVP."""
        A = jnp.array([[2.0, 0.0], [0.0, 4.0]])

        # Scalar output f(A) = sum(inv(A))
        def f(a):
            return jnp.sum(qm_inv(a))

        grad_A = jax.grad(f)(A)
        # Finite-difference check. Use a comparatively large step: at float32
        # precision a tiny eps (e.g. 1e-5) makes inv(Ap) - inv(Am) dominated by
        # rounding (catastrophic cancellation); eps=1e-2 keeps the difference
        # well above the float32 noise floor while truncation error stays small.
        eps = 1e-2
        fd = jnp.zeros_like(A)
        for i in range(2):
            for j in range(2):
                Ap = A.at[i, j].add(eps)
                Am = A.at[i, j].add(-eps)
                fd_val = (jnp.sum(jnp.linalg.inv(Ap)) - jnp.sum(jnp.linalg.inv(Am))) / (
                    2 * eps
                )
                fd = fd.at[i, j].set(fd_val)
        assert jnp.allclose(grad_A, fd, atol=1e-2)

    def test_inv_jit_grad(self):
        """jit(grad(sum(inv(A)))) works correctly."""
        A = jnp.array([[2.0, 0.0], [0.0, 4.0]])

        def f(a):
            return jnp.sum(qm_inv(a))

        grad_A = jax.jit(jax.grad(f))(A)

        def g(a):
            return jnp.sum(jnp.linalg.inv(a))

        expected = jax.grad(g)(A)
        assert jnp.allclose(grad_A, expected, atol=1e-6)

    def test_inv_vmap(self):
        """inv_p works under jax.vmap — maps over a batch of matrices."""
        A = jnp.stack(
            [jnp.diag(jnp.array([2.0, 4.0])), jnp.diag(jnp.array([1.0, 2.0]))]
        )
        results = jax.vmap(qm_inv)(A)
        expected = jax.vmap(jnp.linalg.inv)(A)
        assert jnp.allclose(results, expected)

    def test_inv_jit_vmap(self):
        """jit(vmap(inv)) works correctly."""
        A = jnp.stack(
            [jnp.diag(jnp.array([2.0, 4.0])), jnp.diag(jnp.array([1.0, 2.0]))]
        )
        results = jax.jit(jax.vmap(qm_inv))(A)
        expected = jax.vmap(jnp.linalg.inv)(A)
        assert jnp.allclose(results, expected)

    def test_inv_batched_shape(self):
        """inv_p on a (*batch, n, n) array returns shape (*batch, n, n)."""
        A = jnp.stack([jnp.eye(2)] * 6).reshape(3, 2, 2, 2)
        result = jax.vmap(jax.vmap(qm_inv))(A)
        assert result.shape == (3, 2, 2, 2)


class TestInvQuantityMatrix:
    """Tests for inv_p Quax dispatch on QuantityMatrix."""

    def test_heterogeneous_units_rejected(self):
        """Inv requires uniform units; a matrix inverse isn't elementwise."""
        A = QMat(jnp.array([[2.0, 0.0], [0.0, 3.0]]), unit=((_m, _s), (_kg, _m)))
        with pytest.raises(ValueError, match="uniform units"):
            quax.quaxify(qm_inv)(A)

    def test_non_2d_rejected(self):
        """Inv of a 1-D (vector) QuantityMatrix raises: a matrix inverse needs 2-D."""
        v = QMat(jnp.array([1.0, 2.0]), unit=(_m, _m))
        with pytest.raises(ValueError, match="2-D unit structure"):
            quax.quaxify(qm_inv)(v)

    def test_returns_QuantityMatrix(self):
        """Inv of a 2×2 QuantityMatrix returns a QuantityMatrix."""
        A = QMat(jnp.array([[4, 0], [0, 1]]), unit=((_m, _m), (_m, _m)))
        result = quax.quaxify(qm_inv)(A)
        assert isinstance(result, QMat)

    def test_value_2x2_diagonal(self):
        """Numeric value equals jnp.linalg.inv of the value array."""
        A = QMat(jnp.array([[4, 0], [0, 1]]), unit=((_m, _m), (_m, _m)))
        result = quax.quaxify(qm_inv)(A)
        expected_val = jnp.linalg.inv(jnp.array([[4.0, 0.0], [0.0, 1.0]]))
        assert jnp.allclose(result.value, expected_val)

    def test_unit_reciprocal(self):
        """Unit of the inverse is the reciprocal of the original unit: 1/m."""
        A = QMat(jnp.array([[4, 0], [0, 1]]), unit=((_m, _m), (_m, _m)))
        result = quax.quaxify(qm_inv)(A)
        expected_unit = u.unit("1 / m")
        assert result.unit[0, 0] == expected_unit
        assert result.unit[1, 1] == expected_unit

    def test_unit_m2_per_rad2(self):
        """Inv of a metric with m²/rad² entries carries rad²/m² units."""
        m2_r2 = u.unit("m2 / rad2")
        A = QMat(jnp.array([[4, 0], [0, 1]]), unit=((m2_r2, m2_r2), (m2_r2, m2_r2)))
        result = quax.quaxify(qm_inv)(A)
        assert result.unit[0, 0] == u.unit("rad2 / m2")

    def test_jit_QuantityMatrix(self):
        """Inv of QuantityMatrix works under jax.jit."""
        A = QMat(jnp.array([[4.0, 0.0], [0.0, 1.0]]), unit=((_m, _m), (_m, _m)))
        result = jax.jit(quax.quaxify(qm_inv))(A)
        assert jnp.allclose(result.value, jnp.array([[0.25, 0.0], [0.0, 1.0]]))
        assert result.unit[0, 0] == u.unit("1 / m")

    def test_grad_QuantityMatrix(self):
        """Grad flows through the QuantityMatrix inv dispatch (value path)."""

        def loss(mat_val):
            A = QMat(mat_val, unit=((_m, _m), (_m, _m)))
            return jnp.sum(quax.quaxify(qm_inv)(A).value)

        base = jnp.array([[2.0, 0.0], [0.0, 4.0]])
        # Compare against grad of the plain-array reference.
        expected = jax.grad(lambda a: jnp.sum(jnp.linalg.inv(a)))(base)
        assert jnp.allclose(jax.grad(loss)(base), expected, atol=1e-6)

    def test_vmap_QuantityMatrix(self):
        """Vmap maps inv over a batch of QuantityMatrix values (unit preserved)."""
        vals = jnp.stack(
            [jnp.diag(jnp.array([2.0, 4.0])), jnp.diag(jnp.array([1.0, 2.0]))]
        )

        def one(mat_val):
            return quax.quaxify(qm_inv)(QMat(mat_val, unit=((_m, _m), (_m, _m))))

        out = jax.vmap(one)(vals)
        assert jnp.allclose(out.value, jax.vmap(jnp.linalg.inv)(vals))
        assert out.unit[0, 0] == u.unit("1 / m")

    def test_roundtrip_identity(self):
        """A @ inv(A) ≈ I for a QuantityMatrix (value check)."""
        A = QMat(jnp.array([[2, 1], [1, 3]]), unit=((_m, _m), (_m, _m)))
        Ainv = quax.quaxify(qm_inv)(A)
        product = A.value @ Ainv.value
        assert jnp.allclose(product, jnp.eye(2), atol=1e-6)
