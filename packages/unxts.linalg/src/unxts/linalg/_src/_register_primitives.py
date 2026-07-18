"""Quax primitive registrations for QuantityMatrix arithmetic.

Registers handlers for the following JAX primitives:
- ``lax.add_p`` — element-wise addition
- ``lax.sub_p`` — element-wise subtraction
- ``lax.mul_p`` — element-wise multiplication
- ``lax.div_p`` — element-wise division
- ``lax.dot_general_p`` — dot product / matrix multiply
- ``lax.transpose_p`` — matrix transpose
- ``lax.gather_p`` — element-selection gather (e.g. jnp.diag)
- ``lax.reduce_sum_p`` — summation reduction
"""

from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import quax
from jax import lax

import unxt as u
from ._quantity_matrix import (
    QuantityMatrix,
    _convert_value,
    _convert_value_matrix,
)
from ._units_matrix import UnitsMatrix
from unxt.quantity import AllowValue

# Vectorised uconvert_value — used by dot-product helpers.
vec_uconvert_value = np.vectorize(u.uconvert_value)

_DMLS = u.unit("")


# ── add / sub ────────────────────────────────────────────────────────────


@quax.register(lax.add_p)
def add_qm_qm(x: QuantityMatrix, y: QuantityMatrix, /) -> QuantityMatrix:
    """Element-wise addition of two `QuantityMatrix` objects.

    The result adopts the units of *x*.  Each element is converted from
    ``y.unit`` → ``x.unit`` before the numeric add.

    Works for both 1D (vector) and 2D (matrix) cases.

    >>> import jax.numpy as jnp
    >>> import quaxed.numpy as qnp
    >>> import unxt as u
    >>> from unxts.linalg import QuantityMatrix

    2D case:

    >>> a = QuantityMatrix(jnp.ones((2, 2)), unit=(("m", "s"), ("kg", "rad")))
    >>> b = QuantityMatrix(jnp.ones((2, 2)), unit=(("km", "ms"), ("g", "deg")))

    >>> result = qnp.add(a, b)
    >>> result.unit.to_string()
    '((m, s), (kg, rad))'

    >>> result.value
    Array([[1001.       ,    1.001    ],
           [   1.001    ,    1.0174533]], dtype=float32)

    1D case:

    >>> a1d = QuantityMatrix(jnp.ones(3), unit=("m", "s", "kg"))
    >>> b1d = QuantityMatrix(jnp.ones(3), unit=("km", "ms", "g"))

    >>> result1d = qnp.add(a1d, b1d)
    >>> result1d.unit.to_string()
    '(m, s, kg)'

    >>> result1d.value
    Array([1001.   ,    1.001,    1.001], dtype=float32)

    """
    y_converted = _convert_value(y.value, y.unit, x.unit)
    return QuantityMatrix(value=lax.add(x.value, y_converted), unit=x.unit)


@quax.register(lax.sub_p)
def sub_qm_qm(x: QuantityMatrix, y: QuantityMatrix, /) -> QuantityMatrix:
    """Element-wise subtraction of two `QuantityMatrix` objects.

    The result adopts the units of *x*.  Each element is converted from
    ``y.unit`` → ``x.unit`` before the numeric subtract.

    Works for both 1D (vector) and 2D (matrix) cases.

    >>> import jax.numpy as jnp
    >>> import quaxed.numpy as qnp
    >>> import unxt as u
    >>> from unxts.linalg import QuantityMatrix

    2D case:

    >>> a = QuantityMatrix(jnp.ones((2, 2)), unit=(("m", "s"), ("kg", "rad")))
    >>> b = QuantityMatrix(
    ...     value=jnp.ones((2, 2)),
    ...     unit=(("km", u.unit("ms")), (u.unit("g"), u.unit("deg"))),
    ... )

    >>> result = qnp.subtract(a, b)
    >>> result.unit.to_string()
    '((m, s), (kg, rad))'

    >>> result.value
    Array([[-9.990000e+02,  9.990000e-01],
           [ 9.990000e-01,  9.825467e-01]], dtype=float32)

    1D case:

    >>> a1d = QuantityMatrix(value=jnp.ones(3), unit=("m", "s", "kg"))
    >>> b1d = QuantityMatrix(value=jnp.ones(3), unit=("km", u.unit("ms"), u.unit("g")))

    >>> result1d = qnp.subtract(a1d, b1d)
    >>> result1d.unit.to_string()
    '(m, s, kg)'

    >>> result1d.value
    Array([-999.   ,    0.999,    0.999], dtype=float32)

    """
    y_converted = _convert_value(y.value, y.unit, x.unit)
    return QuantityMatrix(value=lax.sub(x.value, y_converted), unit=x.unit)


# ── mul / div (element-wise) ──────────────────────────────────────────────
#
# Element-wise (Hadamard) product / quotient. Unlike add/sub, the per-element
# units *compose* multiplicatively (no unit conversion), so `m * s -> m s`.
# QuantityMatrix subclasses AbstractQuantity, whose generic `mul_p` rule would
# do `x.unit * y.unit` and build the *left operand's* type — producing a plain
# Quantity wrapping a UnitsMatrix (a malformed object) when a Quantity is on the
# left. These handlers keep the result a QuantityMatrix for every operand order.


@quax.register(lax.mul_p)
def mul_qm_qm(x: QuantityMatrix, y: QuantityMatrix, /, **kw: Any) -> QuantityMatrix:
    """Element-wise product of two `QuantityMatrix` objects (units multiply)."""
    return QuantityMatrix(x.value * y.value, unit=x.unit * y.unit)


@quax.register(lax.mul_p)
def mul_qm_qty(
    x: QuantityMatrix, y: u.AbstractQuantity, /, **kw: Any
) -> QuantityMatrix:
    """`QuantityMatrix` x uniform-unit Quantity: scale values, multiply units."""
    y_unit = u.unit_of(y)
    y_val = u.ustrip(AllowValue, y_unit, y)
    return QuantityMatrix(x.value * y_val, unit=x.unit * y_unit)


@quax.register(lax.mul_p)
def mul_qty_qm(
    x: u.AbstractQuantity, y: QuantityMatrix, /, **kw: Any
) -> QuantityMatrix:
    """Quantity x `QuantityMatrix` — multiplication commutes."""
    return mul_qm_qty(y, x)


@quax.register(lax.mul_p)
def mul_qm_arr(x: QuantityMatrix, y: jax.Array, /, **kw: Any) -> QuantityMatrix:
    """`QuantityMatrix` x dimensionless array: scale values, units unchanged."""
    return QuantityMatrix(x.value * y, unit=x.unit)


@quax.register(lax.mul_p)
def mul_arr_qm(x: jax.Array, y: QuantityMatrix, /, **kw: Any) -> QuantityMatrix:
    """Dimensionless array x `QuantityMatrix`."""
    return QuantityMatrix(x * y.value, unit=y.unit)


@quax.register(lax.div_p)
def div_qm_qm(x: QuantityMatrix, y: QuantityMatrix, /, **kw: Any) -> QuantityMatrix:
    """Element-wise quotient of two `QuantityMatrix` objects (units divide)."""
    return QuantityMatrix(x.value / y.value, unit=x.unit / y.unit)


@quax.register(lax.div_p)
def div_qm_qty(
    x: QuantityMatrix, y: u.AbstractQuantity, /, **kw: Any
) -> QuantityMatrix:
    """`QuantityMatrix` / uniform-unit Quantity."""
    y_unit = u.unit_of(y)
    y_val = u.ustrip(AllowValue, y_unit, y)
    return QuantityMatrix(x.value / y_val, unit=x.unit / y_unit)


@quax.register(lax.div_p)
def div_qty_qm(
    x: u.AbstractQuantity, y: QuantityMatrix, /, **kw: Any
) -> QuantityMatrix:
    """Uniform-unit Quantity / `QuantityMatrix` (not commutative)."""
    x_unit = u.unit_of(x)
    x_val = u.ustrip(AllowValue, x_unit, x)
    return QuantityMatrix(x_val / y.value, unit=x_unit / y.unit)


@quax.register(lax.div_p)
def div_qm_arr(x: QuantityMatrix, y: jax.Array, /, **kw: Any) -> QuantityMatrix:
    """`QuantityMatrix` / dimensionless array: scale values, units unchanged."""
    return QuantityMatrix(x.value / y, unit=x.unit)


@quax.register(lax.div_p)
def div_arr_qm(x: jax.Array, y: QuantityMatrix, /, **kw: Any) -> QuantityMatrix:
    """Dimensionless array / `QuantityMatrix` (units invert)."""
    return QuantityMatrix(x / y.value, unit=_DMLS / y.unit)


# ── dot_general helpers ───────────────────────────────────────────────────


def _check_contract(lhs_dim: int, rhs_dim: int, /) -> None:
    """Validate that the contraction dimensions match.

    An explicit check (not ``assert``, which ``python -O`` strips) so a shape
    mismatch fails deterministically with a clear message.
    """
    if lhs_dim != rhs_dim:
        msg = (
            f"QuantityMatrix dot_general contraction mismatch: {lhs_dim} != {rhs_dim}."
        )
        raise ValueError(msg)


def _dot_general_1d_1d(
    lhs: QuantityMatrix,
    rhs: QuantityMatrix,
    /,
    *,
    dimension_numbers: lax.DotDimensionNumbers,
    precision: Any = None,
    preferred_element_type: Any = None,
    **kw: Any,
) -> u.Q:
    """Vector dot product: (N,) @ (N,) → scalar.

    Result = Σ_i  lhs[i] * rhs[i]

    All terms must be unit-compatible. We convert to the unit of the first term.
    """
    n = lhs.shape[-1]
    _check_contract(n, rhs.shape[-1])

    # Reference unit: lhs.unit[0] * rhs.unit[0]
    ref_unit = lhs.unit[0] * rhs.unit[0]

    # Compute scale factors. ``uconvert_value`` returns Python floats, so a bare
    # ``jnp.array`` is float64 and would silently upcast a float32 contraction
    # under jax_enable_x64=True. Cast to ``result_type(values, 1.0)``: the weak
    # float keeps the scale *at least* floating (so integer operands still get a
    # correct fractional conversion) without widening float32 → float64.
    scales = jnp.array(
        [u.uconvert_value(ref_unit, lhs.unit[i] * rhs.unit[i], 1.0) for i in range(n)],
        dtype=jnp.result_type(lhs.value, rhs.value, 1.0),
    )

    # Compute dot product with rescaling
    result_value = jnp.sum(scales * lhs.value * rhs.value, axis=-1)

    return u.Q(result_value, ref_unit)


def _dot_general_2d_1d(
    lhs: QuantityMatrix,
    rhs: QuantityMatrix,
    /,
    *,
    dimension_numbers: lax.DotDimensionNumbers,
    precision: Any = None,
    preferred_element_type: Any = None,
    **kw: Any,
) -> QuantityMatrix:
    """Matrix-vector multiply: (N, K) @ (K,) → (N,).

    For ``w = A @ v`` where ``A`` is ``(N, K)`` and ``v`` is ``(K,)``:

    ``w[i] = Σ_j  A[i, j] * v[j]``

    Each product ``A[i,j] * v[j]`` has unit ``A.unit[i][j] * v.unit[j]``.  All
    ``K`` terms in the sum for output row ``i`` must be unit-compatible.  We
    convert every term to the unit of the *first* term (``j = 0``) for each
    output row ``i``: ``ref[i] = A.unit[i][0] * v.unit[0]``.

    >>> import jax.numpy as jnp
    >>> import quaxed.numpy as qnp
    >>> import unxt as u
    >>> from unxts.linalg import QuantityMatrix

    Identity matrix times a vector:

    >>> A = QuantityMatrix(jnp.eye(3), unit=(("", "", ""), ("", "", ""), ("", "", "")))
    >>> v = QuantityMatrix(jnp.array([1.0, 2.0, 3.0]), unit=("m", "m", "m"))
    >>> w = qnp.matmul(A, v)
    >>> w.value
    Array([1., 2., 3.], dtype=float32)

    Mixed units on contraction axis (km column converted to m):

    >>> A2 = QuantityMatrix(
    ...     jnp.array([[1.0, 2.0], [3.0, 4.0]]), unit=(("m", "km"), ("m", "km"))
    ... )
    >>> v2 = QuantityMatrix(jnp.array([1.0, 1.0]), unit=("s", "s"))
    >>> w2 = qnp.matmul(A2, v2)
    >>> w2.value
    Array([2001., 4003.], dtype=float32)
    >>> w2.unit.to_string()
    '(m s, m s)'

    """
    _check_contract(lhs.shape[-1], rhs.shape[-1])

    # 1) Output units: ref[i] = lhs.unit[i][0] * rhs.unit[0]
    out_unit = UnitsMatrix(np.multiply(lhs.unit._units[:, 0], rhs.unit._units[0]))

    # 2) Precompute scale factors: scale[i, j] converts
    #    lhs.unit[i][j]*rhs.unit[j] → ref[i]
    scale_2d = jnp.asarray(
        vec_uconvert_value(
            out_unit._units[:, None],  # (N, 1) — broadcast over K
            np.multiply(lhs.unit._units, rhs.unit._units[None, :]),  # (N, K)
            1.0,
        ),
        dtype=jnp.result_type(lhs.value, rhs.value, 1.0),
    )

    # 3) Vectorised contraction:
    #    w[..., i] = Σ_j  scale[i, j] * A[..., i, j] * v[..., j]
    accum = jnp.einsum("ij,...ij,...j->...i", scale_2d, lhs.value, rhs.value)

    return QuantityMatrix(value=accum, unit=out_unit)


def _dot_general_1d_2d(
    lhs: QuantityMatrix,
    rhs: QuantityMatrix,
    /,
    *,
    dimension_numbers: lax.DotDimensionNumbers,
    precision: Any = None,
    preferred_element_type: Any = None,
    **kw: Any,
) -> QuantityMatrix:
    """Vector-matrix multiply: (K,) @ (K, M) → (M,).

    For ``w = v @ A`` where ``v`` is ``(K,)`` and ``A`` is ``(K, M)``:

    ``w[k] = Σ_j  v[j] * A[j, k]``

    Each product ``v[j] * A[j,k]`` has unit ``v.unit[j] * A.unit[j][k]``.  All
    ``K`` terms in the sum for output column ``k`` must be unit-compatible.  We
    convert every term to the unit of the *first* term (``j = 0``) for each
    output column ``k``: ``ref[k] = v.unit[0] * A.unit[0][k]``.  This is the
    transpose of the matrix-vector case (:func:`_dot_general_2d_1d`).

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import unxts.linalg as ul

    Row vector times a matrix:

    >>> v = ul.QuantityMatrix(jnp.array([1.0, 1.0]), unit=("s", "s"))
    >>> A = ul.QuantityMatrix(
    ...     jnp.array([[1.0, 2.0], [3.0, 4.0]]), unit=(("m", "km"), ("m", "km"))
    ... )
    >>> w = ul.vecmat(v, A)
    >>> w.value
    Array([4., 6.], dtype=float32)
    >>> w.unit.to_string()
    '(m s, km s)'

    """
    _check_contract(lhs.shape[-1], rhs.shape[-2])

    # 1) Output units: ref[k] = lhs.unit[0] * rhs.unit[0][k]
    out_unit = UnitsMatrix(np.multiply(lhs.unit._units[0], rhs.unit._units[0, :]))

    # 2) Precompute scale factors: scale[j, k] converts
    #    lhs.unit[j]*rhs.unit[j][k] → ref[k]
    scale_2d = jnp.asarray(
        vec_uconvert_value(
            out_unit._units[None, :],  # (1, M) — broadcast over K
            np.multiply(lhs.unit._units[:, None], rhs.unit._units),  # (K, M)
            1.0,
        ),
        dtype=jnp.result_type(lhs.value, rhs.value, 1.0),
    )

    # 3) Vectorised contraction:
    #    w[..., k] = Σ_j  scale[j, k] * v[..., j] * A[..., j, k]
    accum = jnp.einsum("jk,...j,...jk->...k", scale_2d, lhs.value, rhs.value)

    return QuantityMatrix(value=accum, unit=out_unit)


def _dot_general_2d_2d(
    lhs: QuantityMatrix,
    rhs: QuantityMatrix,
    /,
    *,
    dimension_numbers: lax.DotDimensionNumbers,
    precision: Any = None,
    preferred_element_type: Any = None,
    **kw: Any,
) -> QuantityMatrix:
    """Matrix multiply: (N, K) @ (K, M) → (N, M).

    For ``C = A @ B`` where ``A`` is ``(N, K)`` and ``B`` is ``(K, M)``:

    ``C[i, k] = Σ_j  A[i, j] * B[j, k]``

    Each product ``A[i,j] * B[j,k]`` has unit ``A.unit[i][j] * B.unit[j][k]``.
    All ``K`` terms in the sum **must** be unit-compatible.  We convert every
    term to the unit of the *first* term (``j = 0``) using `u.uconvert_value`,
    then sum with a plain matmul.

    The strategy:
    1. Pick a reference unit for each ``(i, k)`` output element:
       ``ref[i][k] = A.unit[i][0] * B.unit[0][k]``.
    2. For each contraction index ``j``, compute per-element conversion
       factors from ``A.unit[i][j] * B.unit[j][k]`` to ``ref[i][k]``.
       Because the products are *multiplicative* compositions, the
       conversion from ``u_A * u_B`` to ``ref`` is multiplicative even
       when the individual units are affine — the product of two
       absolute quantities is always absolute.
       So we can safely compute a scale factor:
       ``scale[i][j][k] = uconvert_value(ref[i][k], A.unit[i][j] * B.unit[j][k], 1.0)``
    3. Build the rescaled sum as:
       ``C_val[i, k] = Σ_j  scale[i][j][k] * A_val[i, j] * B_val[j, k]``
       Done via ``C_val = (A_val * S_ij) @ B_val`` per output column, or
       equivalently with a loop + accumulate.
    """
    # Check contraction axis
    _check_contract(lhs.shape[-1], rhs.shape[-2])

    # 1) Compute output units: ref[i][k] = lhs.unit[i][0] * rhs.unit[0][k]
    out_unit = np.multiply(lhs.unit._units[:, 0:1], rhs.unit._units[0:1, :])

    # 2) Precompute all scale factors as a (N, K, M) constant array.
    #    scale[i, j, k] converts lhs.unit[i][j]*rhs.unit[j][k] → out_unit[i][k].
    #
    #    CORRECTNESS NOTE — why a multiplicative scale factor is exact:
    #    Affine units (°C, °F) are the only units where a bare
    #    multiplicative scale would be wrong (they have an additive
    #    offset).  But astropy rejects product conversions involving
    #    affine units — e.g. ``(deg_C * s).to(deg_F * s)`` raises
    #    ``UnitConversionError``.  Every product unit that astropy
    #    *does* accept (including logarithmic units like dex, mag) is
    #    a plain ``CompositeUnit`` whose conversion is purely
    #    multiplicative.  So ``uconvert_value(to, from, 1.0)`` yields
    #    an exact scale factor for all valid product units.
    #
    #    The tests in ``TestAffineProductUnitsRejected`` assert that
    #    astropy keeps rejecting affine product conversions.  If that
    #    ever changes, those tests will fail, alerting us that this
    #    assumption needs revisiting.
    scale_3d = jnp.asarray(
        vec_uconvert_value(
            out_unit[:, None, :],  # (N, 1, M)
            np.multiply(lhs.unit._units[:, :, None], rhs.unit._units[None, :, :]),
            1.0,  # ꜛ (N, K, M)
        ),
        dtype=jnp.result_type(lhs.value, rhs.value, 1.0),
    )

    # 3) Vectorised contraction — no Python loop, no accumulator.
    #    C[..., i, k] = Σ_j  scale[i, j, k] * A[..., i, j] * B[..., j, k]
    accum = jnp.sum(  # (N, K, M) * (..., N, K, 1) * (..., 1, K, M)
        scale_3d * lhs.value[..., :, :, None] * rhs.value[..., None, :, :], axis=-2
    )

    return QuantityMatrix(value=accum, unit=out_unit)


# ── dot_general dispatch ──────────────────────────────────────────────────


@quax.register(lax.dot_general_p)
def dot_general_qm_qm(
    lhs: QuantityMatrix,
    rhs: QuantityMatrix,
    /,
    *,
    dimension_numbers: lax.DotDimensionNumbers,
    precision: Any = None,
    preferred_element_type: Any = None,
    **kw: Any,
) -> QuantityMatrix | u.Q:
    """Dot product / matrix multiply two `QuantityMatrix` objects.

    Delegates to specialized implementations based on the (logical)
    dimensionality of each operand:
    - 1D @ 1D → scalar (vector dot product)
    - 2D @ 1D → 1D (matrix-vector product)
    - 1D @ 2D → 1D (vector-matrix product)
    - 2D @ 2D → 2D (matrix-matrix multiply)

    Leading batch axes on the value arrays are broadcast over. The batch-aware
    entry points are the wrappers in `unxts.linalg` (``matmul``/``matvec``/
    ``vecmat``/``vecdot``); ``matmul`` alone cannot express batched
    matrix-vector products (see `unxts.linalg._src._products`).

    For the standard matmul contraction: contracting_dims = ((-1,), (-2,)).
    Leading *batch* axes on the value arrays are supported: ``jnp.matmul`` /
    ``jnp.dot`` emit them as leading batch dimensions, which the
    ``_dot_general_*`` helpers broadcast over via ``...``.

    >>> import jax.numpy as jnp
    >>> import quaxed.numpy as qnp
    >>> import unxt as u
    >>> from unxts.linalg import QuantityMatrix

    1D @ 1D (dot product):

    >>> v1 = QuantityMatrix(jnp.array([1.0, 2.0]), unit=("m", "km"))
    >>> v2 = QuantityMatrix(jnp.array([3.0, 4.0]), unit=("s", "s"))
    >>> result = qnp.dot(v1, v2)
    >>> result.value
    Array(8003., dtype=float32)
    >>> result.unit
    Unit("m s")

    2D @ 2D (matrix multiply):

    >>> a = QuantityMatrix(
    ...     jnp.array([[1.0, 2.0], [3.0, 4.0]]), unit=(("m", "km"), ("m", "km"))
    ... )
    >>> b = QuantityMatrix(
    ...     jnp.array([[1.0, 0.0], [0.0, 1.0]]), unit=(("s", "s"), ("s", "s"))
    ... )

    >>> c = qnp.matmul(a, b)
    >>> c.unit.to_string()
    '((m s, m s), (m s, m s))'

    >>> c.value
    Array([[1.e+00, 2.e+03],
           [3.e+00, 4.e+03]], dtype=float32)

    """
    # The `_dot_general_*` helpers ignore `dimension_numbers` and assume the
    # standard (optionally batched) matmul/matvec/vecmat/vecdot contraction:
    # lhs contracts its last axis; rhs its last axis for a vector or its
    # second-last for a matrix; batch dims are the *leading* axes on both
    # operands. Anything else (e.g. a general einsum/tensordot) would get
    # incorrect unit propagation, so reject it explicitly. This is a user-facing
    # precondition, so it raises rather than `assert` (which `-O` strips).
    (lhs_c, rhs_c), (lhs_b, rhs_b) = dimension_numbers
    rhs_contract_expected = rhs.value.ndim - (1 if rhs.unit.ndim == 1 else 2)
    is_standard = (
        len(lhs_c) == 1
        and len(rhs_c) == 1
        and lhs_c[0] == lhs.value.ndim - 1
        and rhs_c[0] == rhs_contract_expected
        and lhs_b == tuple(range(len(lhs_b)))
        and rhs_b == tuple(range(len(rhs_b)))
        and len(lhs_b) == len(rhs_b)
    )
    if not is_standard:
        msg = (
            "QuantityMatrix supports only the standard (optionally batched) "
            "matmul/matvec/vecmat/vecdot contraction — lhs contracts its last "
            "axis, rhs its last (vector) or second-last (matrix) axis, with "
            "batch dims leading. Got dimension_numbers="
            f"{dimension_numbers!r}."
        )
        raise NotImplementedError(msg)

    # Delegate based on dimensionality
    if lhs.ndim == 1 and rhs.ndim == 1:
        return _dot_general_1d_1d(
            lhs,
            rhs,
            dimension_numbers=dimension_numbers,
            precision=precision,
            preferred_element_type=preferred_element_type,
            **kw,
        )
    if lhs.ndim == 2 and rhs.ndim == 2:
        return _dot_general_2d_2d(
            lhs,
            rhs,
            dimension_numbers=dimension_numbers,
            precision=precision,
            preferred_element_type=preferred_element_type,
            **kw,
        )
    if lhs.ndim == 2 and rhs.ndim == 1:
        return _dot_general_2d_1d(
            lhs,
            rhs,
            dimension_numbers=dimension_numbers,
            precision=precision,
            preferred_element_type=preferred_element_type,
            **kw,
        )
    if lhs.ndim == 1 and rhs.ndim == 2:
        return _dot_general_1d_2d(
            lhs,
            rhs,
            dimension_numbers=dimension_numbers,
            precision=precision,
            preferred_element_type=preferred_element_type,
            **kw,
        )
    msg = f"Unsupported dimensionality: lhs.ndim={lhs.ndim}, rhs.ndim={rhs.ndim}"
    raise NotImplementedError(msg)


def _wrap_operand(
    value: jax.Array, element_unit: Any, batch_axes: tuple[int, ...], /
) -> QuantityMatrix:
    """Wrap a plain value as a `QuantityMatrix` with a uniform per-element unit.

    The logical (vector vs matrix) rank is inferred from the number of *batch*
    axes in ``dimension_numbers`` — ``value.ndim - len(batch_axes)`` — so a
    batched vector ``(*batch, K)`` is correctly wrapped with a 1-D unit
    structure rather than being mistaken for a matrix by its raw ``ndim``.
    """
    logical_ndim = value.ndim - len(batch_axes)
    if logical_ndim == 1:
        n = value.shape[-1]
        unit = UnitsMatrix(tuple(element_unit for _ in range(n)))
    elif logical_ndim == 2:
        nr, nc = value.shape[-2], value.shape[-1]
        unit = UnitsMatrix(
            tuple(tuple(element_unit for _ in range(nc)) for _ in range(nr))
        )
    else:
        # Only vector/matrix operands are supported; fail here with a clear
        # message rather than an IndexError on value.shape[-2] below.
        msg = (
            "QuantityMatrix dot_general only supports vector (1-D) or matrix "
            f"(2-D) operands; got logical ndim {logical_ndim} for a value of "
            f"shape {value.shape} with {len(batch_axes)} batch axes."
        )
        raise NotImplementedError(msg)
    return QuantityMatrix(value, unit=unit)


@quax.register(lax.dot_general_p)
def dot_general_qm_arr(
    lhs: QuantityMatrix,
    rhs: jax.Array,
    /,
    *,
    dimension_numbers: lax.DotDimensionNumbers,
    precision: Any = None,
    preferred_element_type: Any = None,
    **kw: Any,
) -> "QuantityMatrix | u.Q":
    """Dot product of a :class:`QuantityMatrix` with a plain JAX array.

    The plain array is treated as dimensionless.  Delegates to
    :func:`dot_general_qm_qm` after wrapping ``rhs`` in a dimensionless
    :class:`QuantityMatrix`.

    >>> import jax.numpy as jnp
    >>> import quaxed.numpy as qnp
    >>> from unxts.linalg import QuantityMatrix, UnitsMatrix

    2D metric x 1D plain vector:

    >>> g = QuantityMatrix(
    ...     jnp.array([[2.0, 0.0], [0.0, 3.0]]),
    ...     unit=UnitsMatrix((("m2", "m2"), ("m2", "m2"))),
    ... )
    >>> v = jnp.array([1.0, 1.0])
    >>> w = qnp.matmul(g, v)
    >>> w.unit.to_string()
    '(m2, m2)'
    >>> w.value
    Array([2., 3.], dtype=float32)

    """
    rhs_qm = _wrap_operand(rhs, _DMLS, dimension_numbers[1][1])
    return dot_general_qm_qm(
        lhs,
        rhs_qm,
        dimension_numbers=dimension_numbers,
        precision=precision,
        preferred_element_type=preferred_element_type,
        **kw,
    )


@quax.register(lax.dot_general_p)
def dot_general_qm_qty(
    lhs: QuantityMatrix,
    rhs: u.AbstractQuantity,
    /,
    *,
    dimension_numbers: lax.DotDimensionNumbers,
    precision: Any = None,
    preferred_element_type: Any = None,
    **kw: Any,
) -> "QuantityMatrix | u.Q":
    """Dot product of a :class:`QuantityMatrix` with a :class:`~unxt.AbstractQuantity`.

    The Quantity carries a single scalar unit that applies uniformly to all
    elements.  The ``rhs`` is wrapped as a uniform-unit
    :class:`QuantityMatrix` and delegated to :func:`dot_general_qm_qm`.

    Note that :class:`QuantityMatrix` is itself a subtype of
    :class:`~unxt.AbstractQuantity`, so :func:`dot_general_qm_qm` takes
    precedence when both sides are :class:`QuantityMatrix`.

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import quaxed.numpy as qnp
    >>> from unxts.linalg import QuantityMatrix, UnitsMatrix

    2D metric with units @ uniform-unit Quantity vector:

    >>> g = QuantityMatrix(
    ...     jnp.array([[2.0, 0.0], [0.0, 3.0]]),
    ...     unit=UnitsMatrix((("m2 / rad2", "m2 / rad2"), ("m2 / rad2", "m2 / rad2"))),
    ... )
    >>> v = u.Q(jnp.array([1.0, 1.0]), "rad")
    >>> w = qnp.matmul(g, v)
    >>> w.unit.to_string()
    '(m2 / rad, m2 / rad)'
    >>> w.value
    Array([2., 3.], dtype=float32)

    """
    rhs_unit = u.unit_of(rhs)
    rhs_val = cast("jax.Array", u.ustrip(AllowValue, rhs_unit, rhs))
    rhs_qm = _wrap_operand(rhs_val, rhs_unit, dimension_numbers[1][1])
    return dot_general_qm_qm(
        lhs,
        rhs_qm,
        dimension_numbers=dimension_numbers,
        precision=precision,
        preferred_element_type=preferred_element_type,
        **kw,
    )


@quax.register(lax.dot_general_p)
def dot_general_qty_qm(
    lhs: u.AbstractQuantity,
    rhs: QuantityMatrix,
    /,
    *,
    dimension_numbers: lax.DotDimensionNumbers,
    precision: Any = None,
    preferred_element_type: Any = None,
    **kw: Any,
) -> "QuantityMatrix | u.Q":
    """Dot product of a :class:`~unxt.AbstractQuantity` with a :class:`QuantityMatrix`.

    The mirror of :func:`dot_general_qm_qty`. The ``lhs`` Quantity carries a
    single scalar unit that applies uniformly to all elements; it is wrapped as
    a uniform-unit :class:`QuantityMatrix` and delegated to
    :func:`dot_general_qm_qm`. Without this rule a plain Quantity on the left
    fell through to unxt's generic ``AbstractQuantity`` dot_general, which built
    a `~unxt.Quantity` whose ``.unit`` was a `UnitsMatrix` -- a malformed object.

    (:class:`QuantityMatrix` is itself an :class:`~unxt.AbstractQuantity`
    subtype, so :func:`dot_general_qm_qm` still takes precedence when both sides
    are :class:`QuantityMatrix`.)

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import quaxed.numpy as qnp
    >>> from unxts.linalg import QuantityMatrix, UnitsMatrix

    Uniform-unit Quantity vector @ 2D metric with units:

    >>> v = u.Q(jnp.array([1.0, 1.0]), "rad")
    >>> g = QuantityMatrix(
    ...     jnp.array([[2.0, 0.0], [0.0, 3.0]]),
    ...     unit=UnitsMatrix((("m2 / rad2", "m2 / rad2"), ("m2 / rad2", "m2 / rad2"))),
    ... )
    >>> w = qnp.matmul(v, g)
    >>> isinstance(w, QuantityMatrix)
    True
    >>> w.value
    Array([2., 3.], dtype=float32)

    """
    lhs_unit = u.unit_of(lhs)
    lhs_val = cast("jax.Array", u.ustrip(AllowValue, lhs_unit, lhs))
    lhs_qm = _wrap_operand(lhs_val, lhs_unit, dimension_numbers[1][0])
    return dot_general_qm_qm(
        lhs_qm,
        rhs,
        dimension_numbers=dimension_numbers,
        precision=precision,
        preferred_element_type=preferred_element_type,
        **kw,
    )


@quax.register(lax.dot_general_p)
def dot_general_arr_qm(
    lhs: jax.Array,
    rhs: QuantityMatrix,
    /,
    *,
    dimension_numbers: lax.DotDimensionNumbers,
    precision: Any = None,
    preferred_element_type: Any = None,
    **kw: Any,
) -> "QuantityMatrix | u.Q":
    """Dot product of a plain JAX array with a :class:`QuantityMatrix`.

    The plain array is treated as dimensionless.  Delegates to
    :func:`dot_general_qm_qm` after wrapping ``lhs`` in a dimensionless
    :class:`QuantityMatrix`.

    >>> import jax.numpy as jnp
    >>> import quaxed.numpy as qnp
    >>> from unxts.linalg import QuantityMatrix

    Dimensionless identity @ QuantityMatrix vector:

    >>> A = jnp.eye(2)
    >>> v = QuantityMatrix(jnp.array([2.0, 3.0]), unit=("m / s", "m / s"))
    >>> w = qnp.matmul(A, v)
    >>> w.unit.to_string()
    '(m / s, m / s)'
    >>> w.value
    Array([2., 3.], dtype=float32)

    """
    lhs_qm = _wrap_operand(lhs, _DMLS, dimension_numbers[1][0])
    return dot_general_qm_qm(
        lhs_qm,
        rhs,
        dimension_numbers=dimension_numbers,
        precision=precision,
        preferred_element_type=preferred_element_type,
        **kw,
    )


# ── transpose ────────────────────────────────────────────────────────────


@quax.register(lax.transpose_p)
def transpose_qm(
    x: QuantityMatrix, /, *, permutation: tuple[int, ...]
) -> QuantityMatrix:
    """Transpose a ``QuantityMatrix``, swapping only the last two (matrix) axes.

    Leading batch dimensions must be preserved unchanged.  Only permutations
    that swap the last two axes while keeping all batch axes in place are
    supported, because the unit structure is purely 2-D and cannot represent
    arbitrary axis re-orderings.

    >>> import jax.numpy as jnp
    >>> import quaxed.numpy as qnp
    >>> from unxts.linalg import QuantityMatrix

    2-D (no batch):

    >>> a = QuantityMatrix(
    ...     jnp.array([[1.0, 2.0], [3.0, 4.0]]), unit=(("m", "s"), ("kg", "rad"))
    ... )
    >>> aT = qnp.matrix_transpose(a)
    >>> aT.value
    Array([[1., 3.],
           [2., 4.]], dtype=float32)
    >>> aT.unit.to_string()
    '((m, kg), (s, rad))'

    Batched ``(B, N, M)`` — batch axis is preserved:

    >>> import jax
    >>> b = QuantityMatrix(jnp.ones((3, 2, 2)), unit=(("m", "s"), ("kg", "rad")))
    >>> bT = qnp.matrix_transpose(b)
    >>> bT.shape
    (3, 2, 2)

    """
    ndim_val = len(permutation)  # full ndim of the value array (includes batch dims)
    # An identity permutation is a no-op that preserves the unit structure — most
    # relevant for a 1-D vector, whose only permutation is (0,). JAX normally
    # elides such transposes before they bind ``transpose_p``, but handle them
    # directly so correctness does not depend on that elision.
    if tuple(permutation) == tuple(range(ndim_val)):
        return x
    if ndim_val < 2:
        msg = f"transpose_qm requires ndim >= 2, got ndim={ndim_val}"
        raise NotImplementedError(msg)
    # Validate: batch axes must be unchanged, last two must be swapped.
    expected = (*range(ndim_val - 2), ndim_val - 1, ndim_val - 2)
    if tuple(permutation) != expected:
        msg = (
            f"transpose_qm only supports matrix transpose of the last two axes "
            f"(expected permutation {expected}), got {tuple(permutation)}"
        )
        raise NotImplementedError(msg)
    transposed_value = lax.transpose(x.value, permutation)
    return QuantityMatrix(value=transposed_value, unit=x.unit.T)


# ── gather ───────────────────────────────────────────────────────────────


def _jit_fallback_uniform_unit(
    units: UnitsMatrix, out_shape: tuple[int, ...]
) -> UnitsMatrix:
    """Return a ``UnitsMatrix`` of shape *out_shape* if all units are equal.

    Used as a JIT-mode fallback inside ``gather_qm`` when the concrete gather
    indices are not available.  Raises ``ValueError`` for heterogeneous inputs.
    """
    all_units = jtu.tree_leaves(units.to_tuple())
    first = all_units[0]
    if any(u_i != first for u_i in all_units[1:]):
        msg = (
            "QuantityMatrix gather (e.g. jnp.diag) under jit requires all units "
            "to be equal when indices cannot be concretized. "
            "Call eagerly (outside jit) for heterogeneous-unit QuantityMatrix."
        )
        raise ValueError(msg)
    return UnitsMatrix(np.full(out_shape, first, dtype=object))


@quax.register(lax.gather_p)
def gather_qm(
    x: QuantityMatrix,
    start_indices: jax.Array,
    /,
    *,
    dimension_numbers: lax.GatherDimensionNumbers,
    slice_sizes: tuple[int, ...],
    indices_are_sorted: bool = False,
    mode: Any = None,
    fill_value: Any = None,
    unique_indices: bool = False,
    **kwargs: Any,
) -> QuantityMatrix:
    """Handle element-selection gathers (e.g. ``jnp.diag``) for ``QuantityMatrix``.

    Supports only *element-selection* gathers where every input dimension is
    collapsed (``offset_dims == ()`` and all ``slice_sizes == 1``).  This
    covers ``jnp.diag``, ``jnp.diagonal``, and integer-array fancy indexing on
    ``QuantityMatrix`` objects.

    Unit extraction:

    ``QuantityMatrix.unit`` is declared ``static=True`` and is therefore always
    a concrete Python object, even inside ``jax.jit``.  The *indices*, however,
    are traced under JIT and cannot be read concretely.  Because JAX's
    ``jnp.diag`` uses ``platform_dependent`` internally, quax always traces
    both branches via ``make_jaxpr``, so the JIT fallback path is taken for
    unit resolution.  Consequently, all units in the input must be equal;
    heterogeneous-unit inputs raise ``ValueError``.

    >>> import jax.numpy as jnp
    >>> from unxts.linalg import QuantityMatrix

    Diagonal of a 3x3 dimensionless matrix:

    >>> A = QuantityMatrix(
    ...     jnp.diag(jnp.array([1.0, 4.0, 9.0])),
    ...     unit=(("", "", ""), ("", "", ""), ("", "", "")),
    ... )
    >>> d = A.diag()
    >>> d.unit.shape
    (3,)
    >>> d.unit.ndim
    1
    >>> d.value
    Array([1., 4., 9.], dtype=float32)

    ```{note}
    ``jnp.diag`` uses JAX's ``platform_dependent`` internally, which causes
    quax to trace both branches via ``make_jaxpr`` even in eager mode.  This
    means the JIT fallback path is always taken for the unit computation, so
    **heterogeneous-unit matrices are not supported** with ``qnp.diag``.
    All units in the input must be equal; otherwise a ``ValueError`` is raised.
    ```
    """
    result_value = lax.gather(
        x.value,
        start_indices,
        dimension_numbers,
        slice_sizes,
        indices_are_sorted=indices_are_sorted,
        mode=mode,
        fill_value=fill_value,
        unique_indices=unique_indices,
    )

    # Only element-selection gathers are supported: all input dimensions must
    # be collapsed and every slice_size must be 1.
    n_input_dims = x.value.ndim
    normalized_collapsed = {
        d % n_input_dims for d in dimension_numbers.collapsed_slice_dims
    }
    is_element_selection = (
        dimension_numbers.offset_dims == ()
        and normalized_collapsed == set(range(n_input_dims))
        and all(s == 1 for s in slice_sizes)
    )
    if not is_element_selection:
        msg = (
            "QuantityMatrix: only element-selection gathers (all input dims "
            "collapsed, all slice_sizes == 1) are supported. "
            f"Got offset_dims={dimension_numbers.offset_dims}, "
            f"collapsed_slice_dims={dimension_numbers.collapsed_slice_dims}, "
            f"slice_sizes={slice_sizes}."
        )
        raise NotImplementedError(msg)

    # Output index-batch shape. For an element-selection gather JAX packs the
    # indices as ``start_indices.shape == (*index_shape, index_vector_dim)``, so
    # the output (and its unit structure) has shape ``index_shape``. This holds
    # for 1-D index arrays (``index_shape`` 1-D, e.g. jnp.diag) *and* for
    # multi-dimensional advanced indexing (``index_shape`` 2-D+). The last axis
    # is the index vector, indexed via ``idx_np[..., k]``.
    out_shape = start_indices.shape[:-1]

    # A scalar-output gather (``index_shape == ()``) selects a single element,
    # which carries a single unit. ``UnitsMatrix`` only represents 1-D/2-D unit
    # structures, so return a plain scalar ``Quantity`` instead.
    if out_shape == ():
        if isinstance(start_indices, jax.core.Tracer):  # ty: ignore[possibly-missing-submodule]
            scalar_unit = _jit_fallback_uniform_unit(x.unit, (1,))._units[0]
        else:
            idx_np = np.asarray(start_indices)
            scalar_unit = (
                x.unit._units[idx_np[0]]
                if x.unit.ndim == 1
                else x.unit._units[idx_np[0], idx_np[1]]
            )
        return u.Q(result_value, scalar_unit)

    if isinstance(start_indices, jax.core.Tracer):  # ty: ignore[possibly-missing-submodule]
        # JIT path: indices are traced — fall back to uniform-unit check.
        out_unit = _jit_fallback_uniform_unit(x.unit, out_shape)
    else:
        # Eager path: indices are concrete — look up units directly.
        idx_np = np.asarray(start_indices)
        if x.unit.ndim == 1:
            out_unit = UnitsMatrix(x.unit._units[idx_np[..., 0]])
        else:  # x.unit.ndim == 2
            out_unit = UnitsMatrix(x.unit._units[idx_np[..., 0], idx_np[..., 1]])

    return QuantityMatrix(value=result_value, unit=out_unit)


# ── reduce_sum ───────────────────────────────────────────────────────────


@quax.register(lax.reduce_sum_p)
def reduce_sum_p_qm(
    operand: QuantityMatrix, /, *, axes: Any, **kwargs: Any
) -> QuantityMatrix:
    """Handle ``lax.reduce_sum`` for ``QuantityMatrix``.

    ``jnp.diag`` on a square 2-D matrix uses ``platform_dependent`` which traces
    *both* the default (gather-based) and Mosaic implementation.  The Mosaic
    path computes ``reduce(mul(eye, A), axis=0)`` — JAX's JIT optimises
    ``lax.reduce(x, 0, lax.add, (0,))`` to the simpler ``reduce_sum_p``
    primitive.  This handler ensures the output carries the correct 1-D unit
    structure so that both branches produce the *same* pytree — required by
    ``platform_dependent`` / ``lax.switch``.

    Unit reduction rule:

    When reducing a 2-D ``QuantityMatrix`` along ``axes=(0,)`` (rows): the
    output unit for column *j* is taken from ``operand.unit[0, j]`` (the first
    row).  The other elements in each column are first *converted* to that
    reference unit before summing, so a column of unit-compatible-but-different
    units (e.g. ``m`` and ``km``) sums correctly rather than being silently
    relabelled.  Summing along a column of *incompatible* units raises (via
    ``uconvert_value``), as it must.

    Analogously for ``axes=(1,)`` (column reduction), the output unit for row
    *i* is ``operand.unit[i, 0]`` and each row is converted to it before
    summing.

    >>> import jax.numpy as jnp
    >>> from unxts.linalg import QuantityMatrix

    ``QuantityMatrix.diag()`` on a 3x3 uniform-unit matrix:

    >>> A = QuantityMatrix(
    ...     jnp.diag(jnp.array([1.0, 4.0, 9.0])),
    ...     unit=(("m", "m", "m"), ("m", "m", "m"), ("m", "m", "m")),
    ... )
    >>> d = A.diag()
    >>> d.unit.shape
    (3,)
    >>> d.unit.ndim
    1

    """
    # `axes` index the *value* array, which carries `n_batch` leading batch
    # axes that the unit structure does not. Map them to the logical axes.
    n_batch = operand.value.ndim - operand.unit.ndim
    axset = {a % operand.value.ndim for a in axes}
    logical_axes = frozenset(a - n_batch for a in axset if a >= n_batch)

    # Reducing only batch axes leaves the per-element unit structure unchanged.
    if not logical_axes:
        result_value = lax.reduce_sum_p.bind(operand.value, axes=axes, **kwargs)
        return QuantityMatrix(value=result_value, unit=operand.unit)

    units = operand.unit._units
    if operand.unit.ndim == 2 and logical_axes == {0}:
        # Row reduction → unit = first row's units. Convert every row to that
        # reference row before summing so compatible-but-different units add up.
        out_unit = UnitsMatrix(units[0])
        target = tuple(tuple(units[0]) for _ in range(units.shape[0]))
        value = _convert_value_matrix(operand.value, units, target)
    elif operand.unit.ndim == 2 and logical_axes == {1}:
        # Column reduction → unit = first column's units; convert each column.
        out_unit = UnitsMatrix(units[:, 0])
        target = tuple(
            tuple(units[i, 0] for _ in range(units.shape[1]))
            for i in range(units.shape[0])
        )
        value = _convert_value_matrix(operand.value, units, target)
    else:
        msg = (
            f"reduce_sum_p_qm: unsupported reduction over logical axes "
            f"{sorted(logical_axes)} of a {operand.unit.ndim}-D QuantityMatrix "
            f"(axes={axes}, n_batch={n_batch})."
        )
        raise NotImplementedError(msg)

    result_value = lax.reduce_sum_p.bind(value, axes=axes, **kwargs)
    return QuantityMatrix(value=result_value, unit=out_unit)
