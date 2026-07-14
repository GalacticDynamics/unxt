"""Custom ``inv_p`` JAX primitive with full JAX transform support.

JAX has no standalone ``inv`` primitive.  ``jnp.linalg.inv`` decomposes into
``lu_p`` + ``triangular_solve_p``, and Quax has no dispatch for those on
``QuantityMatrix`` (which raises on materialise).  We define ``inv_p`` to give
Quax a single interception point with full unit tracking.

Supported transforms:
  - JIT via MLIR lowering that delegates to ``jnp.linalg.inv``
  - Forward-mode autodiff (JVP): d(A^{-1}) = -A^{-1} dA A^{-1}
  - Reverse-mode autodiff (VJP): derived automatically from JVP
  - Batching (vmap): move batch axis to front, call inv_p
"""

import jax
import jax.numpy as jnp
import quax
from jax.extend import core as jexc
from jax.interpreters import ad as jax_ad, batching as jax_batching, mlir as jax_mlir
from jaxtyping import Array

from ._quantity_matrix import QuantityMatrix

inv_p = jexc.Primitive("inv")
inv_p.multiple_results = False


def inv(x: Array, /) -> Array:
    """Compute the matrix inverse of a square matrix via the ``inv_p`` primitive.

    Delegates to ``inv_p``, a custom JAX primitive that supports JIT,
    forward and reverse differentiation, and batching (vmap).

    For plain arrays the result is a bare :class:`~jaxtyping.Array`.
    For :class:`~unxts.linalg.QuantityMatrix` inputs the Quax
    dispatch intercepts the call (see ``_inv_p_QuantityMatrix``) and
    returns a :class:`~unxts.linalg.QuantityMatrix` with
    reciprocal units.

    Parameters
    ----------
    x : Array, shape ``(*batch, n, n)``
        Square matrix or batch of square matrices.

    Returns
    -------
    Array, shape ``(*batch, n, n)``
        Matrix inverse of each square matrix.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from unxts.linalg import inv

    Plain 2x2 diagonal matrix:

    >>> inv(jnp.array([[2.0, 0.0], [0.0, 4.0]]))
    Array([[0.5 , 0.  ],
           [0.  , 0.25]], dtype=float32)

    Under JIT:

    >>> import jax
    >>> jax.jit(inv)(jnp.array([[2.0, 0.0], [0.0, 4.0]]))
    Array([[0.5 , 0.  ],
           [0.  , 0.25]], dtype=float32)

    Gradient (via reverse-mode autodiff) — returns a rank-4 Jacobian:

    >>> jac = jax.jacobian(inv)(jnp.array([[2.0, 0.0], [0.0, 4.0]]))
    >>> jac.shape
    (2, 2, 2, 2)

    Batched (vmap):

    >>> A = jnp.stack(
    ...     [jnp.diag(jnp.array([2.0, 4.0])), jnp.diag(jnp.array([1.0, 2.0]))]
    ... )
    >>> jax.vmap(inv)(A)
    Array([[[0.5 , 0.  ],
            [0.  , 0.25]],
    <BLANKLINE>
           [[1.  , 0.  ],
            [0.  , 0.5 ]]], dtype=float32)

    """
    return inv_p.bind(x)


# ── 1. Primal evaluation rule ─────────────────────────────────────────────


def _inv_impl(x: Array, /) -> Array:
    return jnp.linalg.inv(x)


inv_p.def_impl(_inv_impl)


# ── 2. Abstract evaluation rule ───────────────────────────────────────────


def _inv_abstract_eval(x: "jax.core.ShapedArray", /) -> "jax.core.ShapedArray":  # ty: ignore[possibly-missing-submodule]
    if x.ndim < 2:
        raise ValueError(f"inv_p requires at least 2-D input, got ndim={x.ndim}")
    if x.shape[-1] != x.shape[-2]:
        raise ValueError(
            f"inv_p requires a square matrix "
            f"(shape[-2] == shape[-1]), got shape={x.shape}"
        )
    return x.update(shape=x.shape)  # same shape as input


inv_p.def_abstract_eval(_inv_abstract_eval)


# ── 3. MLIR / XLA lowering ────────────────────────────────────────────────

jax_mlir.register_lowering(
    inv_p,
    jax_mlir.lower_fun(_inv_impl, multiple_results=False),
)


# ── 4. Forward-mode differentiation (JVP) ────────────────────────────────
#
# d(A^{-1})(dA) = -A^{-1} dA A^{-1}


def _inv_jvp(primals: tuple, tangents: tuple) -> tuple[Array, Array]:
    (x,) = primals
    (dx,) = tangents
    primal_out = inv_p.bind(x)
    if type(dx) is jax_ad.Zero:
        tangent_out = jax_ad.Zero.from_primal_value(primal_out)  # ty: ignore[unresolved-attribute]
    else:
        tangent_out = -primal_out @ dx @ primal_out
    return primal_out, tangent_out


jax_ad.primitive_jvps[inv_p] = _inv_jvp


# ── 5. Batching rule (vmap) ───────────────────────────────────────────────


def _inv_batch(args: tuple, batch_axes: tuple) -> tuple:
    (x,) = args
    (ax,) = batch_axes
    x = jnp.moveaxis(x, ax, 0)
    return inv_p.bind(x), 0


jax_batching.primitive_batchers[inv_p] = _inv_batch


# ── 6. Quax dispatch for QuantityMatrix ──────────────────────────────────


@quax.register(inv_p)
def _inv_p_QuantityMatrix(x: QuantityMatrix, /) -> QuantityMatrix:
    """Compute the inverse of a 2-D :class:`~unxts.linalg.QuantityMatrix`.

    The numeric value is computed via ``inv_p.bind(x.value)``.
    Units must be uniform (all entries share the same physical unit, as is the
    case for metrics produced by the Cartesian-Jacobian pullback); the inverse
    then carries the reciprocal unit throughout. A matrix inverse mixes the
    entries, so for heterogeneous units the per-element reciprocal is not the
    correct unit structure — that case raises ``ValueError``.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import quax
    >>> import unxt as u
    >>> from unxts.linalg import QuantityMatrix, UnitsMatrix
    >>> from unxts.linalg import inv

    >>> A = QuantityMatrix(
    ...     jnp.array([[4.0, 0.0], [0.0, 1.0]]),
    ...     unit=UnitsMatrix((("m2", "m2"), ("m2", "m2"))),
    ... )
    >>> quax.quaxify(inv)(A)
    QuantityMatrix(
        Array([[0.25, 0.  ],
               [0.  , 1.  ]], dtype=float32), unit='((1 / m2, 1 / m2), (1 / m2, 1 / m2))'
    )

    """
    if x.ndim != 2:
        raise ValueError(
            f"inv_p QuantityMatrix dispatch requires a 2-D unit structure, got ndim={x.ndim}"
        )

    # The reciprocal-unit result is only correct when the units are uniform;
    # a matrix inverse mixes entries, so heterogeneous units are not simply
    # reciprocated element-by-element.
    flat = x.unit._units.ravel()
    if any(unit_i != flat[0] for unit_i in flat[1:]):
        msg = (
            "inv on a QuantityMatrix requires uniform units (all entries equal); "
            "the inverse of a heterogeneous-unit matrix is not an element-wise "
            f"reciprocal. Got units: {x.unit.to_string()}."
        )
        raise ValueError(msg)

    inv_val = inv_p.bind(x.value)
    return QuantityMatrix(inv_val, unit=x.unit.inverse())
