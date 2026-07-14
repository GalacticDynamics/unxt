"""Batch-safe matrix/vector products for `QuantityMatrix`.

These mirror the NumPy / Array-API family of contraction functions —
``matmul`` (matrix @ matrix), ``matvec`` (matrix @ vector), ``vecmat``
(vector @ matrix), and ``vecdot`` (vector · vector) — each dispatching to the
`QuantityMatrix` Quax rules registered in ``_register_primitives``.

Why four functions instead of just ``matmul``: a `QuantityMatrix` carries its
logical 1-D/2-D unit structure in the *trailing* axes and treats any *leading*
axes of the value array as batch dimensions. ``matmul`` uses NumPy's
shape-broadcasting, under which a batched vector value ``(B, K)`` is
indistinguishable from a matrix ``(B, K)`` — so batched matrix-vector products
cannot be expressed with ``matmul``. ``matvec``/``vecmat`` name the operand
ranks explicitly and therefore broadcast the batch axis correctly.

Each wrapper accepts `QuantityMatrix`, `unxt.Quantity`, or plain arrays (via
``quax``) and returns a `QuantityMatrix` (or a scalar `unxt.Quantity` for
``vecdot``).
"""

from typing import Any

import jax.numpy as jnp
import quax

from ._quantity_matrix import QuantityMatrix

__all__ = ("matmul", "matvec", "vecdot", "vecmat")


def _reject_batched_vector(a: Any, b: Any, /) -> None:
    """Raise if a `QuantityMatrix` operand is a *batched* (logically 1-D) vector.

    ``matmul`` infers its contraction from NumPy shape-broadcasting, under which
    a batched vector's value ``(*batch, K)`` is indistinguishable from a matrix.
    So a batched matrix-vector product either contracts the wrong axes (raising
    only when the sizes happen not to line up) or silently succeeds — a trap.
    We reject it up front and point at the rank-explicit products instead.
    """
    for x in (a, b):
        if isinstance(x, QuantityMatrix) and x.unit.ndim == 1 and x.value.ndim > 1:
            msg = (
                "matmul cannot express a batched matrix-vector product: a "
                "batched vector's value shape (*batch, K) is ambiguous with a "
                "matrix. Use unxts.linalg.matvec / vecmat / vecdot instead."
            )
            raise ValueError(msg)


def matmul(a: Any, b: Any, /) -> Any:
    """Matrix-matrix product ``a @ b`` (batch dims broadcast over leading axes).

    Only matrix operands may carry leading batch axes. A *batched* vector
    operand is rejected (see :func:`matvec` / :func:`vecmat` / :func:`vecdot`),
    because ``matmul`` cannot disambiguate its shape from a matrix.

    >>> import jax.numpy as jnp
    >>> import unxts.linalg as ul

    >>> A = ul.QuantityMatrix(
    ...     jnp.array([[1.0, 2.0], [3.0, 4.0]]), unit=(("m", "m"), ("m", "m"))
    ... )
    >>> B = ul.QuantityMatrix(
    ...     jnp.array([[1.0, 0.0], [0.0, 1.0]]), unit=(("s", "s"), ("s", "s"))
    ... )
    >>> ul.matmul(A, B).unit.to_string()
    '((m s, m s), (m s, m s))'

    A batched matrix-vector product is rejected with a helpful pointer:

    >>> Ab = ul.QuantityMatrix(jnp.ones((2, 2, 2)), unit=(("m", "m"), ("m", "m")))
    >>> vb = ul.QuantityMatrix(jnp.ones((2, 2)), unit=("s", "s"))
    >>> ul.matmul(Ab, vb)
    Traceback (most recent call last):
    ...
    ValueError: matmul cannot express a batched matrix-vector product: ...

    """
    _reject_batched_vector(a, b)
    return quax.quaxify(jnp.matmul)(a, b)


def matvec(a: Any, b: Any, /) -> Any:
    """Matrix-vector product: ``(..., N, K) @ (..., K) -> (..., N)``.

    Unlike :func:`matmul`, this broadcasts a leading batch axis on **both**
    operands, so a batch of matrices can be applied to a batch of vectors.

    >>> import jax.numpy as jnp
    >>> import unxts.linalg as ul

    >>> A = ul.QuantityMatrix(
    ...     jnp.array([[1.0, 2.0], [3.0, 4.0]]), unit=(("m", "m"), ("m", "m"))
    ... )
    >>> v = ul.QuantityMatrix(jnp.array([1.0, 1.0]), unit=("s", "s"))
    >>> ul.matvec(A, v).value
    Array([3., 7.], dtype=float32)

    Batched — a stack of matrices times a stack of vectors:

    >>> Ab = ul.QuantityMatrix(
    ...     jnp.broadcast_to(A.value, (2, 2, 2)), unit=(("m", "m"), ("m", "m"))
    ... )
    >>> vb = ul.QuantityMatrix(jnp.ones((2, 2)), unit=("s", "s"))
    >>> ul.matvec(Ab, vb).value
    Array([[3., 7.],
           [3., 7.]], dtype=float32)

    """
    return quax.quaxify(jnp.matvec)(a, b)


def vecmat(a: Any, b: Any, /) -> Any:
    """Vector-matrix product: ``(..., K) @ (..., K, M) -> (..., M)``.

    The transpose of :func:`matvec`; also broadcasts a leading batch axis on
    both operands.

    >>> import jax.numpy as jnp
    >>> import unxts.linalg as ul

    >>> v = ul.QuantityMatrix(jnp.array([1.0, 1.0]), unit=("s", "s"))
    >>> A = ul.QuantityMatrix(
    ...     jnp.array([[1.0, 2.0], [3.0, 4.0]]), unit=(("m", "km"), ("m", "km"))
    ... )
    >>> ul.vecmat(v, A).value
    Array([4., 6.], dtype=float32)

    """
    return quax.quaxify(jnp.vecmat)(a, b)


def vecdot(a: Any, b: Any, /) -> Any:
    """Vector dot product: ``(..., K) · (..., K) -> (...)`` (a scalar quantity).

    Broadcasts a leading batch axis on both operands.

    >>> import jax.numpy as jnp
    >>> import unxts.linalg as ul

    >>> a = ul.QuantityMatrix(jnp.array([1.0, 2.0]), unit=("m", "km"))
    >>> b = ul.QuantityMatrix(jnp.array([3.0, 4.0]), unit=("s", "s"))
    >>> ul.vecdot(a, b).value
    Array(8003., dtype=float32)

    """
    return quax.quaxify(jnp.vecdot)(a, b)
