"""QuantityMatrix class and unit-conversion helpers."""

from typing import Any, NoReturn

import equinox as eqx
import jax
import jax.core
import jax.numpy as jnp
import jax.tree_util as jtu
import plum
from jaxtyping import Array, Shaped

import unxt as u
from ._units_matrix import UnitsMatrix
from ._utils import _DMLS, CDict, strict_zip
from unxt.quantity import AllowValue


class QuantityMatrix(u.AbstractQuantity):
    """Quantity container whose elements may each carry different units.

    `QuantityMatrix` stores one numeric array together with a static
    `UnitsMatrix` describing the unit of each logical element. The shape of the
    unit structure determines whether the object behaves as a heterogeneous
    vector or matrix.

    Only 1-D and 2-D logical structures are supported.

    Parameters
    ----------
    value : Array, shape ``(..., *shape)``
        Numeric payload. For 1D: ``(..., N)``. For 2D: ``(..., N, M)``.
        The value of element ``[i]`` (1D) or ``[i, j]`` (2D) is expressed
        in the corresponding unit.
    unit : UnitsMatrix
        Per-element units. For 1D: ``(u0, u1, ...)``.
        For 2D: ``((u00, u01, ...), (u10, u11, ...), ...)``.
        Must be a static (hashable) nested tuple structure whose shape
        matches the trailing dimensions of ``value``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> from unxts.linalg import QuantityMatrix

    1D case (vector):

    >>> qv = QuantityMatrix(jnp.array([1.0, 2.0, 3.0]), unit=("m", "s", "kg"))
    >>> qv.value
    Array([1., 2., 3.], dtype=float32)
    >>> qv.unit.shape
    (3,)

    >>> 2 * qv
    QuantityMatrix(Array([2., 4., 6.], dtype=float32), unit='(m, s, kg)')

    >>> qv2 = QuantityMatrix(jnp.array([0.1, 200.0, 300.0]), unit=("km", "ms", "g"))
    >>> qv + qv2
    QuantityMatrix(Array([101. ,   2.2,   3.3], dtype=float32), unit='(m, s, kg)')

    2D case (matrix):

    >>> qm = QuantityMatrix(jnp.ones((2, 2)), unit=(("m", "s"), ("kg", "rad")))
    >>> qm.value.shape
    (2, 2)
    >>> qm.unit.shape
    (2, 2)

    >>> 2 * qm
    QuantityMatrix(Array([[2., 2.],
                          [2., 2.]], dtype=float32), unit='((m, s), (kg, rad))')

    >>> qm2 = QuantityMatrix(
    ...     jnp.array([[0.1, 200.0], [300.0, 0.5]]), unit=(("km", "ms"), ("g", "deg"))
    ... )
    >>> qm + qm2
    QuantityMatrix(
        Array([[101.       ,   1.2      ],
               [  1.3      ,   1.0087266]], dtype=float32), unit='((m, s), (kg, rad))'
    )

    Indexing:

    >>> qv[0]
    Quantity(Array(1., dtype=float32), unit='m')
    >>> qm[0]
    QuantityMatrix(Array([1., 1.], dtype=float32), unit='(m, s)')
    >>> qm[1, 0]
    Quantity(Array(1., dtype=float32), unit='kg')

    """

    value: Shaped[Array, "..."] = eqx.field()
    unit: UnitsMatrix = eqx.field(static=True, converter=u.unit)  # ty: ignore[invalid-assignment]

    @property
    def ndim(self) -> int:
        """Number of real dimensions (1 for vector, 2 for matrix)."""
        return self.unit.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape, including batch dimensions."""
        return self.value.shape

    @classmethod
    def from_cdict(
        cls, v: CDict, /, keys: tuple[str, ...] | None = None
    ) -> "QuantityMatrix":
        """Pack a component dictionary into a 1-D ``QuantityMatrix``.

        Each value in *v* is stripped to its numeric value and stacked into a
        single JAX array.  Values that carry units (``unxt.Quantity``) retain
        those units in the resulting ``UnitsMatrix``; plain arrays are treated
        as dimensionless.

        Examples
        --------
        >>> import unxt as u
        >>> from unxts.linalg import QuantityMatrix

        From a dictionary of quantities:

        >>> v = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "s"), "z": u.Q(3.0, "kg")}
        >>> qv = QuantityMatrix.from_cdict(v)
        >>> qv.unit.to_string()
        '(m, s, kg)'
        >>> qv.value
        Array([1., 2., 3.], dtype=float32)

        Selecting and reordering a subset of keys:

        >>> qv2 = QuantityMatrix.from_cdict(v, keys=("z", "x"))
        >>> qv2.unit.to_string()
        '(kg, m)'
        >>> qv2.value
        Array([3., 1.], dtype=float32)

        Dimensionless entries (bare arrays) are accepted:

        >>> import jax.numpy as jnp
        >>> v2 = {"a": jnp.array(4.0), "b": u.Q(5.0, "m")}
        >>> qv3 = QuantityMatrix.from_cdict(v2)
        >>> qv3.unit.to_string()
        '(, m)'

        """
        keys = tuple(v) if keys is None else keys
        vs = [v[k] for k in keys]
        us = [u.unit_of(x) or _DMLS for x in vs]
        svs = jnp.stack([u.ustrip(AllowValue, unt, x) for x, unt in strict_zip(vs, us)])
        return cls(svs, unit=UnitsMatrix(us))

    def __getitem__(self, index: Any, /) -> "u.Q | QuantityMatrix":  # ty: ignore[invalid-method-override]
        """Index into the QuantityMatrix to retrieve a specific element.

        Indexing a logical dimension returns a ``Quantity`` when the result is
        a scalar unit, or a ``QuantityMatrix`` when the result still has
        structure.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import unxt as u
        >>> from unxts.linalg import QuantityMatrix

        **1-D vector** — indexing a single element returns a ``Quantity``:

        >>> qv = QuantityMatrix(jnp.array([1.0, 2.0, 3.0]), unit=("m", "s", "kg"))
        >>> qv[0]
        Quantity(Array(1., dtype=float32), unit='m')
        >>> qv[2]
        Quantity(Array(3., dtype=float32), unit='kg')

        **2-D matrix** — indexing a row returns a ``QuantityMatrix``:

        >>> qm = QuantityMatrix(
        ...     jnp.ones((2, 3)), unit=(("m", "s", "kg"), ("rad", "deg", "m"))
        ... )
        >>> qm[0]
        QuantityMatrix(Array([1., 1., 1.], dtype=float32), unit='(m, s, kg)')

        Indexing a specific element returns a ``Quantity``:

        >>> qm[1, 2]
        Quantity(Array(1., dtype=float32), unit='m')

        """
        value_item = self.value[index]
        unit_item = self.unit[index]
        if isinstance(unit_item, UnitsMatrix):
            return QuantityMatrix(value=value_item, unit=unit_item)
        return u.Q(value_item, unit_item)

    # ── Quax API ─────────────────────────────────────────────────────

    def aval(self) -> jax.core.ShapedArray:
        return jax.core.ShapedArray(self.value.shape, self.value.dtype)

    def materialise(self) -> NoReturn:
        msg = "Refusing to materialise `QuantityMatrix`."
        raise RuntimeError(msg)

    def diag(self) -> "QuantityMatrix":
        """Return a 1-D ``QuantityMatrix`` containing the diagonal of this matrix.

        Unlike ``qnp.diag``, this method operates directly on the static
        ``unit`` structure and the raw value array, so it works correctly under
        ``jax.jit`` and with heterogeneous-unit matrices.

        Only supported for 2-D ``QuantityMatrix`` objects.

        Returns
        -------
        QuantityMatrix
            1-D ``QuantityMatrix`` of length ``min(n_rows, n_cols)`` whose
            ``unit[i]`` is ``self.unit[i, i]`` and whose ``value[..., i]`` is
            ``self.value[..., i, i]``.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from unxts.linalg import QuantityMatrix

        Uniform units:

        >>> A = QuantityMatrix(
        ...     jnp.diag(jnp.array([1.0, 4.0, 9.0])),
        ...     unit=(("m", "m", "m"), ("m", "m", "m"), ("m", "m", "m")),
        ... )
        >>> d = A.diag()
        >>> d.unit.shape
        (3,)
        >>> d.value
        Array([1., 4., 9.], dtype=float32)

        Heterogeneous units — works under jit:

        >>> B = QuantityMatrix(
        ...     jnp.diag(jnp.array([1.0, 2.0, 3.0])),
        ...     unit=(("m", "s", "kg"), ("m", "s", "kg"), ("m", "s", "kg")),
        ... )
        >>> db = B.diag()
        >>> db.unit.to_string()
        '(m, s, kg)'
        >>> db.value
        Array([1., 2., 3.], dtype=float32)

        """
        if self.ndim != 2:
            raise ValueError(
                f"QuantityMatrix.diag() requires a 2D matrix, got ndim={self.ndim}"
            )
        # `jnp.diagonal` extracts the main diagonal of the last two axes in a
        # single primitive (result diagonal is the trailing axis), avoiding an
        # n-op Python loop; units come from the static `UnitsMatrix`.
        diag_value = jnp.diagonal(self.value, axis1=-2, axis2=-1)
        diag_unit = UnitsMatrix(self.unit._units.diagonal())
        return QuantityMatrix(value=diag_value, unit=diag_unit)

    @property
    def T(self) -> "QuantityMatrix":
        """Transpose a 2-D ``QuantityMatrix`` (swap rows/columns and units).

        Returns a new ``QuantityMatrix`` whose value array and unit structure
        are both transposed.  Only 2-D matrices are supported.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import quaxed.numpy as qnp
        >>> from unxts.linalg import QuantityMatrix

        >>> a = QuantityMatrix(
        ...     jnp.array([[1.0, 2.0], [3.0, 4.0]]), unit=(("m", "s"), ("kg", "rad"))
        ... )
        >>> aT = a.T
        >>> aT.value
        Array([[1., 3.],
               [2., 4.]], dtype=float32)
        >>> aT.unit.to_string()
        '((m, kg), (s, rad))'

        Also accessible via ``jax.numpy.transpose``:

        >>> aT2 = qnp.matrix_transpose(a)
        >>> aT2.value
        Array([[1., 3.],
               [2., 4.]], dtype=float32)
        >>> aT2.unit.to_string()
        '((m, kg), (s, rad))'

        """
        if self.ndim != 2:
            msg = f"QuantityMatrix.T requires a 2-D matrix, got ndim={self.ndim}"
            raise ValueError(msg)
        return QuantityMatrix(value=jnp.swapaxes(self.value, -2, -1), unit=self.unit.T)


QM = QuantityMatrix
"""Short alias for `QuantityMatrix` (cf. ``Q`` for ``Quantity``)."""


##############################################################################
# Unit-conversion helpers


def _convert_value_vector(
    value: Shaped[Array, "*batch N"],
    from_units: tuple[u.AbstractUnit, ...],
    to_units: tuple[u.AbstractUnit, ...],
) -> Shaped[Array, "*batch N"]:
    """Convert every element of *value* from *from_units* to *to_units* (1D case).

    Each ``value[..., i]`` is converted individually via
    `u.uconvert_value` so that **all** conversion types are handled
    correctly.
    """
    n = len(to_units)
    return jnp.stack(
        [u.uconvert_value(to_units[i], from_units[i], value[..., i]) for i in range(n)],
        axis=-1,
    )


def _convert_value_matrix(
    value: Shaped[Array, "*batch N M"],
    from_units: tuple[tuple[u.AbstractUnit, ...], ...],
    to_units: tuple[tuple[u.AbstractUnit, ...], ...],
) -> Shaped[Array, "*batch N M"]:
    """Convert every element of *value* from *from_units* to *to_units* (2D case).

    Each ``value[..., i, j]`` is converted individually via
    `u.uconvert_value` so that **all** conversion types are handled
    correctly — including nonlinear ones like dB, mag, and dex (which
    are logarithmic, not affine).
    """
    n = len(to_units)
    m = len(to_units[0])
    return jnp.stack(
        [
            jnp.stack(
                [
                    u.uconvert_value(to_units[i][j], from_units[i][j], value[..., i, j])
                    for j in range(m)
                ],
                axis=-1,
            )
            for i in range(n)
        ],
        axis=-2,
    )


@plum.conversion_method(type_from=QuantityMatrix, type_to=u.Q)
def QuantityMatrix_to_quantity(x: QuantityMatrix, /) -> u.Q:
    """Convert a ``QuantityMatrix`` to a regular ``Quantity``.

    Conversion is only valid when all elements of ``x`` share the same unit.  If
    units are heterogeneous, this conversion is ambiguous and raises
    ``ValueError``.

    >>> import plum
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> from unxts.linalg import QuantityMatrix

    Uniform units convert to a plain quantity:

    >>> qmat = QuantityMatrix(jnp.array([1.0, 2.0, 3.0]), unit=("m", "m", "m"))
    >>> plum.convert(qmat, u.Q)
    Quantity(Array([1., 2., 3.], dtype=float32), unit='m')

    Mixed units are rejected:

    >>> bad = QuantityMatrix(jnp.array([1.0, 2.0]), unit=("m", "s"))
    >>> plum.convert(bad, u.Q)
    Traceback (most recent call last):
    ...
    ValueError: Cannot convert QuantityMatrix to Quantity unless all units are
    identical.

    """
    units = jtu.tree_leaves(x.unit.to_tuple())

    if not units:
        msg = "Cannot convert QuantityMatrix with no unit entries."
        raise ValueError(msg)

    first = units[0]
    if any(unit != first for unit in units[1:]):
        msg = (
            "Cannot convert QuantityMatrix to Quantity unless all units are identical."
        )
        raise ValueError(msg)

    return u.Q(x.value, first)


def _convert_value(
    value: Array,
    from_units: UnitsMatrix,
    to_units: UnitsMatrix,
) -> Array:
    """Convert value with heterogeneous units (works for both 1D and 2D)."""
    from_tup = from_units.to_tuple()
    to_tup = to_units.to_tuple()
    if from_units.ndim == 1:
        return _convert_value_vector(value, from_tup, to_tup)
    if from_units.ndim == 2:
        return _convert_value_matrix(value, from_tup, to_tup)
    msg = f"Unsupported ndim={from_units.ndim}"
    raise NotImplementedError(msg)


@plum.dispatch
def uconvert(to_units: UnitsMatrix, x: QuantityMatrix, /) -> QuantityMatrix:
    """Convert a ``QuantityMatrix`` to different (but compatible) units.

    Unlike the generic astropy ``StructuredUnit.to()`` path, this dispatch uses
    ``_convert_value`` directly so that the regular 2D JAX array in ``x.value``
    is converted element-by-element without requiring a numpy structured array.

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> from unxts.linalg import QuantityMatrix

    >>> x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    >>> q = QuantityMatrix(x, (("m", "rad"), ("m", "rad")))
    >>> target = u.unit((("km", "deg"), ("km", "deg")))
    >>> q.uconvert(target).unit.to_string()
    '((km, deg), (km, deg))'

    """
    if x.unit == to_units:
        return x
    value = _convert_value(x.value, x.unit, to_units)
    return QuantityMatrix(value=value, unit=to_units)
