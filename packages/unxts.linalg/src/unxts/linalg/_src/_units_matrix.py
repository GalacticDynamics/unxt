"""Immutable, hashable unit structure for QuantityMatrix."""

import operator
from typing import Any, TypeAlias, TypeVar, final

import numpy as np
import plum

import unxt as u

T = TypeVar("T")

NestedTuple: TypeAlias = T | tuple["NestedTuple[T]", ...]
UnitTree: TypeAlias = NestedTuple[u.AbstractUnit]


def _normalize_unit(x: Any, /) -> u.AbstractUnit:
    """Convert *x* to an ``AbstractUnit``; accept unit strings and AbstractUnit.

    Raises ``TypeError`` for unsupported types.
    """
    if isinstance(x, str):
        return u.unit(x)  # ty: ignore[invalid-return-type]
    if isinstance(x, u.AbstractUnit):
        return x
    msg = f"Expected an AbstractUnit or unit string; got {type(x).__name__!r}"
    raise TypeError(msg)


def _split_top_level(inner: str, /) -> list[str]:
    """Split *inner* on commas that are not enclosed in parentheses.

    Unit strings never contain commas but may contain (balanced) parentheses
    — e.g. ``"m / (kg s)"`` — so a depth-aware split unambiguously separates the
    elements/rows of a structural string.
    """
    parts: list[str] = []
    depth = 0
    buf: list[str] = []
    for ch in inner:
        if ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth -= 1
            if depth < 0:
                msg = f"Unbalanced parentheses (unmatched ')'): {inner!r}"
                raise ValueError(msg)
            buf.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if depth != 0:
        msg = f"Unbalanced parentheses (unmatched '('): {inner!r}"
        raise ValueError(msg)
    tail = "".join(buf).strip()
    if tail:  # drop the empty tail from a trailing comma, e.g. "(m,)"
        parts.append(tail)
    return parts


def _parse_structure_string(s: str, /) -> tuple[Any, ...]:
    """Parse a `to_string()` / `__repr__` structural string into a nested tuple.

    e.g. ``"(m, s)" -> ("m", "s")`` and
    ``"((m, s), (kg, rad))" -> (("m", "s"), ("kg", "rad"))``. A top-level part
    that is itself parenthesized is a nested row (2-D); otherwise it is a unit
    string (units never begin with ``"("``).
    """
    out: list[Any] = []
    for part in _split_top_level(s[1:-1]):  # drop the outer parentheses
        if part.startswith("(") and part.endswith(")"):
            out.append(_parse_structure_string(part))
        else:
            out.append(part)
    return tuple(out)


def _build_object_array(iterable: Any, /) -> np.ndarray:  # noqa: C901
    """Build a 1-D or 2-D numpy object array of ``AbstractUnit`` from *iterable*.

    Accepts:

    - A numpy object array (element-normalize and validate ndim).
    - A plain tuple/list of units or unit strings → 1-D output.
    - A plain tuple/list of tuples of units or unit strings → 2-D output.

    Raises ``TypeError`` if a non-sequence is passed, ``ValueError`` if the
    structure is ragged or has unsupported ndim.
    """
    if isinstance(iterable, np.ndarray) and iterable.dtype == object:
        if iterable.ndim not in (1, 2):
            msg = f"UnitsMatrix only supports 1D or 2D; got ndim={iterable.ndim}"
            raise ValueError(msg)
        flat = [_normalize_unit(v) for v in iterable.flat]
        data: np.ndarray = np.empty(iterable.shape, dtype=object)
        data.flat[:] = flat
        return data

    if isinstance(iterable, str):
        # Accept the structural form produced by ``to_string()`` / ``__repr__``
        # (e.g. "(m, s)"), so a repr round-trips. A *bare* unit string is
        # rejected: it is iterable and would otherwise be split into
        # per-character "units" (e.g. "ms" -> ("m", "s")); wrap it in a tuple.
        stripped = iterable.strip()
        if stripped.startswith("(") and stripped.endswith(")"):
            return _build_object_array(_parse_structure_string(stripped))
        msg = (
            f"UnitsMatrix does not accept a bare unit string ({iterable!r}); wrap "
            f"a single unit in a tuple, e.g. ({iterable!r},)."
        )
        raise TypeError(msg)

    # Sequence path: tuple, list, or any iterable
    items = list(iterable)  # raises TypeError if not iterable

    if not items:
        raise ValueError("UnitsMatrix requires at least one element")

    first = items[0]
    if isinstance(first, (tuple, list)):
        # 2-D: sequence of rows — validate and fill in one pass
        n, m = len(items), len(first)
        data = np.empty((n, m), dtype=object)
        for i, row in enumerate(items):
            if not isinstance(row, (tuple, list)) or len(row) != m:
                raise ValueError("ragged structure")
            for j, v in enumerate(row):
                if isinstance(v, (tuple, list)):
                    raise ValueError("ragged structure")
                data[i, j] = _normalize_unit(v)
        return data

    # 1-D: sequence of units / unit strings
    n = len(items)
    data = np.empty(n, dtype=object)
    for i, v in enumerate(items):
        if isinstance(v, (tuple, list)):  # Mixed leaf/nested → ragged
            raise ValueError("ragged structure")
        data[i] = _normalize_unit(v)
    return data


@final
class UnitsMatrix:
    """Immutable, hashable unit structure for `QuantityMatrix`.

    `UnitsMatrix` wraps a numpy object array (``dtype=object``) of
    `~unxt.AbstractUnit` elements. Only 1-D and 2-D structures are accepted.

    The class supports tuple-style indexing, iteration, `to_tuple()`, and
    `to_string()`. It is **not** a subclass of `astropy.units.StructuredUnit`.

    Hashability is achieved via ``hash(self.to_tuple())``, so the underlying
    ``AbstractUnit`` objects must themselves be hashable (they are).

    For 1D: ``UnitsMatrix(("m", "s", "kg"))``
    For 2D: ``UnitsMatrix((("m", "s"), ("kg", "rad")))``

    Examples
    --------
    >>> import unxt as u
    >>> from unxts.linalg import UnitsMatrix

    1D case:

    >>> units_1d = UnitsMatrix(("m", "s", "kg"))
    >>> units_1d.shape
    (3,)
    >>> units_1d[0]
    Unit("m")
    >>> units_1d.to_string()
    '(m, s, kg)'

    2D case:

    >>> units_2d = UnitsMatrix((("m", "s"), ("kg", "rad")))
    >>> units_2d.shape
    (2, 2)
    >>> units_2d[0, 1]
    Unit("s")
    >>> units_2d.to_string()
    '((m, s), (kg, rad))'

    """

    __slots__ = ("_units",)

    def __init__(self, iterable: Any, /) -> None:
        if isinstance(iterable, UnitsMatrix):
            # Copy from another UnitsMatrix — avoids sharing the mutable array.
            data = iterable._units.copy()
        else:
            data = _build_object_array(iterable)
        if data.ndim not in (1, 2):
            msg = f"UnitsMatrix only supports 1D or 2D, but got ndim={data.ndim}"
            raise ValueError(msg)
        self._units = data

    # ── Shape / structure ─────────────────────────────────────────────

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the N-D unit structure."""
        return tuple(self._units.shape)

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return int(self._units.ndim)

    @property
    def T(self) -> "UnitsMatrix":
        """Compute the all-axis units array transpose.

        Examples
        --------
        >>> from unxts.linalg import UnitsMatrix

        >>> units = UnitsMatrix(("m", "s"))
        >>> units.T
        UnitsMatrix("(m, s)")

        >>> units = UnitsMatrix((("m", "s"), ("kg", "rad")))
        >>> units.T
        UnitsMatrix("((m, kg), (s, rad))")

        >>> units = UnitsMatrix((("m", "s", "kg"), ("Hz", "candela", "km")))
        >>> units.T
        UnitsMatrix("((m, Hz), (s, cd), (kg, km))")

        """
        return UnitsMatrix(self._units.T)

    def inverse(self) -> "UnitsMatrix":
        r"""Inverse unit structure — each unit raised to the power -1.

        For a 1-D (diagonal) ``UnitsMatrix`` the inversion is done
        entry-by-entry in *O(n)*, providing a speedup over the general 2-D
        case.  For a 2-D ``UnitsMatrix`` with a uniform unit (all entries
        equal) the reciprocal is computed once and broadcast in *O(1)*;
        mixed-unit 2-D structures fall back to an element-wise *O(nm)* loop.

        Examples
        --------
        >>> from unxts.linalg import UnitsMatrix

        1-D (diagonal) case — element-wise reciprocal:

        >>> UnitsMatrix(("m2", "s2")).inverse()
        UnitsMatrix("(1 / m2, 1 / s2)")

        2-D uniform-unit case:

        >>> UnitsMatrix((("m2", "m2"), ("m2", "m2"))).inverse()
        UnitsMatrix("((1 / m2, 1 / m2), (1 / m2, 1 / m2))")

        2-D mixed-unit case:

        >>> UnitsMatrix((("m2", "s2"), ("s2", "rad2"))).inverse()
        UnitsMatrix("((1 / m2, 1 / s2), (1 / s2, 1 / rad2))")

        """
        inv_data = np.empty(self._units.shape, dtype=object)
        if self._units.ndim == 1:
            # Diagonal speedup: 1-D represents a diagonal metric's units.
            for i in range(self._units.shape[0]):
                inv_data[i] = self._units[i] ** (-1)
        else:
            # 2-D: fast path when all entries share the same unit.
            flat = self._units.ravel()
            first = flat[0]
            if all(u == first for u in flat[1:]):
                inv_unit = first ** (-1)
                inv_data[:] = inv_unit
            else:
                n, m = self._units.shape
                for i in range(n):
                    for j in range(m):
                        inv_data[i, j] = self._units[i, j] ** (-1)
        return UnitsMatrix(inv_data)

    # ── Element-wise arithmetic ───────────────────────────────────────

    def _elementwise(self, other: Any, op: Any, /) -> "UnitsMatrix":
        """Apply a binary unit op element-wise against a unit or a UnitsMatrix."""
        if isinstance(other, UnitsMatrix):
            if self.shape != other.shape:
                msg = f"UnitsMatrix shape mismatch: {self.shape} vs {other.shape}."
                raise ValueError(msg)
            other_units = other._units
        elif isinstance(other, u.AbstractUnit):
            # Broadcast the scalar unit into an object array so the op runs
            # element-wise on unit objects (astropy mishandles ndarray * Unit).
            other_units = np.empty(self._units.shape, dtype=object)
            other_units[...] = other
        else:
            return NotImplemented  # ty: ignore[invalid-return-type]
        return UnitsMatrix(op(self._units, other_units))

    def __mul__(self, other: Any, /) -> "UnitsMatrix":
        """Element-wise unit product (against a unit or another UnitsMatrix).

        >>> from unxts.linalg import UnitsMatrix
        >>> import unxt as u
        >>> UnitsMatrix(("m", "s")) * u.unit("kg")
        UnitsMatrix("(kg m, kg s)")
        >>> UnitsMatrix(("m", "s")) * UnitsMatrix(("s", "s"))
        UnitsMatrix("(m s, s2)")

        """
        return self._elementwise(other, operator.mul)

    def __rmul__(self, other: Any, /) -> "UnitsMatrix":
        # Unit multiplication commutes.
        return self._elementwise(other, operator.mul)

    def __truediv__(self, other: Any, /) -> "UnitsMatrix":
        """Element-wise unit quotient (against a unit or another UnitsMatrix).

        >>> from unxts.linalg import UnitsMatrix
        >>> import unxt as u
        >>> UnitsMatrix(("m2", "s2")) / u.unit("s")
        UnitsMatrix("(m2 / s, s)")

        """
        return self._elementwise(other, operator.truediv)

    def __rtruediv__(self, other: Any, /) -> "UnitsMatrix":
        return self._elementwise(other, lambda a, b: operator.truediv(b, a))

    # ── Serialization ─────────────────────────────────────────────────

    def to_tuple(self) -> UnitTree:
        """Convert to a nested tuple of `~unxt.AbstractUnit` objects.

        Examples
        --------
        >>> from unxts.linalg import UnitsMatrix
        >>> import unxt as u
        >>> UnitsMatrix(("m", "s")).to_tuple()
        (Unit("m"), Unit("s"))

        """
        if self._units.ndim == 1:
            return tuple(self._units)
        return tuple(map(tuple, self._units))

    def to_string(self) -> str:
        """Return a human-readable string representation of the unit structure.

        Examples
        --------
        >>> from unxts.linalg import UnitsMatrix
        >>> UnitsMatrix(("m", "s", "kg")).to_string()
        '(m, s, kg)'
        >>> UnitsMatrix((("m", "s"), ("kg", "rad"))).to_string()
        '((m, s), (kg, rad))'

        """
        if self._units.ndim == 1:
            inner = ", ".join(str(x) for x in self._units)
            if len(self._units) == 1:
                return f"({inner},)"
            return f"({inner})"
        # 2D
        row_strs = []
        for row in self._units:
            inner = ", ".join(str(x) for x in row)
            row_strs.append(f"({inner},)" if len(row) == 1 else f"({inner})")
        if len(self._units) == 1:
            return f"({row_strs[0]},)"
        return f"({', '.join(row_strs)})"

    # ── Python data model ─────────────────────────────────────────────

    def __repr__(self) -> str:
        return f'UnitsMatrix("{self.to_string()}")'

    def __eq__(self, other: Any, /) -> bool:
        if isinstance(other, UnitsMatrix):
            if self._units.shape != other._units.shape:
                return False
            return bool(np.all(self._units == other._units))
        if isinstance(other, (tuple, list)):
            try:
                return self == UnitsMatrix(other)
            except (TypeError, ValueError):
                return False
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.to_tuple())

    def __iter__(self) -> Any:
        """Iterate over elements (1D) or row ``UnitsMatrix`` objects (2D).

        Examples
        --------
        >>> from unxts.linalg import UnitsMatrix
        >>> list(UnitsMatrix(("m", "rad", "rad")))
        [Unit("m"), Unit("rad"), Unit("rad")]

        """
        if self._units.ndim == 1:
            yield from self._units
            return
        for row in self._units:
            yield UnitsMatrix(row)

    def __getitem__(self, index: Any, /) -> Any:
        """Index into the UnitsMatrix to retrieve a unit or sub-structure.

        >>> from unxts.linalg import UnitsMatrix
        >>> units = UnitsMatrix((("m", "s"), ("kg", "rad")))

        Indexing a single element returns a unit:

        >>> units[0, 1]
        Unit("s")

        Indexing a row returns a UnitsMatrix:

        >>> units[0]
        UnitsMatrix("(m, s)")

        """
        result = self._units[index]
        if isinstance(result, np.ndarray):
            if result.ndim == 0:  # 0-d array -> extract the contained unit.
                return result.item()
            return UnitsMatrix(result)
        return result


@plum.dispatch
def unit(tuple_of_units: tuple[Any, ...], /) -> UnitsMatrix:
    """Convert a nested tuple of units into a ``UnitsMatrix``.

    This allows users to specify units in a convenient nested tuple format when
    constructing ``QuantityMatrix`` instances, and have them automatically
    converted to the appropriate ``UnitsMatrix``.

    >>> import unxt as u

    1D case:

    >>> u.unit(("m", "s", "kg"))
    UnitsMatrix("(m, s, kg)")

    2D case:

    >>> u.unit((("m", "s"), ("kg", "rad")))
    UnitsMatrix("((m, s), (kg, rad))")

    """
    return UnitsMatrix(tuple_of_units)


@plum.dispatch
def unit(arr: np.ndarray, /) -> UnitsMatrix:
    """Convert a numpy object array of units into a ``UnitsMatrix``.

    >>> import numpy as np
    >>> import unxt as u
    >>> from unxts.linalg import UnitsMatrix
    >>> arr = np.array([u.unit("m"), u.unit("s")], dtype=object)
    >>> u.unit(arr)
    UnitsMatrix("(m, s)")

    """
    return UnitsMatrix(arr)


@plum.dispatch
def unit(obj: UnitsMatrix, /) -> UnitsMatrix:
    """Identity: a UnitsMatrix is returned unchanged by the unit converter."""
    return obj


@plum.dispatch
def unit_of(obj: UnitsMatrix, /) -> UnitsMatrix:
    """Identity conversion for UnitsMatrix to itself.

    >>> import unxt as u
    >>> unit = u.unit(("m", "s", "kg"))
    >>> u.unit_of(unit) is unit
    True

    """
    return obj
