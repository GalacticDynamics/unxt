"""Conversion to quantity value types."""

__all__ = ("StaticValue", "convert_to_quantity_value")

import operator
import warnings
from typing import Any, final, override

import jax
import numpy as np
import plum
import quax
import wadler_lindig as wl
from jaxtyping import Array, ArrayLike

import quaxed.numpy as jnp

from unxt._src import fmt

#: Raised when a traced value is offered to a static field. Shared with
#: `unxt.quantity.StaticQuantity._mk`, which repeats this type switch inline to
#: keep `plum` off its hot path, so the two must not drift.
TRACED_VALUE_MSG = (
    "StaticQuantity cannot hold a traced JAX value; use Quantity under jit/vmap/grad."
)


@final
class StaticValue:
    """Immutable static value wrapper for `StaticQuantity`.

    This stores a read-only NumPy array and is used to keep the value static
    while avoiding Equinox's static-array warnings. Arithmetic operations
    degrade to the wrapped array unless both operands are `StaticValue`, in
    which case a `StaticValue` is returned.

    Note that since the array is immutable, hashing is supported.
    The hash is computed from the array's dtype, shape, and bytes.

    .. rubric:: Equality semantics

    `StaticValue` equality returns a **scalar** `bool`, not an element-wise
    array.  This is intentional: it mirrors Python's structural-equality
    contract (like ``tuple.__eq__``) and is a prerequisite for using a
    ``Quantity(StaticValue, ...)`` as a ``static_argnames`` argument in
    :func:`jax.jit`.  JAX needs a scalar ``bool`` to decide whether a cached
    compilation can be reused.

    This is fundamentally different from a normal :class:`~unxt.quantity.Quantity`
    (backed by a JAX array), whose ``==`` operator follows NumPy broadcasting
    rules and returns an element-wise boolean array.

    Examples
    --------
    >>> import numpy as np
    >>> from unxt.quantity import StaticValue

    Equality between two `StaticValue` objects returns a plain `bool`:

    >>> StaticValue(np.array([1.0, 2.0])) == StaticValue(np.array([1.0, 2.0]))
    True
    >>> StaticValue(np.array([1.0, 2.0])) == StaticValue(np.array([9.0, 9.0]))
    False

    Equality with a NumPy array returns an element-wise boolean array
    (same as NumPy):

    >>> StaticValue(np.array([1.0, 2.0])) == np.array([1.0, 9.0])
    array([ True, False])

    """

    __slots__ = ("_array",)

    def __init__(self, array: object, /) -> None:
        value = np.asarray(array)
        # Copy only when np.asarray returned a view of the caller's buffer so
        # that (a) we don't make the caller's array read-only as a side-effect
        # and (b) the caller cannot later mutate our internal data.
        if isinstance(array, np.ndarray) and np.shares_memory(value, array):
            value = value.copy()
        value.setflags(write=False)
        self._array = value

    @property
    def array(self) -> np.ndarray:
        """Return the contained NumPy array."""
        return self._array

    @property
    def _jnparray(self) -> Array:
        """Return the contained array as a JAX array.

        JAX 0.7.2+ introduces TypedNdArray for type-preserving operations.
        These are not jax.Array instances but carry dtype information.
        We need to explicitly convert them to proper arrays while preserving dtype.
        """
        out = jnp.asarray(self._array)
        if not isinstance(out, jax.Array):  # pragma: no cover -- jax-version guard
            out = jnp.asarray(out, dtype=out.dtype)
        return out

    # Constructor
    @classmethod
    @plum.dispatch.abstract
    def from_(cls: type["StaticValue"], *args: Any, **kwargs: Any) -> "StaticValue":
        """Create a `StaticValue` from given arguments."""
        raise NotImplementedError  # pragma: no cover

    # ==================================================
    # NumPy API

    def __array__(self, dtype: Any = None) -> np.ndarray:
        if dtype is None:
            return np.asarray(self._array)
        return np.asarray(self._array, dtype=dtype)

    # ==================================================
    # Wadler-Lindig API

    def __pdoc__(self, *, show_wrapper: bool = True, **kwargs: Any) -> wl.AbstractDoc:
        """Return the Wadler-Lindig representation of this class."""
        if not show_wrapper:
            return wl.pdoc(self._array, **kwargs)
        return (
            wl.TextDoc("StaticValue(")
            + wl.pdoc(self._array, **kwargs)
            + wl.TextDoc(")")
        )

    # ==================================================
    # Python container API

    def __len__(self) -> int:
        return len(self._array)

    def __iter__(self) -> Any:
        return iter(self._array)

    def __getitem__(self, key: Any) -> Any:
        return self._array[key]

    def __getattr__(self, name: str) -> Any:
        # Only forward *public* attributes to the wrapped array. Dunder and
        # private names must raise ``AttributeError`` so that (a) the copy /
        # pickle protocol (``__deepcopy__``, ``__setstate__``, ...) falls back
        # to the correct slot-based default instead of being answered by the
        # wrapped array, and (b) a lookup during reconstruction -- when the
        # ``_array`` slot is not yet set -- cannot recurse through this method.
        if name.startswith("_"):
            msg = f"{type(self).__name__!r} object has no attribute {name!r}"
            raise AttributeError(msg)
        # Forward to the wrapped *NumPy* array, not ``jnp.asarray(self._array)``:
        # the latter applies JAX's x64-disabled dtype rules and silently
        # downcasts a faithfully-stored int64/float64 value (the whole point of
        # StaticValue is to keep such values verbatim). NumPy implements
        # ``dtype``/``min``/``max``/``mean``/``round``/``astype``/``flatten`` with
        # the correct dtype semantics; the ``_jnparray`` property remains the
        # JAX-conversion path for the arithmetic primitives.
        return getattr(self._array, name)

    def __repr__(self) -> str:
        return wl.pformat(self, short_arrays=False, max_width=80)

    def __format__(self, format_spec: str, /) -> str:
        """Format the wrapped value, delegating to the NumPy array.

        This makes ``format(sv, spec)`` behave like formatting the underlying
        scalar; a non-scalar value raises ``TypeError`` as NumPy does.

        Formatting a `unxt.quantity.StaticQuantity` no longer arrives here.
        A quantity's format spec is applied per *element* by the formatting
        engine, so it works on an array too -- which is why this method is now
        only reached by formatting a `StaticValue` directly.

        Examples
        --------
        >>> import numpy as np
        >>> from unxt._src.quantity.value import StaticValue

        >>> format(StaticValue(np.asarray(3.14159)), ".2f")
        '3.14'

        """
        return format(self._array, format_spec)

    @override
    def __eq__(self, other: object, /) -> bool | np.ndarray | Array:  # type: ignore[override]
        if isinstance(other, StaticValue):
            return np.array_equal(self._array, other._array)
        if isinstance(other, Array):
            return self._array == other
        if isinstance(other, (np.ndarray, list, tuple)):
            return np.equal(self._array, other)
        return NotImplemented

    def __ne__(self, other: object, /) -> bool | np.ndarray | Array:  # type: ignore[override]
        if isinstance(other, StaticValue):
            return not bool(np.array_equal(self._array, other._array))
        if isinstance(other, Array):
            return self._array != other
        if isinstance(other, (np.ndarray, list, tuple)):
            return np.not_equal(self._array, other)
        return NotImplemented

    def __lt__(self, other: Any, /) -> np.ndarray:
        if isinstance(other, StaticValue):
            other = other._array
        return self._array < other

    def __le__(self, other: Any, /) -> np.ndarray:
        if isinstance(other, StaticValue):
            other = other._array
        return self._array <= other

    def __gt__(self, other: Any, /) -> np.ndarray:
        if isinstance(other, StaticValue):
            other = other._array
        return self._array > other

    def __ge__(self, other: Any, /) -> np.ndarray:
        if isinstance(other, StaticValue):
            other = other._array
        return self._array >= other

    def __hash__(self) -> int:
        return hash((self._array.dtype.str, self._array.shape, self._array.tobytes()))

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """Reconstruct by re-running ``__init__`` on the wrapped array.

        Rebuilding through the constructor -- rather than restoring the slot
        directly -- is what keeps the read-only invariant across *every* pickle
        protocol, because ``__init__`` is the one place that calls
        ``setflags(write=False)``.

        Examples
        --------
        >>> import pickle
        >>> import numpy as np
        >>> from unxt.quantity import StaticValue

        >>> sv = StaticValue(np.array([1.0, 2.0]))
        >>> restored = pickle.loads(pickle.dumps(sv, protocol=0))
        >>> restored == sv
        True

        The restored value is still immutable, and so still hashes equal:

        >>> restored.array.flags.writeable
        False
        >>> hash(restored) == hash(sv)
        True

        """
        return (type(self), (self._array,))

    def _binary_op(self, other: Any, op: Any) -> Any:
        if isinstance(other, StaticValue):
            result = op(self._jnparray, jnp.asarray(other.array))
            return StaticValue(np.asarray(result))
        return op(self._jnparray, other)

    def _rbinary_op(self, other: Any, op: Any) -> Any:
        if isinstance(other, StaticValue):
            result = op(jnp.asarray(other.array), self._jnparray)
            return StaticValue(np.asarray(result))
        return op(other, self._jnparray)

    def __add__(self, other: Any) -> Any:
        return self._binary_op(other, operator.add)

    def __radd__(self, other: Any) -> Any:
        return self._rbinary_op(other, operator.add)

    def __sub__(self, other: Any) -> Any:
        return self._binary_op(other, operator.sub)

    def __rsub__(self, other: Any) -> Any:
        return self._rbinary_op(other, operator.sub)

    def __mul__(self, other: Any) -> Any:
        return self._binary_op(other, operator.mul)

    def __rmul__(self, other: Any) -> Any:
        return self._rbinary_op(other, operator.mul)

    def __truediv__(self, other: Any) -> Any:
        return self._binary_op(other, operator.truediv)

    def __rtruediv__(self, other: Any) -> Any:
        return self._rbinary_op(other, operator.truediv)

    def __floordiv__(self, other: Any) -> Any:
        return self._binary_op(other, operator.floordiv)

    def __rfloordiv__(self, other: Any) -> Any:
        return self._rbinary_op(other, operator.floordiv)

    def __pow__(self, other: Any) -> Any:
        return self._binary_op(other, operator.pow)

    def __rpow__(self, other: Any) -> Any:
        return self._rbinary_op(other, operator.pow)

    def __mod__(self, other: Any) -> Any:
        return self._binary_op(other, operator.mod)

    def __rmod__(self, other: Any) -> Any:
        return self._rbinary_op(other, operator.mod)

    def __matmul__(self, other: Any) -> Any:
        return self._binary_op(other, operator.matmul)

    def __rmatmul__(self, other: Any) -> Any:
        return self._rbinary_op(other, operator.matmul)

    def __neg__(self) -> Any:
        return -self._jnparray

    def __pos__(self) -> Any:
        return +self._jnparray

    def __abs__(self) -> Any:
        return abs(self._jnparray)


@StaticValue.from_.dispatch
def from_(cls: type[StaticValue], value: object, /) -> StaticValue:
    """Convert a value for `StaticQuantity`."""
    return cls(np.asarray(value))


@StaticValue.from_.dispatch
def from_(cls: type[StaticValue], value: StaticValue, /) -> StaticValue:
    """Convert a value for `StaticQuantity`."""
    return value


@StaticValue.from_.dispatch
def from_(cls: type[StaticValue], value: jax.Array | jax.core.Tracer, /) -> StaticValue:
    """Materialise a concrete JAX array to NumPy, or reject a traced value.

    A concrete (eager) ``jax.Array`` is just data, so it can back a static
    value -- materialise it. This also lets ops on a ``StaticQuantity`` stay
    static: the primitive rules rebuild via ``replace(x, value=<jax array>)``,
    and that concrete array round-trips back to NumPy here instead of raising.

    A *tracer* (under ``jit``/``vmap``/``grad``) is a placeholder for a value
    that does not exist yet, so it genuinely cannot be static and is rejected.
    """
    if isinstance(value, jax.core.Tracer):
        raise TypeError(TRACED_VALUE_MSG)
    return cls(np.asarray(value))


# ==================================================================
# Converters for quantity value field


@plum.dispatch.abstract
def convert_to_quantity_value(obj: Any, /) -> Any:
    """Convert for the value field of an `AbstractQuantity` subclass."""
    raise NotImplementedError  # pragma: no cover


@plum.dispatch
def convert_to_quantity_value(obj: StaticValue, /) -> StaticValue:
    """Allow static values in quantity-like classes."""
    return obj


@plum.dispatch
def convert_to_quantity_value(obj: quax.ArrayValue, /) -> Any:
    """Convert a `quax.ArrayValue` for the value field.

    >>> import warnings
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue):
    ...     value: Array
    ...
    ...     def aval(self):
    ...         return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...
    ...     def materialise(self):
    ...         return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> with warnings.catch_warnings(record=True, action="always") as w:
    ...     y = convert_to_quantity_value(x)
    >>> y
    MyArray(value=i32[3])
    >>> print(f"Warning caught: {w[-1].message}")
    Warning caught: 'quax.ArrayValue' subclass 'MyArray' ...

    """
    warnings.warn(
        f"'quax.ArrayValue' subclass {type(obj).__name__!r} does not have a registered "
        "converter. Returning the object as is.",
        category=UserWarning,
        stacklevel=2,
    )
    return obj


# NB: the ``list`` / ``tuple`` members are deliberately *unparametrized*. A
# subscripted generic is not "faithful" in plum, and one unfaithful signature
# disables method caching for the whole function -- which here is the
# ``converter=`` on every quantity's ``value`` field, so it runs on every
# construction. ``list`` and ``list[Any]`` match exactly the same values;
# only the caching differs.
@plum.dispatch
def convert_to_quantity_value(obj: ArrayLike | list | tuple, /) -> Array:
    """Convert an array-like object to a `jax.numpy.ndarray`.

    >>> import jax.numpy as jnp
    >>> from unxt.quantity import convert_to_quantity_value

    >>> convert_to_quantity_value([1, 2, 3])
    Array([1, 2, 3], dtype=int32)

    """
    out = jnp.asarray(obj)

    # JAX 0.7.2+ introduces TypedInt, TypedFloat, TypedComplex for
    # type-preserving operations. These are not jax.Array instances but carry
    # dtype information. Materialise them via ``jax.numpy.asarray`` (not the
    # quaxed one), which converts a weak Python scalar into a *weak* ``jax.Array``
    # -- passing an explicit ``dtype`` would force ``weak_type=False`` and make
    # ``Quantity(1.0, ...)`` over-promote relative to native JAX.
    if not isinstance(out, jax.Array):  # pragma: no cover -- jax-version guard
        out = jax.numpy.asarray(out)

    return out


# ==================================================
# Wadler-Lindig: rendering a value
#
# The value half of a quantity's rendering: generic over `jax.Array` and
# `numpy.ndarray` (the latter is what a `StaticValue` wraps), and knowing
# nothing about units. It lives beside the values it renders; the axes that
# name these choices in a format spec are registered by the modules that own
# them -- `unxt._src.units` for the unit spelling, `unxt._src.fmt` for the
# cross-cutting rest.


def custom_pdoc_no_kind(obj: Any, /) -> wl.AbstractDoc | None:
    """Return the array summary without the ``(jax)``/``(numpy)`` kind suffix.

    Handles `numpy.ndarray` as well as `jax.Array`, so a NumPy-backed value
    renders ``f64[2]`` rather than ``f64[2](numpy)``.
    """
    if isinstance(obj, (jax.Array, np.ndarray)):
        dtype = obj.dtype.name
        if getattr(obj, "weak_type", False):
            dtype = f"weak_{dtype}"
        return wl.array_summary(obj.shape, dtype, kind=None)
    return None


def custom_pdoc_noarray(obj: Any, /) -> wl.AbstractDoc | None:
    """Return the compact (values-only) pdoc for an array-like value.

    Handles both a `jax.Array` and a `numpy.ndarray` -- the latter is what a
    `unxt.quantity.StaticQuantity`'s value wraps -- so its ``str`` shows values
    like a plain quantity's rather than an ``f64[2](numpy)`` type summary.
    """
    if isinstance(obj, (jax.Array, np.ndarray)):
        return wl.TextDoc(np.array2string(np.asarray(obj), separator=", "))
    return None


_SENTINEL = "\0"


def _row_doc(kids: list[wl.AbstractDoc], vsep: str, /) -> wl.AbstractDoc:
    """Lay out one bracketed row of already-parsed children.

    Outer rows use ``bracketed``, which breaks every element or none. The
    innermost row (all leaves) fills as numpy does: each element sits in its
    own group with the delimiter that *follows* it (``,`` or the closing
    ``]``), so that delimiter counts when the group is asked whether it fits.
    """
    delim, brk = vsep[:1], wl.BreakDoc(vsep[1:])
    if not kids or not all(isinstance(k, wl.TextDoc) for k in kids):
        return wl.bracketed(
            begin=wl.TextDoc("["),
            docs=kids,
            sep=wl.TextDoc(delim) + brk,
            end=wl.TextDoc("]"),
            indent=1,
        )
    if len(kids) == 1:
        return wl.ConcatDoc(wl.TextDoc("["), kids[0], wl.TextDoc("]"))
    groups = [
        wl.GroupDoc(brk + k + wl.TextDoc("]" if i == len(kids) - 1 else delim))
        for i, k in enumerate(kids[1:], 1)
    ]
    head = wl.ConcatDoc(kids[0], wl.TextDoc(delim))
    return wl.ConcatDoc(
        wl.TextDoc("["), wl.NestDoc(wl.ConcatDoc(head, *groups), indent=1)
    )


def _array_doc(text: str, *, vsep: str, escape: Any) -> wl.AbstractDoc:
    """Parse numpy's bracketed text (elements joined by ``_SENTINEL``) into a doc.

    numpy fixes the *content* (dtype formatting, padding, summarisation); this
    only recovers the nesting so wadler-lindig owns the *layout*.
    """
    pos = 0

    def node() -> wl.AbstractDoc:
        nonlocal pos
        if text[pos] != "[":
            end = pos
            while end < len(text) and text[end] not in "[]" + _SENTINEL:
                end += 1
            leaf, pos = text[pos:end], end
            return wl.TextDoc(escape(leaf))
        pos += 1
        kids: list[wl.AbstractDoc] = []
        while text[pos] != "]":
            if kids:  # consume the separator and the row break after it
                pos += 1
                if text[pos] == "\n":
                    while text[pos] == "\n":
                        pos += 1
                    while text[pos] == " ":
                        pos += 1
            kids.append(node())
        pos += 1
        return _row_doc(kids, vsep)

    doc = node()
    if pos != len(text):  # trailing text: not the structure we assumed
        raise IndexError(pos)
    return doc


@fmt.pvalue.dispatch  # type: ignore[misc]
def pvalue(
    obj: jax.Array | np.ndarray,
    /,
    *,
    markup: str = "text",
    short_arrays: Any = "compact",
    value_spec: str | None = None,
    **kw: Any,
) -> wl.AbstractDoc:
    """Render an array: its values (``compact``), or a shape/dtype summary.

    A `jax.core.Tracer` forces the summary -- under `jax.jit` only shape and
    dtype exist. ``value_spec`` is applied to every element.
    """
    table = fmt.markup_table(markup)
    escape = table["escape"] or (lambda s: s)
    if isinstance(obj, jax.core.Tracer):
        short_arrays = True
    if short_arrays == "compact":
        formatter = {"all": lambda v: format(v, value_spec)} if value_spec else None
        text = np.array2string(
            np.asarray(obj),
            separator=_SENTINEL,
            formatter=formatter,
            max_line_width=10**9,
        )
        vsep = table["vsep"]  # one delimiter char, then the break text
        # Only numeric text has brackets that are all structure; a string or
        # object element (or a bracket fill in ``value_spec``) can hold ``[``.
        # (ml_dtypes such as bfloat16 report kind ``V``, hence ``issubdtype``.)
        dtype = np.asarray(obj).dtype
        if dtype.kind == "b" or jax.numpy.issubdtype(dtype, jax.numpy.number):
            try:
                return _array_doc(text, vsep=vsep, escape=escape)
            except IndexError:  # not bracket-structured numbers
                pass
        # Never truncate or raise: one flat TextDoc, no break points.
        flat = escape(text).replace(_SENTINEL + "\n", vsep.rstrip() + "\n")
        return wl.TextDoc(flat.replace(_SENTINEL, vsep))
    # ``show_wrapper=False`` is for ``StaticValue``; the summary hook belongs
    # only on the ``True`` path (``custom=None`` would be called and raise).
    custom = {"custom": custom_pdoc_no_kind} if short_arrays else {}
    doc = wl.pdoc(obj, short_arrays=short_arrays, show_wrapper=False, **custom)
    if table["escape"] is None:
        return doc
    return wl.TextDoc(escape(fmt.doc_to_str(doc, fmt.FLAT)))


@fmt.pvalue.dispatch  # type: ignore[misc]
def pvalue(obj: StaticValue, /, **kw: Any) -> Any:
    """Render a static value as the array it wraps."""
    return fmt.pvalue(np.asarray(obj), **kw)
