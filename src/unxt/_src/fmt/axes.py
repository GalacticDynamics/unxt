"""`unxt`'s layer over the string-formatting engine.

Everything here is domain knowledge the engine deliberately does not have: the
axes `unxt` puts into the grammar, the aliases it names them by, and the
array-rendering helpers those axes need. `coordinax` and `galax` add their own
by importing `register_axis` and doing the same thing -- they are peers of this
module, not clients of it.

The import direction is one-way: this module imports the engine, never the
reverse. That is the seam along which the engine lifts out into a package of
its own, and a test enforces it.

"""

__all__ = (
    "VALUE_FROM_SHORT_ARRAYS",
    "custom_pdoc_no_kind",
    "custom_pdoc_noarray",
)

from typing import Any, Final

import jax
import jax.numpy as jnp
import numpy as np
import wadler_lindig as wl

from .engine import (
    _FLAT,
    Axis,
    _markup_table,
    doc_to_str,
    register_alias,
    register_axis,
)
from .generic import _SHORT_ARRAYS, pvalue

#: ``short_arrays`` back to the ``value`` axis, for reading `unxt.config`.
#:
#: The config traits are public, documented API and keep their own spelling;
#: this is the one place the two vocabularies are reconciled, so ``repr`` and
#: ``str`` can be defined as specs without renaming anything users configure.
#: Derived by inversion rather than written out, so the two cannot drift.
VALUE_FROM_SHORT_ARRAYS: Final[dict[Any, str]] = {
    v: k for k, v in _SHORT_ARRAYS.items()
}


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


@pvalue.dispatch  # type: ignore[misc]
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
    table = _markup_table(markup)
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
        if dtype.kind == "b" or jnp.issubdtype(dtype, jnp.number):
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
    return (
        doc if table["escape"] is None else wl.TextDoc(escape(doc_to_str(doc, _FLAT)))
    )


# ============================================================================
# The axes `unxt` puts into the grammar

#: Which spelling of the unit to show. The two layouts want different things
#: from the same choice, which is exactly why an axis translates *per layout*
#: rather than naming one keyword argument.
register_axis(
    Axis(
        name="unit",
        keywords={"symbol": "symbol", "name": "name", "dim": "dim"},
        default="symbol",
        layouts={
            "call": lambda v: {"show_units": v != "dim"},
            "product": lambda v: {"unit_style": v},
        },
    )
)

#: The abbreviated call form -- one idea spelled per type: a short class name
#: for a quantity, unquoted units for a unit system.
register_axis(
    Axis(
        name="abbrev",
        keywords={"abbrev": True},
        default=False,
        layouts={"call": lambda v: {"use_short_name": v, "quote_units": not v}},
    )
)

register_alias("compact", "call-abbrev")
register_alias("full", "call-array")
register_alias("dims", "call-dim")
