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
    "custom_pdoc_no_kind",
    "custom_pdoc_noarray",
    "value_str",
)

from typing import Any

import jax
import numpy as np
import wadler_lindig as wl

from .engine import (
    Axis,
    _markup_table,
    register_alias,
    register_axis,
)
from .generic import VALUE_FROM_SHORT_ARRAYS  # noqa: F401


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


def value_str(
    value: Any,
    /,
    *,
    markup: str = "text",
    short_arrays: Any = "compact",
    value_spec: str | None = None,
) -> str:
    """Render a quantity's value, in one of the three ``value``-axis forms.

    ``short_arrays`` is the axis in `__pdoc__`'s spelling: ``"compact"`` for
    the values (``[1., 2.]``), `True` for a shape/dtype summary (``f32[2]``),
    `False` for the full array repr (``Array([1., 2.], dtype=float32)``).

    A `jax.core.Tracer` forces the summary: under `jax.jit` only the shape and
    dtype exist, and `numpy.array2string` on a tracer raises -- so
    ``value_spec`` goes unused there too, like any other per-element detail a
    summary cannot show.

    ``value_spec``, when given, is a Python format spec (e.g. ``".3g"``)
    applied to every element via `numpy.array2string`'s ``formatter``, instead
    of NumPy's own default float rendering.

    The text comes back in the markup's own dialect -- already escaped for HTML
    or LaTeX, with LaTeX element separators in place -- so the caller marks it
    ``kind="markup"`` rather than having it escaped a second time.
    """
    table = _markup_table(markup)
    escape = table["escape"] or (lambda s: s)
    if isinstance(value, jax.core.Tracer):
        short_arrays = True
    if short_arrays == "compact":
        formatter = {"all": lambda v: format(v, value_spec)} if value_spec else None
        # A placeholder separator, so escaping the elements cannot touch the
        # markup's own ``vsep`` (LaTeX's ``,~`` would be escaped to nothing).
        if table["escape"] is None:
            return np.array2string(
                np.asarray(value), separator=table["vsep"], formatter=formatter
            )
        s = np.array2string(np.asarray(value), separator="\0", formatter=formatter)
        return escape(s).replace("\0", table["vsep"])
    # ``show_wrapper=False`` is for ``StaticValue``, whose ``__pdoc__`` would
    # otherwise print ``StaticValue(...)`` around the array.
    #
    # The hook *builds* a summary, so it belongs only on the `True` path;
    # ``custom=None`` is called and raises, so it is omitted, not blanked.
    kw = {"custom": custom_pdoc_no_kind} if short_arrays else {}
    return escape(
        wl.pformat(value, short_arrays=short_arrays, show_wrapper=False, **kw)
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
