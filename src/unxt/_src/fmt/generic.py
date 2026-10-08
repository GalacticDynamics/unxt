"""The generic axes (``markup``, ``sep``, ``value``) -- package side.

Like `engine`, this imports only `wadler_lindig`, `plum` and the standard
library: it is part of the future standalone package.
"""

__all__ = (
    "VALUE_FROM_SHORT_ARRAYS",
    "pvalue",
    "register_markup",
    "unregister_markup",
)

from collections.abc import Mapping
from typing import Any, Final

import wadler_lindig as wl

from .engine import (
    _FLAT,
    _KEYWORDS,
    _MARKUPS,
    ALIASES,
    REQUIRED_MARKUP_KEYS,
    Axis,
    _markup_table,
    dispatch,
    doc_to_str,
    register_axis,
)

#: The ``value`` axis as `__pdoc__`'s ``short_arrays`` argument. The public
#: `unxt.config` traits keep their own spelling of the same three-way choice.
_SHORT_ARRAYS: Final[dict[str, Any]] = {
    "array": False,
    "values": "compact",
    "type": True,
}

#: ``short_arrays`` back to the ``value`` axis, for reading `unxt.config`.
#:
#: The config traits are public, documented API and keep their own spelling;
#: this is the one place the two vocabularies are reconciled, so ``repr`` and
#: ``str`` can be defined as specs without renaming anything users configure.
#: Derived by inversion rather than written out, so the two cannot drift.
VALUE_FROM_SHORT_ARRAYS: Final[dict[Any, str]] = {
    v: k for k, v in _SHORT_ARRAYS.items()
}


def _value_product_kwargs(value: Any, /) -> dict[str, Any]:
    """Translate the ``value`` axis for product layout.

    The axis holds *either* one of its keywords or a Python format spec, so
    this is the one place that distinction becomes two arguments: how verbose
    the array is, and how each element is formatted. Free text implies the
    values form -- a shape/dtype summary has no elements to format, which is
    why ``type`` and a format spec cannot both be asked for.
    """
    if value in _SHORT_ARRAYS:
        return {"short_arrays": _SHORT_ARRAYS[value], "value_spec": None}
    return {"short_arrays": "compact", "value_spec": value}


#: Markup spec word -> markup name; `register_markup` extends it in place, and
#: the ``markup`` axis below shares this very dict.
_MARKUP_WORDS: Final[dict[str, str]] = {
    "text": "text",
    "html": "html",
    "latex": "latex",
}

#: Which markup the fragments are wrapped in. Product layout only: a call-style
#: rendering is a constructor expression, which has no markup form.
register_axis(
    Axis(
        name="markup",
        keywords=_MARKUP_WORDS,
        default="text",
        layouts={"product": lambda v: {"markup": v}},
    )
)

#: Whether the join between parts shows its operator. ``mul`` does not
#: override, leaving whatever the object's own `pparts` emitted (``" * "`` for
#: a quantity), so it need not hard-code that string here.
register_axis(
    Axis(
        name="sep",
        keywords={"mul": "mul", "bare": "bare"},
        default="bare",
        layouts={"product": lambda v: {"sep": v}},
    )
)

#: How verbose the numeric payload is, or how to format each element.
#:
#: This is the axis that accepts free text: a spec's trailing run is a Python
#: format spec applied per element. Holding it *on* the axis rather than beside
#: it is what makes ``type-.2f`` an ordinary "value is set twice" error, rather
#: than a hand-written consistency check between two keys describing one thing.
register_axis(
    Axis(
        name="value",
        keywords={"array": "array", "values": "values", "type": "type"},
        default="values",
        layouts={
            "call": lambda v: {"short_arrays": _SHORT_ARRAYS[v]},
            "product": _value_product_kwargs,
        },
        free_text=("product",),
    )
)


def _check_markup_row(row: Mapping[str, Any], /) -> None:
    """Reject a row the renderers could not use, naming what is wrong."""
    if missing := [k for k in REQUIRED_MARKUP_KEYS if k not in row]:
        msg = f"markup row is missing required keys {missing}"
        raise ValueError(msg)
    for key in ("wrap", "_content"):
        if not isinstance(row[key], str):
            msg = f"markup {key!r} must be a str, got {row[key]!r}"
            raise ValueError(msg)  # noqa: TRY004
    # ``wrap`` is consumed by ``split("{}")``, ``_content`` by ``str.format``.
    if row["wrap"].count("{}") != 1:
        msg = f"markup 'wrap' must contain exactly one '{{}}', got {row['wrap']!r}"
        raise ValueError(msg)
    try:
        row["_content"].format("x")
    except (IndexError, KeyError, ValueError):
        msg = f"markup '_content' must format one argument, got {row['_content']!r}"
        raise ValueError(msg) from None


def register_markup(
    name: str, row: Mapping[str, Any], /, *, replace: bool = False
) -> None:
    """Add a markup (a `MARKUPS` row) and make it selectable by spec."""
    _check_markup_row(row)
    if name in _MARKUPS and not replace:
        msg = f"markup {name!r} is already registered"
        raise ValueError(msg)
    if name not in _MARKUP_WORDS and (name in _KEYWORDS or name in ALIASES):
        msg = f"{name!r} is already a keyword or alias"
        raise ValueError(msg)
    _MARKUPS[name] = dict(row)
    if name not in _MARKUP_WORDS:
        _MARKUP_WORDS[name] = name
        _KEYWORDS.setdefault(name, []).append("markup")


def unregister_markup(name: str, /) -> None:
    """Remove a markup added by `register_markup`.

    A test/reload helper: removing a built-in markup (``text``, ``html``,
    ``latex``) breaks the product layouts that name it.
    """
    del _MARKUPS[name]
    del _MARKUP_WORDS[name]
    _KEYWORDS[name].remove("markup")
    if not _KEYWORDS[name]:
        del _KEYWORDS[name]


@dispatch.abstract
def pvalue(
    obj: Any,
    /,
    *,
    markup: str = "text",
    short_arrays: Any = "compact",
    value_spec: str | None = None,
    **kw: Any,
) -> wl.AbstractDoc:
    """Render a *value* as a wadler-lindig document, escaped for ``markup``.

    The extension point for how numbers (or anything else a type holds) are
    shown. Register a method for your value type; the package default below
    renders through ``repr`` / ``format`` / `wadler_lindig.pdoc`.
    """
    raise NotImplementedError  # pragma: no cover


@dispatch  # type: ignore[no-redef]
def pvalue(
    obj: Any,
    /,
    *,
    markup: str = "text",
    short_arrays: Any = "compact",
    value_spec: str | None = None,
    **kw: Any,
) -> wl.AbstractDoc:
    """Fall back to ``repr`` (or ``format`` when a value spec is given)."""
    escape = _markup_table(markup)["escape"]
    if short_arrays == "compact":
        text = format(obj, value_spec) if value_spec else repr(obj)
        return wl.TextDoc(escape(text) if escape else text)
    doc = wl.pdoc(obj, short_arrays=short_arrays, show_wrapper=False, **kw)
    return doc if escape is None else wl.TextDoc(escape(doc_to_str(doc, _FLAT)))
