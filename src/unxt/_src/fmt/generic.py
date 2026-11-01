"""The generic axes (``markup``, ``sep``, ``value``) -- package side.

Like `engine`, this imports only `wadler_lindig`, `plum` and the standard
library: it is part of the future standalone package.
"""

__all__ = ("VALUE_FROM_SHORT_ARRAYS",)

from typing import Any, Final

from .engine import Axis, register_axis

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


#: Which markup the fragments are wrapped in. Product layout only: a call-style
#: rendering is a constructor expression, which has no markup form.
register_axis(
    Axis(
        name="markup",
        keywords={"text": "text", "html": "html", "latex": "latex"},
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
