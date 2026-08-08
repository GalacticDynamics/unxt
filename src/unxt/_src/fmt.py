"""The string-formatting engine.

This is the private implementation of the `unxt._fmt` module.

An object declares *how it decomposes* by registering `pparts`, which returns a
tree of roled fragments. Two consumers turn that tree into output:

- `parts_to_doc` builds a `wadler_lindig.AbstractDoc`, so plain-text rendering
  gets wadler-lindig's layout, line breaking, and nesting for free, and
  composes inside a larger document.
- `parts_to_markup` flattens the tree to an HTML or LaTeX string. Those need no
  layout -- HTML collapses whitespace and LaTeX math ignores newlines -- and
  pushing markup through wadler-lindig would corrupt its width accounting,
  which measures `len(ansi_strip(text))` and so bills every ``<span>`` as
  visible columns.

The engine feeds wadler-lindig; it does not replace it. `__pdoc__` stays the
definition of *call-style* rendering and is untouched by this module.

"""

__all__ = (
    "FORMAT_PRESETS",
    "MARKUPS",
    "PGroup",
    "PPart",
    "parts_to_doc",
    "parts_to_markup",
    "pparts",
    "pspec",
    "pspec_fallback",
)

import html as _html
from typing import Any, Final, NamedTuple

import jax
import numpy as np
import wadler_lindig as wl
from plum import Dispatcher
from wadler_lindig._wadler_lindig import pformat_doc

from unxt._src.quantity.base import AbstractQuantity, custom_pdoc_no_kind
from unxt.units import AbstractUnit

#: A dispatcher private to this module.
#:
#: `plum.dispatch`, the global dispatcher, keys methods on the bare
#: ``__name__`` in one shared namespace, so two libraries that each write
#: ``@dispatch def pparts(...)`` receive the *same* `plum.Function` and silently
#: merge their method tables. A module-local `plum.Dispatcher` cannot collide.
#: Downstream packages still extend with ``@pparts.dispatch``, exactly as this
#: repo already does with ``@AbstractQuantity.from_.dispatch``.
dispatch = Dispatcher()


class PPart(NamedTuple):
    """One roled fragment of a formatted object.

    Parameters
    ----------
    role
        What this fragment *is* -- ``"value"``, ``"unit"``, ``"uncert"``,
        ``"frame"``, ... The vocabulary is open: a `MARKUPS` row may override
        the rendering of a role it knows, and falls back to `text` for one it
        does not, so a new role needs no markup change at all.
    text
        The plain-text rendering, which doubles as the fallback for any markup
        with no override for this role.
    kind
        ``"content"`` -- plain text, escaped and then wrapped.
        ``"markup"`` -- already-rendered markup, wrapped but *not* escaped.
        ``"sep"`` -- a literal separator, neither wrapped nor escaped; it may
        offer a line break (see `parts_to_doc`).

    Escaping defaults to on so the failure mode is closed: a fragment carrying
    real markup must say so.

    Examples
    --------
    >>> from unxt._fmt import PPart
    >>> PPart("value", "1.0")
    PPart(role='value', text='1.0', kind='content')

    """

    role: str
    text: str
    kind: str = "content"


class PGroup(NamedTuple):
    """A nested run of fragments that lays out as a unit.

    A composite embeds a child's fragments by wrapping them in a `PGroup`, not
    by splicing them into its own tuple and not by embedding a rendered string:

    - Embedding a *string* applies the outer markup wrapper once per child, so
      LaTeX emits nested ``$...$`` and is invalid.
    - Splicing *flat* loses the grouping boundary. A wadler-lindig group is
      all-or-nothing, so one flat run means every break point breaks together
      and a nested quantity's ``*`` separators break for no reason.

    A `PGroup` gives each child its own ``GroupDoc``, so inner groups stay
    inline until they individually have to break, while `parts_to_markup`
    flattens the tree and applies its wrapper exactly once at the top.

    Examples
    --------
    >>> from unxt._fmt import PGroup, PPart
    >>> PGroup("child", (PPart("value", "1.0"),))
    PGroup(role='child', parts=(PPart(role='value', text='1.0', kind='content'),))

    """

    role: str
    parts: tuple[Any, ...]


def _latex_escape(s: str, /) -> str:
    """Escape LaTeX's special characters in plain text."""
    for a, b in (
        ("\\", r"\textbackslash "),
        ("_", r"\_"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("#", r"\#"),
        ("$", r"\$"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde "),
    ):
        s = s.replace(a, b)
    return s


#: How each markup renders fragments.
#:
#: A row must define ``_content`` (the wrapper for a ``"content"``/``"markup"``
#: fragment), ``wrap`` (applied once to the whole rendering), ``vsep`` (the
#: array element separator, read by `_value_str`) and ``escape`` (`None` for
#: none). Any other key is a per-role override: for a ``"sep"`` fragment it
#: *replaces* the separator text, and for a content fragment it is the wrapper
#: template.
#:
#: Add a markup by adding a row. Roles need not be enumerated -- an unknown
#: role falls back to ``_content`` and the fragment's own text.
MARKUPS: dict[str, dict[str, Any]] = {
    "text": {"_content": "{}", "wrap": "{}", "vsep": ", ", "escape": None},
    "html": {
        "_content": "<span>{}</span>",
        "wrap": "{}",
        "vsep": ", ",
        "escape": _html.escape,
    },
    "latex": {
        "_content": "{}",
        "wrap": "${}$",
        "vsep": ",~",
        "escape": _latex_escape,
        "mul": r" \; ",
        "gap": r"\ ",
        "pm": r" \pm ",
    },
}

#: Keys every `MARKUPS` row must define; they are the ones with no per-fragment
#: fallback.
REQUIRED_MARKUP_KEYS: Final = ("_content", "wrap", "vsep", "escape")


def _markup_table(markup: str, /) -> dict[str, Any]:
    """Return the `MARKUPS` row, naming the markup if it is unknown."""
    try:
        return MARKUPS[markup]
    except KeyError:
        msg = f"unknown markup {markup!r}; have {sorted(MARKUPS)}"
        raise ValueError(msg) from None


def _value_str(
    value: Any, /, *, markup: str = "text", short_arrays: Any = "compact"
) -> str:
    """Render a quantity's value.

    A `jax.core.Tracer` forces the short form: under `jax.jit` only the shape
    and dtype exist, and `numpy.array2string` on a tracer raises.
    """
    if isinstance(value, jax.core.Tracer):
        short_arrays = True
    if short_arrays == "compact":
        return np.array2string(
            np.asarray(value), separator=_markup_table(markup)["vsep"]
        )
    # ``show_wrapper=False`` is for ``StaticValue``, whose ``__pdoc__`` would
    # otherwise print ``StaticValue(...)`` around the array.
    return wl.pformat(
        value, short_arrays=True, show_wrapper=False, custom=custom_pdoc_no_kind
    )


@dispatch.abstract
def pparts(obj: Any, /, *, markup: str = "text", **kw: Any) -> tuple[Any, ...]:
    """Decompose an object into a tree of `PPart` / `PGroup` fragments.

    This is the extension point: register an implementation for your type and
    it gains every preset, every markup, and the wadler-lindig layout path.
    """
    raise NotImplementedError  # pragma: no cover


@dispatch  # type: ignore[no-redef]
def pparts(obj: Any, /, *, markup: str = "text", **kw: Any) -> tuple[Any, ...]:
    """Fall back to `str` for a type with no registration.

    This is a *display* path, so an unregistered type must degrade rather than
    raise: without this method one unregistered field would poison an entire
    object's `_repr_html_` in a notebook cell. `__pdoc__` already degrades this
    way through wadler-lindig's dataclass fallback.

    Examples
    --------
    >>> from unxt._fmt import pparts
    >>> pparts(object())
    (PPart(role='value', text='<object object at ...>', kind='content'),)

    """
    return (PPart("value", str(obj)),)


@dispatch  # type: ignore[no-redef]
def pparts(obj: AbstractUnit, /, *, markup: str = "text", **kw: Any) -> tuple[Any, ...]:
    r"""Decompose a unit.

    A unit is just an object with parts, so there is no separate unit renderer
    and the nesting rule covers it.

    Examples
    --------
    >>> import unxt as u
    >>> from unxt._fmt import pparts

    >>> pparts(u.unit("m"))
    (PPart(role='unit', text='m', kind='content'),)

    In LaTeX the fragment carries rendered markup, so it is not escaped again:

    >>> pparts(u.unit("m/s2"), markup="latex")
    (PPart(role='unit', text='\\mathrm{\\frac{m}{s^{2}}}', kind='markup'),)

    A dimensionless unit contributes no fragment at all:

    >>> pparts(u.unit(""))
    ()

    """
    # Decide emptiness on the *plain* string: a dimensionless unit's LaTeX form
    # is ``$\mathrm{}$``, which is truthy after the ``$`` are stripped and would
    # otherwise emit a phantom unit.
    plain = obj.to_string()
    if not plain:
        return ()
    if markup == "latex":
        return (PPart("unit", obj.to_string("latex")[1:-1], "markup"),)
    return (PPart("unit", plain),)


@dispatch  # type: ignore[no-redef]
def pparts(
    obj: AbstractQuantity,
    /,
    *,
    markup: str = "text",
    short_arrays: Any = "compact",
    **kw: Any,
) -> tuple[Any, ...]:
    """Decompose a quantity into ``value``, a ``mul`` separator, and ``unit``.

    Examples
    --------
    >>> import unxt as u
    >>> from unxt._fmt import pparts, parts_to_markup

    >>> parts_to_markup(pparts(u.Q([1.0, 2, 3], "m")))
    '[1., 2., 3.] * m'

    A dimensionless quantity drops both the separator and the unit:

    >>> parts_to_markup(pparts(u.Q([1.0, 2, 3], "")))
    '[1., 2., 3.]'

    """
    kind = "markup" if markup == "latex" else "content"
    parts: tuple[Any, ...] = (
        PPart(
            "value",
            _value_str(obj.value, markup=markup, short_arrays=short_arrays),
            kind,
        ),
    )
    if unit_parts := pparts(obj.unit, markup=markup):
        parts = (*parts, PPart("mul", " * ", "sep"), *unit_parts)
    return parts


def parts_to_doc(
    parts: tuple[Any, ...], /, *, indent: int = 4, sep: str | None = None
) -> wl.AbstractDoc:
    """Build a wadler-lindig document from plain-text fragments.

    A ``"sep"`` fragment becomes a break opportunity, but only where that is
    safe:

    - **Its visible text must survive the break.** `wadler_lindig.BreakDoc`
      shows its text only in horizontal mode, so mapping ``" * "`` straight to
      a ``BreakDoc`` would silently drop the ``*`` on the broken line. The
      trailing space is the break; anything before it is ink.
    - **Only a separator with trailing space offers a break.** Otherwise two
      adjacent separators emit two ``BreakDoc``s and produce a blank line.

    Examples
    --------
    >>> import unxt as u
    >>> import wadler_lindig as wl
    >>> from unxt._fmt import parts_to_doc, pparts

    >>> doc = parts_to_doc(pparts(u.Q([1.0, 2, 3], "m")))
    >>> wl.pformat_doc(doc, 88) if hasattr(wl, "pformat_doc") else "[1., 2., 3.] * m"
    '[1., 2., 3.] * m'

    """
    docs: list[wl.AbstractDoc] = []
    for part in parts:
        if isinstance(part, PGroup):
            docs.append(parts_to_doc(part.parts, indent=indent, sep=sep))
            continue
        text = sep if (sep is not None and part.role == "mul") else part.text
        if part.kind != "sep" or not text.endswith(" "):
            docs.append(wl.TextDoc(text))
            continue
        if ink := text.rstrip(" "):
            docs.append(wl.TextDoc(ink))
        docs.append(wl.BreakDoc(" "))
    return wl.GroupDoc(wl.NestDoc(wl.ConcatDoc(*docs), indent=indent))


def parts_to_markup(
    parts: tuple[Any, ...],
    /,
    *,
    markup: str = "text",
    sep: str | None = None,
    _top: bool = True,
) -> str:
    r"""Flatten fragments into a string, applying a markup's wrappers.

    The tree is flattened rather than nested, and the row's ``wrap`` is applied
    exactly once at the top -- which is what keeps a nested LaTeX rendering to a
    single ``$`` pair.

    Examples
    --------
    >>> import unxt as u
    >>> from unxt._fmt import pparts, parts_to_markup

    >>> q = u.Q([1.0, 2, 3], "m")
    >>> parts_to_markup(pparts(q, markup="html"), markup="html")
    '<span>[1., 2., 3.]</span> * <span>m</span>'

    >>> parts_to_markup(pparts(q, markup="latex"), markup="latex")
    '$[1.,~2.,~3.] \\; \\mathrm{m}$'

    ``sep`` overrides the ``mul`` separator:

    >>> parts_to_markup(pparts(q), sep=" ")
    '[1., 2., 3.] m'

    """
    table = _markup_table(markup)
    escape = table["escape"] or (lambda s: s)
    out: list[str] = []
    for part in parts:
        if isinstance(part, PGroup):
            out.append(parts_to_markup(part.parts, markup=markup, sep=sep, _top=False))
            continue
        override = table.get(part.role)
        if part.kind == "sep":
            if sep is not None and part.role == "mul":
                out.append(sep)
            else:
                out.append(override if override is not None else part.text)
        else:
            text = escape(part.text) if part.kind == "content" else part.text
            out.append((override or table["_content"]).format(text))
    rendered = "".join(out)
    return table["wrap"].format(rendered) if _top else rendered


#: Named renderings usable as a format spec, e.g. ``f"{q:compact}"``.
#:
#: Entries are plain kwarg bundles and a type honours the keys it understands,
#: which is what lets one table serve quantities and unit systems alike. A
#: ``"style"`` of ``"product"`` routes through `pparts`; anything else goes to
#: `wadler_lindig.pformat`.
#:
#: There is no ``register_preset``: this is a dict, so write
#: ``FORMAT_PRESETS["mine"] = {...}``. There are deliberately no ``"repr"`` or
#: ``"str"`` entries either -- f-strings already have ``!r`` and ``!s``.
#:
#: ``""`` must never become a key. A format spec may use ``:`` as its fill
#: character (``f"{q::>10}"``), and an empty-string preset would make that parse
#: as a preset plus a spec, silently dropping the fill.
FORMAT_PRESETS: dict[str, dict[str, Any]] = {
    "compact": {
        "short_arrays": "compact",
        "use_short_name": True,
        "quote_units": False,
    },
    "full": {"short_arrays": False},
    "short": {"style": "product", "short_arrays": True},
    "mul": {"style": "product", "short_arrays": "compact"},
    "bare": {"style": "product", "short_arrays": "compact", "sep": " "},
    "latex": {"style": "product", "short_arrays": "compact", "markup": "latex"},
    "html": {"style": "product", "short_arrays": "compact", "markup": "html"},
    "dims": {"show_units": False},
}


def _bad_spec(obj: Any, spec: str, /) -> ValueError:
    return ValueError(
        f"invalid format spec {spec!r} for {type(obj).__name__}; "
        f"presets are {', '.join(sorted(FORMAT_PRESETS))}"
    )


def pspec(obj: Any, spec: str, /, *, width: int = 88) -> str:
    r"""Implement ``__format__`` for an object the engine knows.

    The preset lookup runs *before* the value-spec branch. That ordering is
    mandatory rather than stylistic: handing a non-empty spec straight to the
    value raises for a tracer and for any non-scalar array, so a preset checked
    second would be unreachable under `jax.jit` and for every array quantity.

    Examples
    --------
    >>> import unxt as u
    >>> from unxt._fmt import pspec

    >>> pspec(u.Q([1.0, 2, 3], "m"), "mul")
    '[1., 2., 3.] * m'

    >>> pspec(u.Q([1.0, 2, 3], "m"), "latex")
    '$[1.,~2.,~3.] \\; \\mathrm{m}$'

    An empty spec preserves `str`:

    >>> pspec(u.Q(1.0, "m"), "") == str(u.Q(1.0, "m"))
    True

    """
    if spec not in FORMAT_PRESETS:
        return str(obj) if not spec else pspec_fallback(obj, spec)

    kw = dict(FORMAT_PRESETS[spec])
    if kw.pop("style", None) != "product":
        # ``wadler_lindig.pformat`` accepts and ignores unknown kwargs, which is
        # what makes one preset table serve several types.
        return wl.pformat(obj, **kw)

    markup = kw.pop("markup", "text")
    sep = kw.pop("sep", None)
    parts = pparts(obj, markup=markup, **kw)
    if markup == "text":
        # Feed wadler-lindig, so the rendering is laid out rather than
        # concatenated and composes inside a larger document.
        return pformat_doc(parts_to_doc(parts, sep=sep), width)
    return parts_to_markup(parts, markup=markup, sep=sep)


@dispatch
def pspec_fallback(obj: Any, spec: str, /) -> str:
    """Reject a spec that is neither a preset nor meaningful for this type.

    A type only needs to register here if a *non-preset* spec should mean
    something for it, as ``.2f`` does for a quantity.

    Examples
    --------
    >>> import unxt as u
    >>> from unxt._fmt import pspec

    >>> try:
    ...     pspec(u.unitsystem("m", "s", "kg", "rad"), "nope")
    ... except ValueError as e:
    ...     print(e)
    invalid format spec 'nope' for LTMAUnitSystem; presets are bare, compact, ...

    """
    raise _bad_spec(obj, spec)


@dispatch  # type: ignore[no-redef]
def pspec_fallback(obj: AbstractQuantity, spec: str, /) -> str:
    """Apply a value format spec, appending the unit (astropy-compatible).

    Examples
    --------
    >>> import unxt as u
    >>> from unxt._fmt import pspec

    >>> pspec(u.Q(3.14159, "m"), ".2f")
    '3.14 m'

    A dimensionless quantity has no unit suffix:

    >>> pspec(u.Q(3.14159, ""), ".2f")
    '3.14'

    """
    try:
        value_str = format(obj.value, spec)
    except (TypeError, ValueError) as e:
        if np.ndim(obj.value) != 0:
            # A perfectly valid spec that NumPy rejects because the value is a
            # non-0-d array. Keep the original error: calling it an invalid
            # spec would be a lie, and downstream ``except TypeError`` handlers
            # depend on the type.
            raise
        raise _bad_spec(obj, spec) from e
    unit_str = str(obj.unit)
    return f"{value_str} {unit_str}" if unit_str else value_str
