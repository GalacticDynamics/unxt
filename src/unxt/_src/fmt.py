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

**This module knows nothing about quantities, units, or unit systems.** It
imports no `unxt` module at all, and every domain-specific rendering is
*registered into* it by its consumers -- `unxt._src.units`,
`unxt._src.quantity.base`, `unxts.linalg`. Dependencies point inward, which is
what lets those consumers import it at module scope instead of working around
a cycle.

"""

__all__ = (
    "FORMAT_PRESETS",
    "MARKUPS",
    "PGroup",
    "PPart",
    "bad_spec",
    "custom_pdoc_no_kind",
    "custom_pdoc_noarray",
    "doc_to_str",
    "parts_to_doc",
    "unwrap_math",
    "parts_to_markup",
    "pparts",
    "pspec",
    "pspec_fallback",
    "value_str",
)

import html as _html
from typing import Any, Final, NamedTuple

import jax
import numpy as np
import wadler_lindig as wl
from plum import Dispatcher

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
#: array element separator, read by `value_str`) and ``escape`` (`None` for
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


def unwrap_math(text: str, /) -> str:
    r"""Strip enclosing ``$...$`` from a LaTeX fragment, if it has them.

    Conditional on the delimiters actually being present, not on the source
    being *expected* to supply them. Slicing unconditionally corrupts any
    fragment that arrives unwrapped -- ``\mathrm{m}`` becomes ``mathrm{m`` --
    which is the defect this repo already fixed once in `_repr_latex_`.

    The length guard is load-bearing: a lone ``"$"`` satisfies both
    ``startswith`` and ``endswith``, and would otherwise be sliced away
    entirely.

    Examples
    --------
    >>> from unxt._fmt import unwrap_math

    >>> unwrap_math(r"$\mathrm{m}$")
    '\\mathrm{m}'

    Already unwrapped, so left alone:

    >>> unwrap_math(r"\mathrm{m}")
    '\\mathrm{m}'

    >>> unwrap_math("$")
    '$'

    """
    if len(text) >= 2 and text.startswith("$") and text.endswith("$"):
        return text[1:-1]
    return text


class _DocHolder(NamedTuple):
    """Presents a ready-made doc to `wadler_lindig` through its own protocol."""

    doc: wl.AbstractDoc

    def __pdoc__(self, **kw: Any) -> wl.AbstractDoc:
        return self.doc


def doc_to_str(doc: wl.AbstractDoc, /, width: int = 88) -> str:
    """Lay out a wadler-lindig document at ``width``.

    wadler-lindig lays out a *document* only through
    ``wadler_lindig._wadler_lindig.pformat_doc``, a private module path that
    may move between releases. Handing `wadler_lindig.pformat` an object whose
    ``__pdoc__`` returns the document reaches the same code through the public
    API, and is verified to produce identical output.

    Examples
    --------
    >>> import wadler_lindig as wl
    >>> from unxt._fmt import doc_to_str

    >>> doc = wl.TextDoc("[1., 2.]") + wl.BreakDoc(" ") + wl.TextDoc("m")
    >>> doc_to_str(wl.GroupDoc(doc))
    '[1., 2.] m'

    Narrow enough, and the break is taken:

    >>> print(doc_to_str(wl.GroupDoc(doc), 5))
    [1., 2.]
    m

    """
    return wl.pformat(_DocHolder(doc), width=width)


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
    """Render a quantity's value.

    A `jax.core.Tracer` forces the short form: under `jax.jit` only the shape
    and dtype exist, and `numpy.array2string` on a tracer raises -- so
    ``value_spec`` is silently unused there too, same as any other per-element
    detail a shape/dtype summary cannot show.

    ``value_spec``, when given, is a Python format spec (e.g. ``".3g"``)
    applied to every element via `numpy.array2string`'s ``formatter``, instead
    of NumPy's own default float rendering.
    """
    if isinstance(value, jax.core.Tracer):
        short_arrays = True
    if short_arrays == "compact":
        formatter = {"all": lambda v: format(v, value_spec)} if value_spec else None
        vsep = _markup_table(markup)["vsep"]
        return np.array2string(np.asarray(value), separator=vsep, formatter=formatter)
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
#: These are the *call-style* presets -- rendered by `wadler_lindig.pformat`,
#: like ``ClassName(value, unit=...)``. The *product-style* presets
#: (``value * unit``, rendered through `pparts`) are not listed here: they are
#: a small DSL instead of one dict entry per combination, since three
#: independent choices (markup, array verbosity, separator) would otherwise
#: need one hand-written entry per combination. See `_parse_product_spec`.
#:
#: Entries are plain kwarg bundles and a type honours the keys it understands,
#: which is what lets one table serve quantities and unit systems alike.
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
    "dims": {"show_units": False},
}

#: Markup component of a product-style spec, e.g. the ``html`` in
#: ``"html-bare"``. Omitted means ``"text"``; the token *is* the markup name,
#: so there is no separate value table like `_SEP_TOKENS` has.
_MARKUP_TOKENS: Final[frozenset[str]] = frozenset({"html", "latex"})

#: Separator component: whether the join between top-level parts (e.g. value
#: and unit) shows its operator. Value is the `parts_to_doc` / `parts_to_markup`
#: ``sep`` override; `None` means "don't override" -- the object's own `pparts`
#: already renders its default join (``" * "`` for a quantity), so ``"mul"``
#: does not have to hard-code that string here to mean the same thing.
_SEP_TOKENS: Final[dict[str, str | None]] = {"mul": None, "bare": " "}

#: Array component: how many array values render. ``"short"`` is a shape/dtype
#: summary (`short_arrays=True`, e.g. ``f32[3]``) with no per-element values, so
#: it cannot combine with a trailing format spec. ``"compact"`` is the default
#: form (`short_arrays="compact"`, e.g. ``[1., 2., 3.]``) and is nameable so it
#: can be paired with one, e.g. ``"compact-.3g"``. Omitted also means compact.
_ARRAY_TOKENS: Final[frozenset[str]] = frozenset({"short", "compact"})

#: Unit component: which of a unit's names renders it. ``"long"`` picks
#: `astropy.units.UnitBase.long_names` (e.g. ``"meter"``) over the default
#: short/symbol form (e.g. ``"m"``). A unit with no long name (a composite like
#: ``km / s``, or dimensionless) falls back to the default rather than raising.
_UNIT_TOKENS: Final[frozenset[str]] = frozenset({"long"})


def _parse_product_spec(spec: str, /) -> dict[str, Any] | None:
    """Parse a product-style spec: ``<markup>-<array>-<separator>-<unit>``.

    Each component is optional and independent, so a spec is really a *set* of
    up to four tokens -- one per component -- and this parser accepts them in
    any order (``"html-bare"`` and ``"bare-html"`` are the same request).
    ``<markup>-<array>-<separator>-<unit>`` is only the canonical spelling used
    in docs and error messages, not a grammar this function enforces; a fixed
    positional grammar would reject the shorthands (``"html"``, ``"mul"``,
    ``"short"``, bare ``".3g"``) that make single-component specs read the
    same as before this DSL existed.

    The array component alone may carry a trailing Python format spec, applied
    per element (e.g. ``"compact-.3g"``, or just ``".3g"`` -- ``"compact"`` is
    implied). That spec is reassembled from whatever pieces are left over
    after every recognised keyword is pulled out, in their original order, so
    a spec with its own embedded ``-`` (a sign flag, or a custom fill
    character) survives being combined with another component:

    >>> from unxt._src.fmt import _parse_product_spec
    >>> _parse_product_spec("mul-->10.2f")["value_spec"]  # fill='-', align='>'
    '->10.2f'

    Returns `None` -- not a preset -- for anything that is not entirely
    composed of known tokens plus at most one such leftover fragment: a
    duplicated component (``"mul-bare"``), ``"short"`` combined with a value
    spec (no per-element values to format), or a spec with *no* recognised
    keyword at all, like a plain value spec (``".2f"``). That last case is
    what lets `pspec` fall through to the unchanged `pspec_fallback` for a
    bare value spec -- a keyword must be present for this DSL to claim the
    spec at all, so ``".2f"`` alone is untouched by this parser.

    Examples
    --------
    >>> _parse_product_spec("html-bare") == {
    ...     "markup": "html",
    ...     "sep": " ",
    ...     "short_arrays": "compact",
    ...     "value_spec": None,
    ...     "long_unit": False,
    ... }
    True

    Omitted components take their default -- text markup, the object's own
    separator, compact arrays, the short unit name:

    >>> _parse_product_spec("short") == {
    ...     "markup": "text",
    ...     "sep": None,
    ...     "short_arrays": True,
    ...     "value_spec": None,
    ...     "long_unit": False,
    ... }
    True

    A value spec composes with any other component, e.g. markup and unit:

    >>> _parse_product_spec("html-.3g-long") == {
    ...     "markup": "html",
    ...     "sep": None,
    ...     "short_arrays": "compact",
    ...     "value_spec": ".3g",
    ...     "long_unit": True,
    ... }
    True

    Not a product spec -- a plain value spec, a duplicated component, or
    ``short`` combined with a value spec:

    >>> _parse_product_spec(".2f") is None
    True
    >>> _parse_product_spec("mul-bare") is None
    True
    >>> _parse_product_spec("short-.2f") is None
    True

    """
    pieces = spec.split("-")
    markup_i = [i for i, p in enumerate(pieces) if p in _MARKUP_TOKENS]
    sep_i = [i for i, p in enumerate(pieces) if p in _SEP_TOKENS]
    array_i = [i for i, p in enumerate(pieces) if p in _ARRAY_TOKENS]
    unit_i = [i for i, p in enumerate(pieces) if p in _UNIT_TOKENS]
    if max(len(markup_i), len(sep_i), len(array_i), len(unit_i)) > 1:
        return None
    claimed = {*markup_i, *sep_i, *array_i, *unit_i}
    if not claimed:
        return None

    value_spec = "-".join(p for i, p in enumerate(pieces) if i not in claimed)
    is_short = bool(array_i) and pieces[array_i[0]] == "short"
    if is_short and value_spec:
        return None

    return {
        "markup": pieces[markup_i[0]] if markup_i else "text",
        "sep": _SEP_TOKENS[pieces[sep_i[0]]] if sep_i else None,
        "short_arrays": True if is_short else "compact",
        "value_spec": value_spec or None,
        "long_unit": bool(unit_i),
    }


def bad_spec(obj: Any, spec: str, /) -> ValueError:
    return ValueError(
        f"invalid format spec {spec!r} for {type(obj).__name__}; "
        f"presets are {', '.join(sorted(FORMAT_PRESETS))}, or a "
        "'-'-joined combination of up to four parts, each optional: markup "
        f"({', '.join(sorted(_MARKUP_TOKENS))}; default text), array "
        f"({', '.join(sorted(_ARRAY_TOKENS))}, optionally with a trailing "
        "'-<python format spec>' applied per element, e.g. 'compact-.3g'; "
        f"default compact), separator ({', '.join(sorted(_SEP_TOKENS))}; "
        f"default mul), and unit ({', '.join(sorted(_UNIT_TOKENS))}; default "
        "the short name) -- e.g. 'html-bare', 'mul-.3g', or "
        "'html-compact-.2f-bare-long'"
    )


def pspec(obj: Any, spec: str, /, *, width: int = 88) -> str:
    r"""Implement ``__format__`` for an object the engine knows.

    The preset/DSL lookups run *before* the value-spec branch. That ordering
    is mandatory rather than stylistic: handing a non-empty spec straight to
    the value raises for a tracer and for any non-scalar array, so a preset
    checked second would be unreachable under `jax.jit` and for every array
    quantity.

    Examples
    --------
    >>> import unxt as u
    >>> from unxt._fmt import pspec

    >>> pspec(u.Q([1.0, 2, 3], "m"), "mul")
    '[1., 2., 3.] * m'

    >>> pspec(u.Q([1.0, 2, 3], "m"), "latex")
    '$[1.,~2.,~3.] \\; \\mathrm{m}$'

    A product-style spec composes markup, array, separator, and unit --
    each optional, in any order:

    >>> pspec(u.Q([1.0, 2, 3], "m"), "html-bare")
    '<span>[1., 2., 3.]</span> <span>m</span>'

    >>> pspec(u.Q([1.0, 2, 3], "m"), "html-short-mul")
    '<span>f32[3]</span> * <span>m</span>'

    The array component may carry a per-element Python format spec, and the
    unit component may pick the long name over the short/symbol form:

    >>> pspec(u.Q([1.234, 2.345], "m"), "mul-.2f")
    '[1.23, 2.35] * m'

    >>> pspec(u.Q(1.0, "m"), "long")
    '1. * meter'

    An empty spec preserves `str`:

    >>> pspec(u.Q(1.0, "m"), "") == str(u.Q(1.0, "m"))
    True

    """
    if not spec:
        return str(obj)

    if spec in FORMAT_PRESETS:
        # Checked before the DSL parse: "compact" is both a call-style preset
        # here *and* a valid array-component token there (so it can pair with
        # a value spec, e.g. "compact-.3g"). A literal preset name must win --
        # otherwise the pre-existing, documented ``f"{q:compact}"`` would be
        # silently reinterpreted as the array component alone.
        #
        # ``wadler_lindig.pformat`` accepts and ignores unknown kwargs, which is
        # what makes one preset table serve several types.
        return wl.pformat(obj, **FORMAT_PRESETS[spec])

    if (parsed := _parse_product_spec(spec)) is not None:
        markup = parsed["markup"]
        sep = parsed["sep"]
        parts = pparts(
            obj,
            markup=markup,
            short_arrays=parsed["short_arrays"],
            value_spec=parsed["value_spec"],
            long_unit=parsed["long_unit"],
        )
        if markup == "text":
            # Feed wadler-lindig, so the rendering is laid out rather than
            # concatenated and composes inside a larger document.
            return doc_to_str(parts_to_doc(parts, sep=sep), width)
        return parts_to_markup(parts, markup=markup, sep=sep)

    return pspec_fallback(obj, spec)


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
    invalid format spec 'nope' for LTMAUnitSystem; presets are compact, dims, ...

    """
    raise bad_spec(obj, spec)
