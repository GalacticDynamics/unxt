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
    "ALIASES",
    "MARKUPS",
    "PGroup",
    "PPart",
    "Spec",
    "bad_spec",
    "custom_pdoc_no_kind",
    "custom_pdoc_noarray",
    "doc_to_str",
    "parse_spec",
    "parts_to_doc",
    "parts_to_markup",
    "pparts",
    "pspec",
    "render",
    "unwrap_math",
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
    """
    if isinstance(value, jax.core.Tracer):
        short_arrays = True
    if short_arrays == "compact":
        formatter = {"all": lambda v: format(v, value_spec)} if value_spec else None
        vsep = _markup_table(markup)["vsep"]
        return np.array2string(np.asarray(value), separator=vsep, formatter=formatter)
    # ``show_wrapper=False`` is for ``StaticValue``, whose ``__pdoc__`` would
    # otherwise print ``StaticValue(...)`` around the array. ``short_arrays``
    # is forwarded rather than pinned: `False` is the *full* array repr, and
    # hardcoding `True` here collapsed it onto the summary.
    #
    # ``custom_pdoc_no_kind`` only applies to the summary, where it strips the
    # ``(numpy)`` kind suffix. It *builds* a summary, so passing it on the
    # `False` path would force one back and undo the distinction. There is no
    # "no hook" argument -- ``custom=None`` is called and raises -- so the
    # kwarg has to be omitted rather than blanked.
    kw = {"custom": custom_pdoc_no_kind} if short_arrays else {}
    return wl.pformat(value, short_arrays=short_arrays, show_wrapper=False, **kw)


@dispatch.abstract
def pparts(obj: Any, /, *, markup: str = "text", **kw: Any) -> tuple[Any, ...]:
    """Decompose an object into a tree of `PPart` / `PGroup` fragments.

    This is the extension point: register an implementation for your type and
    it gains every preset, every markup, and the wadler-lindig layout path.
    """
    raise NotImplementedError  # pragma: no cover


@dispatch  # type: ignore[no-redef]
def pparts(
    obj: Any, /, *, markup: str = "text", value_spec: str | None = None, **kw: Any
) -> tuple[Any, ...]:
    """Fall back to `str` for a type with no registration.

    This is a *display* path, so an unregistered type must degrade rather than
    raise: without this method one unregistered field would poison an entire
    object's `_repr_html_` in a notebook cell. `__pdoc__` already degrades this
    way through wadler-lindig's dataclass fallback.

    A ``value_spec`` is the one thing this cannot degrade on. It formats
    *elements*, and a type that has not said what its elements are has none to
    format -- so silently dropping it would answer a specific request with a
    different rendering. That is an error, not a fallback.

    Examples
    --------
    >>> from unxt._fmt import pparts
    >>> pparts(object())
    (PPart(role='value', text='<object object at ...>', kind='content'),)

    """
    if value_spec is not None:
        msg = (
            f"{type(obj).__name__} does not support a value format spec "
            f"({value_spec!r}): it registers no `pparts`, so it has no "
            "elements to format"
        )
        raise TypeError(msg)
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


#: Every keyword in the grammar, mapped to the axis it sets and the value it
#: sets that axis to.
#:
#: Keywords live in one flat namespace and must stay **pairwise disjoint** --
#: no word may name two axes. That invariant is what lets the keyword run be
#: order-independent (``"html-bare"`` and ``"bare-html"`` are the same request)
#: with no content-sniffing anywhere, and it is checked by a test rather than
#: left to reviewer diligence.
_KEYWORDS: Final[dict[str, tuple[str, Any]]] = {
    # layout -- how the pieces are arranged
    "call": ("layout", "call"),  # Quantity(Array([1., 2.]), unit='m')
    "product": ("layout", "product"),  # [1., 2.] m
    # value -- how the numeric payload renders
    "array": ("value", "array"),  # Array([1., 2.], dtype=float32)
    "values": ("value", "values"),  # [1., 2.]
    "type": ("value", "type"),  # f32[2]
    # markup
    "text": ("markup", "text"),
    "html": ("markup", "html"),
    "latex": ("markup", "latex"),
    # unit -- which spelling of the unit
    "symbol": ("unit", "symbol"),  # m
    "name": ("unit", "name"),  # meter
    "dim": ("unit", "dim"),  # length
    # separator -- product layout only
    "mul": ("sep", "mul"),  # [1., 2.] * m
    "bare": ("sep", "bare"),  # [1., 2.] m
    # abbreviation -- call layout only
    "abbrev": ("abbrev", True),  # Q(...) rather than Quantity(...)
}

#: Shorthands for whole specs. Each expands *textually* into core keywords
#: before parsing, so an alias can never mean something the core grammar cannot
#: already say, and combining one with a further keyword raises exactly the
#: error its expansion would.
ALIASES: Final[dict[str, str]] = {
    "compact": "call-abbrev",
    "full": "call-array",
    "dims": "call-dim",
}

#: The axes each layout has a concept of. Naming any other is an error, not a
#: silent no-op: ``f"{q:call-mul}"`` says something about a separator that
#: call layout does not have, and quietly dropping it would hide the mistake.
_LAYOUT_AXES: Final[dict[str, frozenset[str]]] = {
    "call": frozenset({"layout", "value", "unit", "abbrev"}),
    "product": frozenset({"layout", "value", "unit", "markup", "sep"}),
}

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
VALUE_FROM_SHORT_ARRAYS: Final[dict[Any, str]] = {
    False: "array",
    "compact": "values",
    True: "type",
}

#: The ``sep`` axis as a `parts_to_doc` / `parts_to_markup` override. ``None``
#: means "do not override", leaving whatever the object's own `pparts` emitted
#: (``" * "`` for a quantity) -- so ``"mul"`` need not hard-code that string.
_SEPARATORS: Final[dict[str, str | None]] = {"mul": None, "bare": " "}


class Spec(NamedTuple):
    """A fully resolved format spec: one settled value per axis.

    Field defaults are the grammar's defaults, so ``Spec()`` is what an
    all-omitted spec means.
    """

    layout: str = "product"
    value: str = "values"
    value_spec: str | None = None
    markup: str = "text"
    unit: str = "symbol"
    sep: str = "bare"
    abbrev: bool = False


def _grammar_help() -> str:
    """Describe the grammar, generated from the tables so it cannot drift."""
    axes: dict[str, list[str]] = {}
    for word, (axis, _) in _KEYWORDS.items():
        axes.setdefault(axis, []).append(word)
    parts = [f"{axis} ({'|'.join(words)})" for axis, words in axes.items()]
    return (
        "a '-'-joined run of keywords, then an optional Python format spec "
        "applied per element: "
        + ", ".join(parts)
        + "; aliases: "
        + ", ".join(f"{k}={v}" for k, v in ALIASES.items())
    )


def bad_spec(obj: Any, spec: str, /, reason: str = "") -> ValueError:
    """Build the one error every rejected spec raises."""
    who = f" for {type(obj).__name__}" if obj is not None else ""
    why = f": {reason}" if reason else ""
    return ValueError(
        f"invalid format spec {spec!r}{who}{why}. Expected {_grammar_help()}"
    )


def _scan_keywords(
    tokens: list[str], spec: str, obj: Any, /
) -> tuple[dict[str, Any], int]:
    """Consume leading keyword tokens; return what they set and where they end.

    Stops at the first token that is not a keyword (an alias counts, expanded
    in place). The caller takes everything from that index on as the value
    spec -- which is what keeps a format spec's own ``-`` from being read as a
    component boundary.
    """
    seen: dict[str, Any] = {}
    for i, token in enumerate(tokens):
        words = ALIASES.get(token, token).split("-")
        if not all(word in _KEYWORDS for word in words):
            return seen, i
        for word in words:
            axis, value = _KEYWORDS[word]
            if axis in seen:
                msg = f"{axis!r} is set twice"
                raise bad_spec(obj, spec, msg)
            seen[axis] = value
    return seen, len(tokens)


def parse_spec(spec: str, /, *, obj: Any = None) -> Spec:
    r"""Resolve a format spec into a `Spec`, or raise `ValueError`.

    The parse is total and strictly left-to-right: split on ``-``, consume
    tokens while they are keywords (expanding an alias in place), and stop at
    the first token that is not one. **Everything from that token onward --
    including any further ``-`` -- is the value spec.**

    That one rule is what keeps the grammar unambiguous once an arbitrary
    Python format spec is in play. A format spec may contain ``-`` itself, as
    a sign flag (``"-.2f"``) or a fill character (``"->10.2f"``), and neither
    can be mistaken for a component boundary, because keywords are only
    recognised *before* the value spec begins.

    Examples
    --------
    >>> from unxt._src.fmt import parse_spec

    Keywords set their axis; everything omitted keeps its default:

    >>> parse_spec("html-bare").markup, parse_spec("html-bare").sep
    ('html', 'bare')

    Order among keywords does not matter:

    >>> parse_spec("bare-html") == parse_spec("html-bare")
    True

    A trailing value spec is applied per element:

    >>> parse_spec("mul-.3g").value_spec
    '.3g'

    A value spec keeps its own ``-``, whether leading or embedded:

    >>> parse_spec("-.2f").value_spec
    '-.2f'
    >>> parse_spec("mul-->10.2f").value_spec
    '->10.2f'

    An alias expands before parsing:

    >>> parse_spec("compact") == parse_spec("call-abbrev")
    True

    """
    tokens = spec.split("-")
    seen, i = _scan_keywords(tokens, spec, obj)

    # Whatever the scan did not claim. `None` when it claimed everything; a
    # spec with no keyword at all is fine -- a bare ".3g" is the commonest
    # there is -- and simply leaves every axis at its default.
    value_spec = "-".join(tokens[i:]) or None

    layout = seen.get("layout", "product")
    allowed = _LAYOUT_AXES[layout]
    for axis in seen:
        if axis not in allowed:
            msg = f"{axis!r} does not apply to {layout!r} layout"
            raise bad_spec(obj, spec, msg)

    value = seen.get("value", "values")
    if value_spec is not None:
        if value != "values":
            msg = f"a value format spec needs value='values', not {value!r}"
            raise bad_spec(obj, spec, msg)
        if layout != "product":
            msg = f"a value format spec does not apply to {layout!r} layout"
            raise bad_spec(obj, spec, msg)

    return Spec(
        layout=layout,
        value=value,
        value_spec=value_spec,
        markup=seen.get("markup", "text"),
        unit=seen.get("unit", "symbol"),
        sep=seen.get("sep", "bare"),
        abbrev=seen.get("abbrev", False),
    )


def render(
    obj: Any, spec: Spec, /, *, width: int = 88, indent: int = 4, **pdoc_kw: Any
) -> str:
    r"""Render ``obj`` according to an already-resolved `Spec`.

    The single rendering entry point: ``repr``, ``str`` and ``__format__`` all
    arrive here, differing only in the `Spec` they bring.

    ``call`` layout goes through `wadler_lindig.pformat`, and so through the
    object's own ``__pdoc__``. That is deliberate and load-bearing: ``__pdoc__``
    is where a type states how to *reconstruct* it, which is what keeps
    ``eval(repr(x)) == x`` true for unit systems.

    Examples
    --------
    >>> import unxt as u
    >>> from unxt._src.fmt import parse_spec, render

    >>> render(u.Q([1.0, 2, 3], "m"), parse_spec("mul"))
    '[1., 2., 3.] * m'

    >>> render(u.Q([1.0, 2, 3], "m"), parse_spec("call"))
    "Quantity([1., 2., 3.], unit='m')"

    """
    if spec.layout == "call":
        # ``pdoc_kw`` carries type-specific ``__pdoc__`` knobs that are not
        # grammar axes (a quantity's ``named_unit``, say). They stay off the
        # spec vocabulary -- one word must mean one thing for every type -- but
        # still need a route through, so `unxt.config` can drive them.
        return wl.pformat(
            obj,
            width=width,
            indent=indent,
            short_arrays=_SHORT_ARRAYS[spec.value],
            use_short_name=spec.abbrev,
            # The abbreviated call form is one idea, spelled per type: a short
            # class name for a quantity, unquoted units for a unit system.
            quote_units=not spec.abbrev,
            show_units=spec.unit != "dim",
            **pdoc_kw,
        )

    parts = pparts(
        obj,
        markup=spec.markup,
        short_arrays=_SHORT_ARRAYS[spec.value],
        value_spec=spec.value_spec,
        unit=spec.unit,
    )
    sep = _SEPARATORS[spec.sep]
    if spec.markup == "text":
        # Feed wadler-lindig, so the rendering is laid out rather than
        # concatenated and composes inside a larger document.
        return doc_to_str(parts_to_doc(parts, indent=indent, sep=sep), width)
    return parts_to_markup(parts, markup=spec.markup, sep=sep)


def pspec(obj: Any, spec: str, /, *, width: int = 88) -> str:
    r"""Implement ``__format__`` for an object the engine knows.

    Examples
    --------
    >>> import unxt as u
    >>> from unxt._fmt import pspec

    >>> pspec(u.Q([1.0, 2, 3], "m"), "mul")
    '[1., 2., 3.] * m'

    >>> pspec(u.Q([1.0, 2, 3], "m"), "latex")
    '$[1.,~2.,~3.] \\mathrm{m}$'

    A value spec is applied per element, and works on an array -- there is one
    value-rendering path, so it does not matter whether a keyword accompanies
    it:

    >>> pspec(u.Q([1.234, 2.345], "m"), ".2f")
    '[1.23, 2.35] m'
    >>> pspec(u.Q([1.234, 2.345], "m"), "mul-.2f")
    '[1.23, 2.35] * m'

    The unit axis picks a spelling:

    >>> pspec(u.Q(1.0, "m"), "name")
    '1. meter'

    An empty spec is `str`:

    >>> pspec(u.Q(1.0, "m"), "") == str(u.Q(1.0, "m"))
    True

    """
    if not spec:
        return str(obj)
    parsed = parse_spec(spec, obj=obj)
    try:
        return render(obj, parsed, width=width)
    except ValueError as e:
        if parsed.value_spec is None:
            raise
        # Anything that is not a keyword is taken as a Python format spec, so
        # a mistyped keyword arrives here as `float.__format__` rejecting it.
        # Python's message names the offending text but not the vocabulary it
        # missed, which is exactly what a typo needs to see.
        msg = f"{parsed.value_spec!r} is not a valid Python format spec ({e})"
        raise bad_spec(obj, spec, msg) from e
