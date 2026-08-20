"""The string-formatting engine.

**This module is domain-agnostic and self-contained.** It knows nothing about
quantities, units, arrays, or `jax`, and imports nothing from `unxt` -- see
`unxt._src.fmt.axes` for the layer that teaches it those. It is written to be
lifted out into a package of its own, with `unxt`, `coordinax` and `galax`
registering into it as peers; a test pins the import restriction so the seam
cannot rot.

That package is intended to be called **``pparts``**, after the extension
point everything here turns on: a type joins in by saying what it is *made
of*.

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

The engine feeds wadler-lindig; it does not replace it. ``__pdoc__`` stays the
definition of *call-style* rendering and is untouched by this module.

The format-spec grammar is likewise a *mechanism* here, not a vocabulary. The
scan rule, the `Spec` record, and the two layouts are built in; every axis and
keyword arrives through `register_axis` / `register_alias`, so a downstream
package's axis is indistinguishable from a core one.

"""

__all__ = (
    "ALIASES",
    "AXES",
    "Axis",
    "MARKUPS",
    "PGroup",
    "PPart",
    "REQUIRED_MARKUP_KEYS",
    "Spec",
    "bad_spec",
    "doc_to_str",
    "parse_spec",
    "parts_to_doc",
    "parts_to_markup",
    "pparts",
    "pspec",
    "register_alias",
    "register_axis",
    "render",
    "unwrap_math",
)

import html as _html
from collections.abc import Callable, Iterator, Mapping
from types import MappingProxyType
from typing import Any, Final, NamedTuple

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
    >>> from unxt._src.fmt import PPart
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
    >>> from unxt._src.fmt import PGroup, PPart
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
    fragment that arrives unwrapped -- ``\mathrm{m}`` becomes ``mathrm{m``.

    The length guard is load-bearing: a lone ``"$"`` satisfies both
    ``startswith`` and ``endswith``, and would otherwise be sliced away
    entirely.

    Examples
    --------
    >>> from unxt._src.fmt import unwrap_math

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
    >>> from unxt._src.fmt import doc_to_str

    >>> doc = wl.TextDoc("[1., 2.]") + wl.BreakDoc(" ") + wl.TextDoc("m")
    >>> doc_to_str(wl.GroupDoc(doc))
    '[1., 2.] m'

    Narrow enough, and the break is taken:

    >>> print(doc_to_str(wl.GroupDoc(doc), 5))
    [1., 2.]
    m

    """
    return wl.pformat(_DocHolder(doc), width=width)


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
    >>> from unxt._src.fmt import pparts
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
    >>> from unxt._src.fmt import parts_to_doc, pparts

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
    >>> from unxt._src.fmt import pparts, parts_to_markup

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


# ============================================================================
# The grammar: a registry, not a vocabulary


class Axis(NamedTuple):
    """One independent choice a format spec can make.

    Parameters
    ----------
    name
        The axis, and the key it occupies in a `Spec`.
    keywords
        Spec word -> the value it sets. Words live in one flat namespace
        shared by every axis, and must stay pairwise disjoint; `register_axis`
        enforces that, which is what lets a keyword run be order-independent
        with no content-sniffing.
    default
        The value when no keyword names this axis.
    layouts
        Layout -> a function turning this axis's value into keyword arguments
        for that layout's renderer. **Membership is applicability**: an axis
        applies to exactly the layouts it has an entry for, so naming it under
        any other layout is an error rather than a silent no-op.
    free_text
        The layouts in which this axis's value may instead be *arbitrary text*
        -- the trailing run a spec ends with, such as a Python format spec.
        Empty for a closed axis, which is most of them.

        At most one axis may claim free text, and the engine enforces it. That
        is not a restriction so much as an observation: the scan rule makes the
        trailing run terminal, so there is only one of them to claim.

    """

    name: str
    keywords: Mapping[str, Any]
    default: Any
    layouts: Mapping[str, Callable[[Any], Mapping[str, Any]]]
    free_text: tuple[str, ...] = ()


#: Registered axes, by name. Populated only through `register_axis`.
AXES: Final[dict[str, Axis]] = {}

#: Spec word -> the axis that claimed it. The flat namespace `Axis.keywords`
#: describes, maintained here so a collision is caught at registration.
_KEYWORDS: Final[dict[str, str]] = {}

#: Shorthands for whole specs. Each expands *textually* into core keywords
#: before parsing, so an alias can never mean something the grammar cannot
#: already say, and combining one with a further keyword raises exactly the
#: error its expansion would.
ALIASES: Final[dict[str, str]] = {}


def _free_text_axis() -> Axis | None:
    """Return the axis accepting free text, if one is registered.

    Derived rather than maintained, so it cannot fall out of step with `AXES`.
    `register_axis` permits only one claimant.
    """
    return next((ax for ax in AXES.values() if ax.free_text), None)


def register_axis(axis: Axis, /) -> Axis:
    """Add an axis to the grammar, rejecting any keyword collision.

    Registration is the only way in. A downstream package registering
    ``vector_form`` gets exactly what the built-in axes get -- there is no
    privileged set.
    """
    if axis.name in AXES:
        msg = f"axis {axis.name!r} is already registered"
        raise ValueError(msg)
    if axis.free_text and (claimed := _free_text_axis()) is not None:
        msg = (
            f"axis {axis.name!r} claims free text, but {claimed.name!r} "
            "already does; a spec has only one trailing run to give"
        )
        raise ValueError(msg)
    for word in axis.keywords:
        if word in _KEYWORDS:
            msg = f"keyword {word!r} is already claimed by axis {_KEYWORDS[word]!r}"
            raise ValueError(msg)
        if word in ALIASES:
            msg = f"keyword {word!r} is already an alias"
            raise ValueError(msg)
    AXES[axis.name] = axis
    _KEYWORDS.update(dict.fromkeys(axis.keywords, axis.name))
    return axis


def register_alias(name: str, expansion: str, /) -> None:
    """Add a whole-spec shorthand, rejecting any collision.

    Both directions are checked, because both are the same mistake: a name
    that already means something must not quietly start meaning something
    else. Silently overwriting is how a spec changes meaning without anyone
    editing the spec.
    """
    if name in _KEYWORDS:
        msg = f"alias {name!r} is already a keyword of axis {_KEYWORDS[name]!r}"
        raise ValueError(msg)
    if name in ALIASES:
        msg = f"alias {name!r} is already registered as {ALIASES[name]!r}"
        raise ValueError(msg)
    ALIASES[name] = expansion


#: Layout -> the function that renders an object in it. A layout is a way of
#: arranging an object's parts, so this stays engine-owned: `register_axis`
#: extends the *vocabulary*, not the set of arrangements.
_LAYOUTS: Final[dict[str, Callable[..., str]]] = {}


class Spec(Mapping[str, Any]):
    """A fully resolved format spec: one settled value for every axis.

    A mapping rather than a record, so a downstream axis is read exactly like
    a built-in one (``spec["markup"]``, ``spec["vector_form"]``). Defaults are
    filled in at parse time, so every registered axis is always present and a
    reader never has to know which were named.
    """

    __slots__ = ("_d",)

    def __init__(self, mapping: Mapping[str, Any] = (), /, **kw: Any) -> None:
        self._d: Mapping[str, Any] = MappingProxyType({**dict(mapping), **kw})

    @classmethod
    def of(cls, /, **overrides: Any) -> "Spec":
        """Build a resolved spec, defaulting every axis not named.

        This is the way to construct one by hand. Taking the defaults from the
        registry is what keeps a hand-built spec complete once a *later* axis
        is registered -- writing the mapping out directly leaves a hole that
        surfaces only as a `KeyError` at render time.

        Examples
        --------
        >>> from unxt._src.fmt import Spec, parse_spec

        >>> Spec.of(layout="call")["unit"]
        'symbol'

        >>> Spec.of() == parse_spec("product")
        True

        """
        unknown = set(overrides) - set(AXES)
        if unknown:
            msg = f"not registered axes: {sorted(unknown)}"
            raise ValueError(msg)
        return cls({n: overrides.get(n, ax.default) for n, ax in AXES.items()})

    def __getitem__(self, key: str) -> Any:
        return self._d[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._d)

    def __len__(self) -> int:
        return len(self._d)

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self._d.items())
        return f"Spec({args})"


def _grammar_help() -> str:
    """Describe the grammar, generated from the registry so it cannot drift."""
    parts = [f"{name} ({'|'.join(ax.keywords)})" for name, ax in AXES.items()]
    aliases = ", ".join(f"{k}={v}" for k, v in ALIASES.items())
    return (
        "a '-'-joined run of keywords, then an optional Python format spec "
        "applied per element: " + ", ".join(parts) + f"; aliases: {aliases}"
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
            axis = _KEYWORDS[word]
            if axis in seen:
                msg = f"{axis!r} is set twice"
                raise bad_spec(obj, spec, msg)
            seen[axis] = AXES[axis].keywords[word]
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

    >>> parse_spec("html-bare")["markup"], parse_spec("html-bare")["sep"]
    ('html', 'bare')

    Order among keywords does not matter:

    >>> parse_spec("bare-html") == parse_spec("html-bare")
    True

    A trailing value spec is applied per element:

    >>> parse_spec("mul-.3g")["value"]
    '.3g'

    A value spec keeps its own ``-``, whether leading or embedded:

    >>> parse_spec("-.2f")["value"]
    '-.2f'
    >>> parse_spec("mul-->10.2f")["value"]
    '->10.2f'

    An alias expands before parsing:

    >>> parse_spec("compact") == parse_spec("call-abbrev")
    True

    """
    tokens = spec.split("-")
    seen, i = _scan_keywords(tokens, spec, obj)

    # Whatever the scan did not claim goes to the axis that accepts free text,
    # as that axis's value -- which is what makes "a keyword and free text for
    # one axis" the ordinary set-twice error below. A spec with no keyword at
    # all is fine; a bare ".3g" is the commonest there is.
    free_text = "-".join(tokens[i:])
    text_axis = _free_text_axis() if free_text else None
    if text_axis is not None:
        if text_axis.name in seen:
            msg = f"{text_axis.name!r} is set twice"
            raise bad_spec(obj, spec, msg)
        seen[text_axis.name] = free_text

    resolved = {name: seen.get(name, ax.default) for name, ax in AXES.items()}
    layout = resolved["layout"]

    for axis in seen:
        if layout not in AXES[axis].layouts:
            msg = f"{axis!r} does not apply to {layout!r} layout"
            raise bad_spec(obj, spec, msg)

    if text_axis is not None and layout not in text_axis.free_text:
        msg = f"free text does not apply to {layout!r} layout"
        raise bad_spec(obj, spec, msg)

    return Spec(resolved)


def _layout_kwargs(spec: Spec, layout: str, /) -> dict[str, Any]:
    """Translate every axis applicable to ``layout`` into renderer kwargs.

    An axis contributes only where it declared a translation, so an axis a
    layout has no concept of contributes nothing -- and `parse_spec` has
    already refused to let one be *named* under that layout.
    """
    kw: dict[str, Any] = {}
    for name, ax in AXES.items():
        translate = ax.layouts.get(layout)
        if translate is not None:
            kw.update(translate(spec[name]))
    return kw


def render(
    obj: Any, spec: Spec, /, *, width: int = 88, indent: int = 4, **extra: Any
) -> str:
    r"""Render ``obj`` according to an already-resolved `Spec`.

    The single rendering entry point: ``repr``, ``str`` and ``__format__`` all
    arrive here, differing only in the `Spec` they bring.

    ``extra`` carries renderer arguments that are not grammar axes -- a type's
    own ``__pdoc__`` knob, say. They stay out of the spec vocabulary, because
    one word must mean one thing for every type, but still need a route
    through so a caller's configuration can drive them.

    Examples
    --------
    >>> import unxt as u
    >>> from unxt._src.fmt import parse_spec, render

    >>> render(u.Q([1.0, 2, 3], "m"), parse_spec("mul"))
    '[1., 2., 3.] * m'

    >>> render(u.Q([1.0, 2, 3], "m"), parse_spec("call"))
    "Quantity([1., 2., 3.], unit='m')"

    """
    layout = spec["layout"]
    return _LAYOUTS[layout](
        obj, spec, width=width, indent=indent, **_layout_kwargs(spec, layout), **extra
    )


def _render_call(obj: Any, spec: Spec, /, *, width: int, indent: int, **kw: Any) -> str:
    """Render through `wadler_lindig.pformat`, and so the object's ``__pdoc__``.

    That indirection is load-bearing rather than incidental: ``__pdoc__`` is
    where a type states how to *reconstruct* itself, which is what keeps
    ``eval(repr(x)) == x`` true for the types that promise it.
    """
    return wl.pformat(obj, width=width, indent=indent, **kw)


def _render_product(
    obj: Any, spec: Spec, /, *, width: int, indent: int, **kw: Any
) -> str:
    """Render as juxtaposed parts, via `pparts`.

    ``markup`` and ``sep`` are the engine's own layout parameters; everything
    else -- including any downstream axis -- is forwarded to `pparts`, which
    is what lets a type act on an axis the engine has never heard of.
    """
    markup = kw.pop("markup", "text")
    sep = kw.pop("sep", None)
    parts = pparts(obj, markup=markup, **kw)
    if markup == "text":
        # Feed wadler-lindig, so the rendering is laid out rather than
        # concatenated and composes inside a larger document.
        return doc_to_str(parts_to_doc(parts, indent=indent, sep=sep), width)
    return parts_to_markup(parts, markup=markup, sep=sep)


_LAYOUTS["call"] = _render_call
_LAYOUTS["product"] = _render_product

#: The ``layout`` axis is the one the engine must own: its keywords name the
#: renderers in `_LAYOUTS`, so it cannot be supplied by a consumer. It
#: contributes no renderer kwargs -- it *selects* the renderer.
#:
#: Both its tables are derived from `_LAYOUTS` rather than written out, so the
#: layout names live in exactly one place.
register_axis(
    Axis(
        name="layout",
        keywords={name: name for name in _LAYOUTS},
        default="product",
        layouts=dict.fromkeys(_LAYOUTS, lambda _: {}),
    )
)


def pspec(obj: Any, spec: str, /, *, width: int = 88) -> str:
    r"""Implement ``__format__`` for an object the engine knows.

    Examples
    --------
    >>> import unxt as u
    >>> from unxt._src.fmt import pspec

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
        # Anything that is not a keyword was taken as a Python format spec.
        # If none was, this error is not about one and belongs to the caller.
        axis = _free_text_axis()
        text = None if axis is None else parsed[axis.name]
        if text is None or text in axis.keywords.values():
            raise
        # Anything that is not a keyword is taken as a Python format spec, so
        # a mistyped keyword arrives here as the value formatter rejecting it.
        # That message names the offending text but not the vocabulary it
        # missed, which is exactly what a typo needs to see.
        msg = f"{text!r} is not a valid Python format spec ({e})"
        raise bad_spec(obj, spec, msg) from e
