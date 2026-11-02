"""Tests for the formatting engine itself -- the future `wadler-lindig-parts`.

Nothing here reaches for a quantity, a unit or an array library: these are the
tests that travel with `engine.py` and `generic.py` when they become their own
package, and they must pass with none of unxt's axes registered.
"""

import pathlib
import re

import pytest
import wadler_lindig as wl

from unxt._src.fmt import (
    ALIASES,
    AXES,
    MARKUPS,
    REQUIRED_MARKUP_KEYS,
    Axis,
    PGroup,
    PPart,
    Spec,
    doc_part,
    doc_to_str,
    engine as engine_module,
    examples,
    inert_axes,
    parse_spec,
    parts_to_doc,
    parts_to_markup,
    pparts,
    pspec,
    pvalue,
    register_alias,
    register_axis,
    register_markup,
    render,
    unregister_alias,
    unregister_axis,
    unregister_markup,
    unwrap_math,
)
from unxt._src.fmt.engine import _ALIASES, _AXES, _KEYWORDS, _MARKUPS
from unxt._src.fmt.generic import _MARKUP_WORDS, _check_markup_row

# ============================================================================
# The grammar's own invariants


def test_keyword_namespace_is_pairwise_disjoint() -> None:
    """One word may never name two axes.

    This is the invariant that lets the keyword run be order-independent with
    no content-sniffing: if `_KEYWORDS` is a function of the word alone, then
    ``"html-bare"`` and ``"bare-html"`` cannot differ. A dict literal already
    enforces uniqueness of *keys*, so the real risk is a new word being added
    that collides with an `ALIASES` entry -- check that too.
    """
    assert not set(_KEYWORDS) & set(ALIASES)


def test_every_axis_applies_to_some_layout() -> None:
    """An axis no layout accepts would be unreachable."""
    for name, ax in AXES.items():
        assert ax.layouts, name


def test_every_keyword_maps_back_to_its_axis() -> None:
    """`_KEYWORDS` is the flat namespace `Axis.keywords` describes."""
    for name, ax in AXES.items():
        for word in ax.keywords:
            assert name in _KEYWORDS[word]


def test_spec_defaults_are_the_grammar_defaults() -> None:
    """An all-omitted spec is exactly ``Spec.of()``."""
    assert parse_spec("product") == Spec.of()


def test_keywords_are_order_independent() -> None:
    """A keyword run is a set, not a sequence."""
    assert parse_spec("html-bare") == parse_spec("bare-html")
    assert parse_spec("latex-type-mul") == parse_spec("mul-latex-type")


@pytest.mark.parametrize(
    ("spec", "value_spec"),
    [
        (".3g", ".3g"),  # bare: no keyword at all
        ("mul-.3g", ".3g"),  # after a keyword
        ("-.2f", "-.2f"),  # leading '-' is a sign flag, not a delimiter
        ("mul-->10.2f", "->10.2f"),  # embedded '-' is a fill character
        ("mul-.2f-.3g", ".2f-.3g"),  # everything after the first non-keyword
        ("mul", "values"),  # keywords only: the axis keeps its default
    ],
)
def test_scan_rule_splits_keywords_from_the_value_spec(
    spec: str, value_spec: str | None
) -> None:
    """The first non-keyword token ends keyword parsing; the rest is the spec.

    This one rule is what keeps the grammar unambiguous with an arbitrary
    Python format spec in play -- a format spec may contain ``-`` itself, and
    it can never be mistaken for a component boundary.
    """
    assert parse_spec(spec)["value"] == value_spec


@pytest.mark.parametrize("markup", sorted(MARKUPS))
def test_markup_row_defines_the_required_keys(markup: str) -> None:
    """These are the keys with no per-fragment fallback."""
    for key in REQUIRED_MARKUP_KEYS:
        assert key in MARKUPS[markup]


def test_plain_roles_are_escaped() -> None:
    """A downstream role carries a class name; ``<``/``&``/``_``/``%`` break output."""
    assert parts_to_markup((PPart("frame", "<A & B>"),), markup="html") == (
        "<span>&lt;A &amp; B&gt;</span>"
    )
    assert parts_to_markup((PPart("frame", "a_b 50%"),), markup="latex") == (
        r"$a\_b 50\%$"
    )


def _long_term_doc() -> wl.AbstractDoc:
    """A rendering long enough to break, owing nothing to units."""
    return parts_to_doc(pparts(examples.Term([float(i) for i in range(8)], "x")))


def test_layout_stays_inline_when_it_fits() -> None:
    assert "\n" not in doc_to_str(_long_term_doc(), 100)


def test_separator_ink_survives_a_break() -> None:
    """``BreakDoc`` shows its text only in horizontal mode.

    Mapping ``" * "`` straight to a ``BreakDoc`` silently dropped the ``*`` on
    the broken line.
    """
    out = doc_to_str(_long_term_doc(), 20)
    assert "\n" in out
    assert out.rstrip().endswith("x")
    assert "*" in out


def test_adjacent_separators_do_not_emit_a_blank_line() -> None:
    """A separator with no trailing space must not offer a break."""
    parts = (PPart("close", ")", "sep"), PPart("gap", " ", "sep"), PPart("frame", "@x"))
    assert not re.search(r"\n\s*\n", doc_to_str(parts_to_doc(parts), 3))


def _vector_parts(markup: str = "text") -> tuple:
    """A composite of two child renderings, owing nothing to units."""
    x = examples.Term([float(i) for i in range(6)], "x")
    y = examples.Term([float(i) for i in range(6)], "y")
    return (
        PPart("open", "(", "sep"),
        PGroup("child", pparts(x, markup=markup)),
        PPart("comma", ", ", "sep"),
        PGroup("child", pparts(y, markup=markup)),
        PPart("close", ")", "sep"),
        PPart("gap", " ", "sep"),
        PPart("frame", "@icrs"),
    )


def test_nesting_keeps_inner_groups_inline() -> None:
    """A wadler-lindig group is all-or-nothing.

    Splicing children flat means every break point breaks together, so the
    inner ``*`` separators break for no reason. This count is what fails if
    anyone re-flattens the tree.
    """
    out = doc_to_str(parts_to_doc(_vector_parts()), 46)
    assert "\n" in out  # the outer group did break
    assert out.count(" *\n") == 0  # ...but no inner one did


def test_nested_latex_has_exactly_one_dollar_pair() -> None:
    """Embedding rendered strings would give nested ``$...$`` and invalid LaTeX."""
    out = parts_to_markup(_vector_parts("latex"), markup="latex")
    assert out.startswith("$")
    assert out.endswith("$")
    assert "$" not in out[1:-1]


class _Uncertain:
    """A type introducing roles no markup has heard of."""

    def __init__(self, value: float, err: float, unit: str) -> None:
        self.value, self.err, self.unit = value, err, unit


@pparts.dispatch  # type: ignore[misc]
def _(obj: _Uncertain, /, *, markup: str = "text", **kw: object) -> tuple:
    return (
        PPart("value", f"{obj.value:g}"),
        PPart("pm", " ± ", "sep"),
        PPart("uncert", f"{obj.err:g}"),
        PPart("mul", " * ", "sep"),
        PPart("unit", obj.unit),
    )


def test_new_roles_work_with_no_markup_change() -> None:
    """``pm``/``uncert`` are unknown to text and html; each falls back to its text."""
    obj = _Uncertain(1.5, 0.2, "m")
    assert parts_to_markup(pparts(obj)) == "1.5 ± 0.2 * m"
    assert parts_to_markup(pparts(obj, markup="html"), markup="html") == (
        "<span>1.5</span> ± <span>0.2</span> * <span>m</span>"
    )


def test_a_markup_may_override_a_new_role() -> None:
    """LaTeX declares ``pm``; that is the whole cost of teaching it a role."""
    out = parts_to_markup(
        pparts(_Uncertain(1.5, 0.2, "m"), markup="latex"), markup="latex"
    )
    assert r"\pm" in out


def test_unregistered_type_degrades_rather_than_raising() -> None:
    """One unregistered field must not poison a whole object's repr."""
    assert (
        parts_to_markup(
            pparts(object.__new__(type("W", (), {"__str__": lambda _: "W!"})))
        )
        == "W!"
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (r"$\mathrm{m}$", r"\mathrm{m}"),  # wrapped: unwrapped once
        (r"\mathrm{m}", r"\mathrm{m}"),  # unwrapped: left intact
        ("$", "$"),  # lone delimiter: satisfies both ends
        ("", ""),
        ("$a", "$a"),
        ("a$", "a$"),
    ],
)
def test_unwrap_math_only_strips_real_delimiters(text: str, expected: str) -> None:
    r"""Slicing unconditionally corrupts an already-unwrapped fragment.

    ``\mathrm{m}`` would become ``mathrm{m`` -- the same defect fixed in
    ``_repr_latex_`` (#870), which this engine must not reintroduce.
    """
    assert unwrap_math(text) == expected


_PACKAGE_SIDE = ("engine", "generic", "examples")


def _import_offenders(source: str) -> list[str]:
    """Imports in `source` that a package-side module may not make.

    Allowed: `wadler_lindig`, `plum`, the standard library, and a
    single-dot relative import of another package-side module.
    """
    import ast  # noqa: PLC0415
    import sys  # noqa: PLC0415

    allowed = {"wadler_lindig", "plum", *sys.stdlib_module_names}
    bad = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            bad += [a.name for a in node.names if a.name.split(".")[0] not in allowed]
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if (node.module or "").split(".")[0] not in allowed:
                    bad.append(node.module or "")
            elif node.level != 1:
                bad.append("." * node.level + (node.module or ""))
            elif node.module is None:  # ``from . import x``
                bad += [f".{a.name}" for a in node.names if a.name not in _PACKAGE_SIDE]
            elif node.module not in _PACKAGE_SIDE:
                bad.append(f".{node.module}")
    return bad


def test_import_offenders_flags_what_it_should() -> None:
    """The boundary check is not vacuous."""
    assert _import_offenders("from .axes import foo") == [".axes"]
    assert _import_offenders("from .. import quantity") == [".."]
    assert _import_offenders("from . import axes") == [".axes"]
    assert _import_offenders("import numpy") == ["numpy"]
    assert _import_offenders("from jax import numpy") == ["jax"]
    assert not _import_offenders(
        "import os\nimport plum\nfrom .engine import x\nfrom . import generic\n"
        "from wadler_lindig import y"
    )


def test_the_example_type_renders_in_every_route() -> None:
    """The engine's own docs are written against this type.

    So it has to exercise the routes those docs demonstrate. Values here are
    measured, not assumed: `mul` leaves the object's own ``.`` separator
    alone, which is what the ``sep`` axis documents.
    """
    v = examples.Version(2, 1)
    assert repr(v) == "2.1"
    assert str(v) == "2.1"
    assert f"{v:bare}" == "2.1"
    assert f"{v:mul}" == "2.1"
    assert f"{v:html}" == "<span>2</span>.<span>1</span>"
    assert f"{v:latex}" == "$2.1$"


def test_the_engine_alone_registers_six_axes_and_no_aliases(tmp_path) -> None:
    """Assembled as a package of its own, the engine's registry is its own.

    `engine` and `generic` are copied into a bare package and imported in a
    subprocess: importing them from `unxt._src.fmt` would run that package's
    ``__init__``, which imports `axes` eagerly, so the registry would already
    hold unxt's two axes and three aliases.

    This is the property the extracted package's suite will depend on, and
    proving it here means the extraction cannot be surprised by it.
    """
    import json  # noqa: PLC0415
    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415

    pkg = tmp_path / "wlp"
    pkg.mkdir()
    (pkg / "__init__.py").touch()
    src = pathlib.Path(engine_module.__file__).parent
    for mod in _PACKAGE_SIDE:
        (pkg / f"{mod}.py").write_text(
            (src / f"{mod}.py").read_text(encoding="utf-8"), encoding="utf-8"
        )

    code = (
        "import json, wlp.engine as e, wlp.generic;"
        "print(json.dumps([sorted(e.AXES), len(e.ALIASES), sorted(e.MARKUPS)]))"
    )
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        cwd=tmp_path,
    ).stdout
    axes, n_aliases, markups = json.loads(out)
    assert axes == ["indent", "layout", "markup", "sep", "value", "width"]
    assert n_aliases == 0
    assert markups == ["html", "latex", "text"]


def test_the_package_side_exposes_no_privates_across_the_seam() -> None:
    """Unxt's layer must reach the engine through public names only.

    After extraction these are separate distributions, where an underscore
    name can change in a patch release without warning.
    """
    import ast  # noqa: PLC0415

    source = (
        pathlib.Path(engine_module.__file__)
        .with_name("axes.py")
        .read_text(encoding="utf-8")
    )
    imported = [
        alias.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom)
        and node.level == 1
        and node.module in _PACKAGE_SIDE
        for alias in node.names
    ]
    assert imported, "the test is vacuous if axes.py imports nothing from the seam"
    assert [n for n in imported if n.startswith("_")] == []


def test_package_side_imports_only_wl_plum_and_stdlib() -> None:
    """engine.py and generic.py are the future `pparts` package.

    They may import `wadler_lindig`, `plum`, the standard library, and each
    other (relative imports). Anything domain-specific belongs in `axes.py`.
    """
    for mod in _PACKAGE_SIDE:
        path = pathlib.Path(engine_module.__file__).with_name(f"{mod}.py")
        assert _import_offenders(path.read_text(encoding="utf-8")) == [], mod


def test_a_downstream_package_can_register_its_own_axis() -> None:
    """A downstream axis must be indistinguishable from a built-in one.

    This is the check the whole registry exists for: before it, `Spec` was a
    closed `NamedTuple` and this raised ``TypeError: Spec.__new__() got an
    unexpected keyword argument``. `coordinax` already carries exactly such an
    axis (``vector_form``) as an ad-hoc ``__pdoc__`` knob with no way to reach
    it from a format spec.
    """
    axis = Axis(
        name="demo_form",
        keywords={"demoform": True},
        default=False,
        layouts={"call": lambda v: {"demo_form": v}},
    )
    register_axis(axis)
    register_alias("demoalias", "call-demoform")
    try:
        assert parse_spec("call-demoform")["demo_form"] is True
        assert parse_spec("call")["demo_form"] is False  # default filled in
        assert parse_spec("demoalias") == parse_spec("call-demoform")
        # Order-independent alongside a built-in axis, like any other keyword.
        assert parse_spec("call-demoform-abbrev") == parse_spec("abbrev-demoform-call")
        # Scoped to its layouts, with the same typed error a built-in gets.
        with pytest.raises(ValueError, match="'demo_form' does not apply"):
            parse_spec("product-demoform")
    finally:
        unregister_axis("demo_form")
        unregister_alias("demoalias")


@pytest.mark.parametrize("word", ["html", "compact"])
def test_an_alias_never_collides_silently(word: str) -> None:
    """An alias has no qualified form, so a clash with one stays fatal.

    A keyword can be disambiguated as ``axis=word``; an alias is a whole spec
    and cannot, so both directions around an alias must still raise.
    """
    with pytest.raises(ValueError, match="already"):
        register_alias(word, "call-abbrev")


def test_an_alias_name_may_not_be_claimed_as_a_keyword() -> None:
    with pytest.raises(ValueError, match="already an alias"):
        register_axis(
            Axis(
                name="clash", keywords={"compact": 1}, default=0, layouts={"call": dict}
            )
        )


def test_only_one_axis_may_claim_free_text() -> None:
    """A spec has exactly one trailing run, so only one axis can receive it.

    Not a restriction so much as an observation about the scan rule: the run
    after the first non-keyword is terminal. Registering a second claimant is
    a mistake worth catching at registration rather than at parse.
    """
    with pytest.raises(ValueError, match="already does"):
        register_axis(
            Axis(
                name="second_free",
                keywords={"secondfree": 1},
                default=0,
                layouts={"product": dict},
                free_text=("product",),
            )
        )
    assert "second_free" not in AXES


def test_registering_an_axis_name_twice_is_an_error() -> None:
    """A second axis of the same name would shadow the first in `AXES`."""
    with pytest.raises(ValueError, match="already registered"):
        register_axis(
            Axis(name="value", keywords={"nope": 1}, default=0, layouts={"call": dict})
        )


def test_spec_of_rejects_an_unregistered_axis() -> None:
    """A typo'd axis would otherwise sit in the mapping and never be read."""
    with pytest.raises(ValueError, match="not registered axes"):
        Spec.of(markup="html", nonsuch="x")


def test_spec_repr_shows_every_axis() -> None:
    """`Spec` is a mapping, so it needs a repr of its own to be legible."""
    text = repr(Spec.of(layout="call"))
    assert text.startswith("Spec(")
    assert "layout='call'" in text


def test_a_probe_that_raises_is_skipped_not_propagated() -> None:
    """The inert check must never break the render it is only inspecting.

    A type may legitimately reject the *default* of an axis while accepting
    the value asked for. The probe swallows that and reports nothing for the
    axis, rather than turning a working format call into an exception.
    """

    class Picky:
        pass

    @pparts.dispatch
    def _(obj: Picky, /, *, markup: str = "text", unit_style: str = "symbol", **kw):
        if unit_style == "symbol":  # the default -- what the probe re-renders with
            msg = "no symbol form"
            raise RuntimeError(msg)
        return (PPart("unit", unit_style),)

    obj = Picky()
    spec = parse_spec("name")
    assert pspec(obj, "name") == "name"
    assert inert_axes(obj, spec, render(obj, spec)) == []


def test_a_render_error_without_free_text_keeps_its_own_exception() -> None:
    """Only a *value spec* failure is reworded as a bad format spec.

    A keyword-only spec that fails is the type's own error, and rewording it
    as "invalid format spec" would blame the spec for something it did not do.
    """

    class Exploding:
        pass

    @pparts.dispatch
    def _(obj: Exploding, /, *, markup: str = "text", **kw):
        msg = "boom from pparts"
        raise ValueError(msg)

    with pytest.raises(ValueError, match="boom from pparts"):
        pspec(Exploding(), "mul")


def test_sep_fragment_is_escaped_unless_overridden():
    """A downstream `sep` is escaped; a markup-table override is trusted."""
    parts = (PPart("a", "x"), PPart("lt", "<", "sep"), PPart("b", "y"))
    assert "&lt;" in parts_to_markup(parts, markup="html")
    assert r" \, " in parts_to_markup(
        (PPart("a", "x"), PPart("bare", " ", "sep"), PPart("b", "y")), markup="latex"
    )


@pytest.mark.parametrize("role", ["escape", "wrap", "vsep", "_content"])
def test_role_named_like_a_config_key_is_not_an_override(role):
    """`MARKUPS` rows mix config and per-role overrides in one namespace."""
    parts = (PPart(role, "x"), PPart(role, "-", "sep"), PPart(role, "y"))
    assert parts_to_markup(parts, markup="html") == "<span>x</span>-<span>y</span>"


def test_default_pvalue_needs_no_numpy():

    assert doc_to_str(pvalue("abc")) == "'abc'"
    assert doc_to_str(pvalue(3.14159, value_spec=".2f")) == "3.14"


def test_unregistered_type_degrades_to_str():
    """A type with no `pparts` renders as its `str`, so display never raises."""

    class Opaque:
        def __str__(self) -> str:
            return "opaque"

    assert pparts(Opaque()) == (PPart("value", "opaque"),)
    assert pspec(Opaque(), "html") == "<span>opaque</span>"


def test_unregistered_type_rejects_a_value_spec():
    """No elements to format: say so rather than answer a different request."""
    with pytest.raises(TypeError, match="does not support a value format spec"):
        pspec(object(), ".2f")


def test_ppart_doc_defaults_to_none_and_keeps_text():
    assert PPart("value", "1.0").doc is None


def test_doc_part_keeps_flat_text_and_the_doc():
    doc = wl.bracketed(
        begin=wl.TextDoc("["),
        docs=[wl.TextDoc("1."), wl.TextDoc("2.")],
        sep=wl.comma,
        end=wl.TextDoc("]"),
        indent=1,
    )
    part = doc_part("value", doc)
    assert part.text == "[1., 2.]"
    assert part.doc is doc


def test_text_layout_uses_the_part_doc_so_width_reaches_it():
    doc = wl.bracketed(
        begin=wl.TextDoc("["),
        docs=[wl.TextDoc(str(i)) for i in range(8)],
        sep=wl.comma,
        end=wl.TextDoc("]"),
        indent=1,
    )
    parts = (doc_part("value", doc), PPart("mul", " * ", "sep"), PPart("unit", "m"))
    assert doc_to_str(parts_to_doc(parts), 88) == "[0, 1, 2, 3, 4, 5, 6, 7] * m"
    narrow = doc_to_str(parts_to_doc(parts), 10)
    assert "\n" in narrow
    assert max(map(len, narrow.splitlines())) <= 10


def test_registries_are_read_only_views():
    for reg in (AXES, ALIASES, MARKUPS):
        with pytest.raises(TypeError):
            reg["x"] = 1  # type: ignore[index]


def test_register_axis_rejects_a_duplicate_unless_replace():
    ax = Axis("tmp_axis", {"tmpword": 1}, 0, {"call": lambda _: {}})
    register_axis(ax)
    try:
        with pytest.raises(ValueError, match="already registered"):
            register_axis(ax)
        register_axis(ax, replace=True)
        assert "tmp_axis" in AXES
    finally:
        unregister_axis("tmp_axis")
    assert "tmp_axis" not in AXES
    assert "tmpword" not in _KEYWORDS


def test_register_alias_replace():
    register_alias("tmp_alias", "bare")
    try:
        with pytest.raises(ValueError, match="already registered"):
            register_alias("tmp_alias", "mul")
        register_alias("tmp_alias", "mul", replace=True)
        assert ALIASES["tmp_alias"] == "mul"
    finally:
        unregister_alias("tmp_alias")
    assert "tmp_alias" not in ALIASES


_GOOD_ROW = {"_content": "[{}]", "wrap": "{}", "vsep": ", ", "escape": None}


@pytest.mark.parametrize(
    ("bad", "match"),
    [
        ({"wrap": "no braces"}, "wrap"),
        ({"wrap": "{}{}"}, "wrap"),
        ({"wrap": 3}, "wrap"),
        ({"_content": "{}{}"}, "_content"),
        ({"_content": "{x}"}, "_content"),
        ({"_content": "{"}, "_content"),
        ({"_content": None}, "_content"),
    ],
)
def test_register_markup_rejects_a_malformed_template(bad, match):
    with pytest.raises(ValueError, match=match):
        register_markup("bad_markup", {**_GOOD_ROW, **bad})
    assert "bad_markup" not in MARKUPS
    assert "bad_markup" not in _KEYWORDS


@pytest.mark.parametrize("content", ["", "<{0}>", "{}", "plain"])
def test_register_markup_accepts_any_content_str_that_formats(content):
    register_markup("ok_markup", {**_GOOD_ROW, "_content": content})
    unregister_markup("ok_markup")


@pytest.mark.parametrize("markup", sorted(MARKUPS))
def test_builtin_markup_rows_satisfy_the_register_markup_checks(markup):
    _check_markup_row(MARKUPS[markup])


def _snapshot():
    return (
        dict(_AXES),
        {k: list(v) for k, v in _KEYWORDS.items()},
        dict(_ALIASES),
        {k: dict(v) for k, v in _MARKUPS.items()},
        dict(_MARKUP_WORDS),
    )


def test_a_failed_replace_leaves_every_registry_untouched():
    before = _snapshot()
    with pytest.raises(ValueError, match="claims free text"):
        register_axis(AXES["sep"]._replace(free_text=("product",)), replace=True)
    assert _snapshot() == before
    assert parse_spec("bare")["sep"] == "bare"


def test_a_keyword_clash_on_replace_leaves_the_old_axis():
    before = _snapshot()
    clash = AXES["sep"]._replace(keywords={"compact": 1})  # an alias
    with pytest.raises(ValueError, match="already an alias"):
        register_axis(clash, replace=True)
    assert _snapshot() == before


def test_a_successful_replace_keeps_keywords_consistent():
    old = AXES["sep"]
    new = old._replace(keywords={"mul": "mul", "bare": "bare", "tmpsep": "mul"})
    register_axis(new, replace=True)
    try:
        assert _KEYWORDS["tmpsep"] == ["sep"]
        assert _KEYWORDS["mul"].count("sep") == 1
    finally:
        register_axis(old, replace=True)
    assert "tmpsep" not in _KEYWORDS
    assert _KEYWORDS["bare"].count("sep") == 1


def test_register_markup_rejects_a_keyword_clash():
    with pytest.raises(ValueError, match="keyword or alias"):
        register_markup("mul", _GOOD_ROW)
    assert "mul" not in MARKUPS


def test_width_does_not_eat_a_format_spec() -> None:
    assert parse_spec("mul-width=40-.2f")["value"] == ".2f"
    assert parse_spec(">10.2f")["value"] == ">10.2f"


def test_width_and_indent_are_not_bare_words() -> None:
    spec = parse_spec("mul")
    assert spec["width"] is None
    assert spec["indent"] is None
    assert "width" not in _KEYWORDS
    assert "40" not in _KEYWORDS
    assert parse_spec("width=40")["width"] == 40


def test_count_mapping_claims_no_bare_word():
    """`width`/`indent` take digits only through `axis=<n>`, never as a bare word."""
    from unxt._src.fmt.engine import _Count  # noqa: PLC0415

    count = _Count()
    assert (len(count), list(count), count["12"]) == (0, [], 12)
    with pytest.raises(KeyError):
        count["m"]


@pytest.mark.parametrize("markup", ["text", "html", "latex"])
def test_width_reaches_every_markup(markup):
    """A narrow width breaks in every markup; wrappers no longer inflate it.

    Built from fragments rather than a `Quantity` on purpose: this is a claim
    about the layout, and routing it through a real value would make it depend
    on whether that value rendered in full -- a traced array comes out as a
    shape summary, which has nothing to wrap.
    """
    parts = (
        PPart("value", "x" * 30),
        PPart("mul", " * ", "sep"),
        PPart("unit", "m"),
    )
    doc = parts_to_doc(parts, markup=markup)
    assert "\n" in doc_to_str(doc, 20)
    # The flat path is what `_repr_html_` / `_repr_latex_` use, and it is still
    # one line -- markup gained layout without its string changing.
    assert "\n" not in doc_to_str(doc, 10**9)
    assert "\n" not in parts_to_markup(parts, markup=markup)


def test_register_markup_rejects_a_non_callable_escape():
    with pytest.raises(ValueError, match="escape"):
        register_markup("bad_escape", {**_GOOD_ROW, "escape": "x"})
    assert "bad_escape" not in MARKUPS


@pytest.mark.parametrize("content", ["{0.foo}", "{0[a]}"])
def test_register_markup_rejects_attribute_or_index_content(content):
    with pytest.raises(ValueError, match="_content"):
        register_markup("bad_content", {**_GOOD_ROW, "_content": content})
    assert "bad_content" not in MARKUPS


@pytest.mark.parametrize("short_arrays", [False, True])
def test_default_pvalue_non_compact_goes_through_wl_pdoc(short_arrays):
    """The numpy-free default also serves the ``array``/``type`` forms."""
    assert doc_to_str(pvalue("a<b", short_arrays=short_arrays)) == "'a<b'"
    assert (
        doc_to_str(pvalue("a<b", short_arrays=short_arrays, markup="html"))
        == "&#x27;a&lt;b&#x27;"
    )


def test_this_suite_reaches_only_for_the_package_side() -> None:
    """These tests travel to `wadler-lindig-parts`, so they may not need unxt.

    In-tree they import the engine from `unxt._src.fmt`, which becomes
    `wadler_lindig_parts` at the move. Anything else from `unxt` -- the
    top-level package, the `_pparts` shim, or `axes` -- would not survive it.

    This is the check that a "does the body mention `u.Q`" heuristic missed:
    three tests reached for `u.StaticQuantity`, the shim, and `axes` through
    import lines, and were only caught when the files were actually copied.
    """
    import ast  # noqa: PLC0415

    allowed = {
        "unxt._src.fmt",
        "unxt._src.fmt.engine",
        "unxt._src.fmt.generic",
        "unxt._src.fmt.examples",
    }
    source = pathlib.Path(__file__).read_text(encoding="utf-8")
    offenders = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            offenders += [a.name for a in node.names if a.name.split(".")[0] == "unxt"]
        elif (
            isinstance(node, ast.ImportFrom)
            and (node.module or "").startswith("unxt")
            and node.module not in allowed
        ):
            offenders.append(node.module)
    assert offenders == []
