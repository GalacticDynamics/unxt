"""Tests for the `unxt._pparts` string-formatting engine."""

import pathlib
import re
import warnings

import jax
import numpy as np
import pytest
import wadler_lindig as wl

import unxt as u
from unxt._src.fmt import (
    ALIASES,
    AXES,
    MARKUPS,
    REQUIRED_MARKUP_KEYS,
    Axis,
    PGroup,
    PPart,
    Spec,
    doc_to_str,
    engine as engine_module,
    inert_axes,
    parse_spec,
    parts_to_doc,
    parts_to_markup,
    pparts,
    pspec,
    register_alias,
    register_axis,
    render,
    unwrap_math,
)
from unxt._src.fmt.engine import _KEYWORDS

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


@pytest.mark.parametrize("alias", sorted(ALIASES))
def test_every_alias_expands_to_a_valid_core_spec(alias: str) -> None:
    """An alias is sugar, never new meaning: it must parse as its expansion."""
    assert parse_spec(alias) == parse_spec(ALIASES[alias])


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


# ============================================================================
# Parsing: the scan rule


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


def test_a_keyword_after_the_value_spec_is_not_a_keyword() -> None:
    """The value spec is last: nothing after it is scanned for keywords.

    ``"mul-name-.2f"`` is the way to say it. ``"mul-.2f-name"`` stops scanning
    at ``.2f``, so ``name`` is swallowed into the value spec -- which then
    fails as the malformed spec it is, naming the vocabulary it missed.
    """
    q = u.Q([1.234, 2.345], "m")
    assert pspec(q, "mul-name-.2f") == "[1.23, 2.35] * meter"

    assert parse_spec("mul-.2f-name")["value"] == ".2f-name"
    with pytest.raises(ValueError, match="not a valid Python format spec"):
        pspec(q, "mul-.2f-name")


def test_a_leftover_run_is_one_spec_not_a_silent_rejoin() -> None:
    """Regression: ``mul-.2f-.3g`` used to be *parsed by elimination*.

    The old parser pulled keywords out and rejoined whatever was left, so this
    silently became the value spec ``".2f-.3g"``. Under the scan rule it is
    still one value spec -- but honestly so, by position -- and Python rejects
    it as the single malformed spec it is.
    """
    assert parse_spec("mul-.2f-.3g")["value"] == ".2f-.3g"
    with pytest.raises(ValueError, match="Invalid format specifier"):
        pspec(u.Q([1.0], "m"), "mul-.2f-.3g")


# ============================================================================
# Rendering each axis


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        # layout
        ("product", "[1., 2., 3.] m"),
        ("call", "Quantity([1., 2., 3.], unit='m')"),
        # value
        ("array", "Array([1., 2., 3.], dtype=float32) m"),
        ("values", "[1., 2., 3.] m"),
        ("type", "f32[3] m"),
        # separator
        ("mul", "[1., 2., 3.] * m"),
        ("bare", "[1., 2., 3.] m"),
        # markup
        ("html", "<span>[1., 2., 3.]</span> <span>m</span>"),
        ("html-mul", "<span>[1., 2., 3.]</span> * <span>m</span>"),
        ("latex", r"$[1.,~2.,~3.] \mathrm{m}$"),
        # unit
        ("symbol", "[1., 2., 3.] m"),
        ("name", "[1., 2., 3.] meter"),
        ("dim", "[1., 2., 3.] length"),
        # aliases
        ("compact", "Q([1., 2., 3.], unit='m')"),
        ("full", "Quantity(Array([1., 2., 3.], dtype=float32), unit='m')"),
        # combinations
        ("html-type-mul", "<span>f32[3]</span> * <span>m</span>"),
        ("latex-mul-name", r"$[1.,~2.,~3.] \; meter$"),
        ("call-type", "Quantity(f32[3], unit='m')"),
    ],
)
def test_each_spec_renders_its_documented_string(spec: str, expected: str) -> None:
    assert pspec(u.Q([1.0, 2, 3], "m"), spec) == expected


def test_default_separator_is_bare() -> None:
    """``f"{q:.2f}"`` must stay astropy-shaped: a space, no ``*``."""
    assert pspec(u.Q(3.14159, "m"), ".2f") == "3.14 m"


# ============================================================================
# One value-rendering path (the old two-implementation split)


def test_a_value_spec_works_on_an_array() -> None:
    """Regression: a bare value spec used to be scalar-only.

    ``.2f`` went through a second implementation that called
    ``format(obj.value, spec)`` directly, which `numpy` rejects for a non-0-d
    array. There is now one value path, so an array formats like a scalar.
    """
    assert pspec(u.Q([1.234, 2.345], "m"), ".2f") == "[1.23, 2.35] m"


def test_a_value_spec_differs_from_a_keyworded_one_only_by_separator() -> None:
    """Regression: ``.2f`` and ``mul-.2f`` used to reach different code.

    One was the astropy-compatible fallback and the other the engine, so they
    differed in array support *and* separator. Now the only difference is the
    separator the spec actually names.
    """
    q = u.Q([1.234, 2.345], "m")
    assert pspec(q, ".2f") == "[1.23, 2.35] m"
    assert pspec(q, "mul-.2f") == "[1.23, 2.35] * m"
    assert pspec(q, "bare-.2f") == pspec(q, ".2f")


def test_dimensionless_drops_the_unit_under_a_value_spec() -> None:
    """The astropy-compatible behaviour, now on the single path."""
    assert pspec(u.Q(3.14159, ""), ".2f") == "3.14"


# ============================================================================
# Errors: every rejection is a typed one


def test_unknown_spec_names_the_type_and_the_grammar() -> None:
    with pytest.raises(ValueError, match=r"invalid format spec 'nonsense'"):
        pspec(u.Q(3.14, "m"), "nonsense")


@pytest.mark.parametrize(
    "spec", ["mul-bare", "html-latex", "type-values", "call-product"]
)
def test_setting_one_axis_twice_is_an_error(spec: str) -> None:
    with pytest.raises(ValueError, match="is set twice"):
        pspec(u.Q(1.0, "m"), spec)


@pytest.mark.parametrize(
    ("spec", "axis"),
    [
        ("call-mul", "sep"),
        ("call-bare", "sep"),
        ("call-html", "markup"),
        ("call-latex", "markup"),
        ("product-abbrev", "abbrev"),
    ],
)
def test_an_axis_the_layout_lacks_is_an_error(spec: str, axis: str) -> None:
    """Regression: the two systems used to fail incoherently.

    ``html-compact`` leaked past the DSL into ``float.__format__`` and raised
    ``Invalid format specifier 'compact' for object of type 'float'``. Naming
    an axis a layout has no concept of now says exactly that instead -- and
    silently ignoring it was never an option, since it would hide the mistake.
    """
    with pytest.raises(ValueError, match=f"{axis!r} does not apply"):
        pspec(u.Q(1.0, "m"), spec)


def test_alias_plus_a_conflicting_keyword_reports_the_expansion_error() -> None:
    """An alias is textual, so it fails exactly as its expansion would."""
    with pytest.raises(ValueError, match="'markup' does not apply"):
        pspec(u.Q(1.0, "m"), "html-compact")


@pytest.mark.parametrize("spec", ["type-.2f", "array-.2f"])
def test_a_keyword_and_free_text_for_one_axis_is_set_twice(spec: str) -> None:
    """Folding the format spec onto the value axis makes this the ordinary rule.

    A shape/dtype summary has no elements a per-element spec could format, so
    the combination is meaningless. It used to need a bespoke consistency
    check between `value` and a separate `value_spec` key; now the axis simply
    cannot be set twice.
    """
    with pytest.raises(ValueError, match="'value' is set twice"):
        pspec(u.Q([1.0, 2], "m"), spec)


def test_a_value_spec_does_not_apply_to_call_layout() -> None:
    with pytest.raises(ValueError, match="free text does not apply to 'call'"):
        pspec(u.Q([1.0, 2], "m"), "call-.2f")


def test_a_type_with_no_pparts_rejects_a_value_spec() -> None:
    """Degrading is right for *display*; silently dropping a request is not.

    A unit system registers no `pparts`, so it has no elements to format. It
    must say so rather than quietly render something else.
    """
    with pytest.raises(TypeError, match="does not support a value format spec"):
        pspec(u.unitsystem("m", "s", "kg", "rad"), ".2f")


def test_unknown_markup_is_named() -> None:
    with pytest.raises(ValueError, match=r"unknown markup 'markdown'"):
        parts_to_markup(pparts(u.Q(1.0, "m")), markup="markdown")


@pytest.mark.parametrize("word", sorted({*_KEYWORDS, *ALIASES}))
def test_grammar_words_are_not_valid_value_specs(word: str) -> None:
    """Claiming these words is strictly additive.

    If any were also a legal Python format spec, the scan rule would silently
    prefer the keyword reading and steal a meaning users already had.
    """
    for value in (3.14, 3, complex(1, 2), np.float32(1.5)):
        with pytest.raises((ValueError, TypeError)):
            format(value, word)


# ============================================================================
# repr / str are the same renderer, reached with a different Spec


def test_repr_and_str_are_call_layout_specs() -> None:
    q = u.Q([1.0, 2, 3], "m")
    assert repr(q) == render(q, Spec.of(layout="call", value="array"), named_unit=True)
    assert str(q) == render(q, Spec.of(layout="call", value="values"), named_unit=True)


def test_the_empty_spec_is_str() -> None:
    """Not a special case any more -- a defined spec that happens to be `str`."""
    q = u.Q([1.0, 2, 3], "m")
    assert f"{q}" == str(q)


def test_unit_system_repr_still_round_trips_through_eval() -> None:
    """``call`` layout routes through ``__pdoc__``, which is what reconstructs.

    Routing ``repr`` anywhere else would break the round-trip; this pins the
    reason the layout axis exists at all.
    """
    usys = u.unitsystem("m", "s", "kg", "rad")
    assert eval(repr(usys), {"unitsystem": u.unitsystem}) == usys  # noqa: S307


# ============================================================================
# Markup table


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


def test_rendered_markup_is_not_escaped_again() -> None:
    r"""Escaping unxt's own LaTeX would turn ``\mathrm`` into ``\textbackslash``."""
    assert pspec(u.Q(1.0, "m"), "latex-mul") == r"$1. \; \mathrm{m}$"


# ============================================================================
# Layout: the wadler-lindig feed


def _long_quantity_doc() -> wl.AbstractDoc:
    return parts_to_doc(pparts(u.Q(np.arange(8.0), "m")))


def test_layout_stays_inline_when_it_fits() -> None:
    assert "\n" not in doc_to_str(_long_quantity_doc(), 100)


def test_separator_ink_survives_a_break() -> None:
    """``BreakDoc`` shows its text only in horizontal mode.

    Mapping ``" * "`` straight to a ``BreakDoc`` silently dropped the ``*`` on
    the broken line.
    """
    out = doc_to_str(_long_quantity_doc(), 20)
    assert "\n" in out
    assert out.rstrip().endswith("m")
    assert "*" in out


def test_adjacent_separators_do_not_emit_a_blank_line() -> None:
    """A separator with no trailing space must not offer a break."""
    parts = (PPart("close", ")", "sep"), PPart("gap", " ", "sep"), PPart("frame", "@x"))
    assert not re.search(r"\n\s*\n", doc_to_str(parts_to_doc(parts), 3))


def _vector_parts(markup: str = "text") -> tuple:
    x = u.Q(np.arange(6.0), "m")
    y = u.Q(np.arange(6.0), "s")
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


# ============================================================================
# Extensibility


class _Uncertain:
    """A type introducing roles no markup has heard of."""

    def __init__(self, value: float, err: float, unit: str) -> None:
        self.value, self.err, self.unit = value, err, u.unit(unit)


@pparts.dispatch  # type: ignore[misc]
def _(obj: _Uncertain, /, *, markup: str = "text", **kw: object) -> tuple:
    return (
        PPart("value", f"{obj.value:g}"),
        PPart("pm", " ± ", "sep"),
        PPart("uncert", f"{obj.err:g}"),
        PPart("mul", " * ", "sep"),
        *pparts(obj.unit, markup=markup),
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


# ============================================================================
# jit


def test_preset_beats_the_value_spec_under_jit() -> None:
    """The preset lookup must precede the value-spec branch.

    Handing a non-empty spec straight to a tracer raises, so a preset checked
    second would be unreachable under `jax.jit`.
    """
    seen: list[str] = []

    @jax.jit
    def f(q: u.Q) -> u.Q:
        seen.append(pspec(q, "mul"))
        return q

    f(u.Q([1.0, 2, 3], "m"))
    assert seen == ["f32[3] * m"]


def test_short_arrays_marks_a_weak_dtype() -> None:
    """A weakly-typed scalar keeps the ``weak_`` prefix in the summary."""
    assert pspec(u.Q(1.0, "m"), "type-mul") == "weak_f32[] * m"


def test_short_arrays_unwraps_a_static_value() -> None:
    """``StaticValue`` is not an array, so the kind hook declines it.

    The wrapper is then suppressed and the inner NumPy array renders without
    the ``(numpy)`` kind suffix.
    """
    assert pspec(u.StaticQuantity([1.0, 2.0], "m"), "type-mul") == "f64[2] * m"


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


# ============================================================================
# The seam: the engine must stay liftable, and downstream must be a peer


def test_engine_imports_nothing_domain_specific() -> None:
    """The engine is meant to lift out into a package of its own.

    Its whole claim is that it knows nothing about quantities, units, arrays
    or `jax` -- a claim an earlier revision made in prose while `import jax`
    sat at the top of the file. Pin it to the import list so it cannot rot
    back: anything domain-specific belongs in `unxt._src.fmt.axes`, which is a
    peer of `coordinax`'s and `galax`'s future layers, not a privileged one.
    """
    source = (
        pathlib.Path(engine_module.__file__).read_text(encoding="utf-8").splitlines()
    )
    imports = [ln for ln in source if re.match(r"^\s*(import|from)\s", ln)]
    banned = ("unxt", "jax", "numpy", "astropy", "quax")
    offenders = [ln for ln in imports if any(f"{b}" in ln for b in banned)]
    assert not offenders, offenders


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
        AXES.pop("demo_form")
        _KEYWORDS.pop("demoform")
        ALIASES.pop("demoalias")


@pytest.mark.parametrize("word", ["html", "compact"])
def test_an_alias_never_collides_silently(word: str) -> None:
    """An alias has no qualified form, so a clash with one stays fatal.

    A keyword can be disambiguated as ``axis:word``; an alias is a whole spec
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


# ============================================================================
# Qualification: `axis:word` when two packages want one word


def test_a_qualified_keyword_resolves_without_ambiguity() -> None:
    """``axis:word`` is always available, collision or not.

    A downstream package can write the qualified form from the start and be
    immune to a word it does not yet share.
    """
    q = u.Q([1.0, 2, 3], "m")
    assert pspec(q, "unit:dim") == pspec(q, "dim")
    assert pspec(q, "markup:latex-sep:mul") == pspec(q, "latex-mul")


def test_two_axes_may_claim_one_word_and_both_stay_reachable() -> None:
    """The collision this exists for: one word, two unrelated meanings.

    ``dim`` is a unit spelling here and could as reasonably be a manifold's
    dimensionality in `coordinax`. Registration no longer refuses the second
    claimant -- which used to make ``import coordinax; import galax`` explode
    in user code, with neither library at fault and no fix open to either.
    """
    q = u.Q([1.0, 2, 3], "m")
    register_axis(
        Axis(
            name="manifold",
            keywords={"dim": 3},
            default=0,
            layouts={"product": lambda v: {"manifold": v}},
        )
    )
    try:
        # Bare is now ambiguous, and says so, naming both ways out.
        with pytest.raises(ValueError, match="ambiguous keyword 'dim'"):
            pspec(q, "dim")
        # Either qualified form still works, and unxt's is unchanged.
        assert pspec(q, "unit:dim") == "[1., 2., 3.] length"
        # Surgical: only the colliding token needs the prefix.
        assert pspec(q, "latex-mul-unit:dim") == pspec(q, "latex-mul-unit:dim")
        # Every other word is untouched by the collision.
        assert pspec(q, "mul") == "[1., 2., 3.] * m"
    finally:
        AXES.pop("manifold")
        _KEYWORDS["dim"].remove("manifold")


def test_an_unknown_qualifier_is_free_text_not_an_error() -> None:
    """A ``:`` in a value spec must not be read as a qualifier.

    ``:>10`` is a fill character and an alignment. It resolves to no axis, so
    it falls through to the value spec exactly like any other non-keyword --
    which is what keeps qualification from colonising the format-spec syntax.
    """
    assert parse_spec(":>6")["value"] == ":>6"
    assert pspec(u.Q(3, "m"), ":>6") == ":::::3 m"


# ============================================================================
# Inert axes: an opt-in check that a spec actually did something


def test_inert_axes_finds_an_axis_that_changed_nothing() -> None:
    """Asked by experiment: re-render with the axis defaulted and compare.

    That needs no cooperation from the type and sees through composites for
    free -- if a *child* consumed the axis the output differs, which is the
    answer wanted. Type-level applicability cannot be declared statically,
    since a composite forwards axes it has never heard of.
    """
    # A composite unit has no long name, so `name` falls back to the symbol.
    q = u.Q([1.0, 2, 3], "km / s")
    spec = parse_spec("name")
    assert inert_axes(q, spec, render(q, spec)) == ["unit"]


@pytest.mark.parametrize("spec", ["name", "dim", "mul", "latex-mul", ".2f"])
def test_a_live_axis_is_never_reported_inert(spec: str) -> None:
    """No false positives: each of these visibly changes a metre quantity."""
    q = u.Q([1.0, 2, 3], "m")
    parsed = parse_spec(spec)
    assert inert_axes(q, parsed, render(q, parsed)) == []


def test_the_warning_is_off_by_default_and_opt_in() -> None:
    """Each probe is a second render, so a library sets this for itself.

    The flag lives on the engine rather than in `unxt.config`, so it survives
    extraction: a consuming package turns it on for its own debugging instead
    of the engine reaching into any one library's configuration.
    """
    q = u.Q([1.0, 2, 3], "km / s")
    assert engine_module.WARN_INERT_AXES is False

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert pspec(q, "name") == "[1., 2., 3.] km / s"
    assert not caught

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert pspec(q, "name", warn_inert=True) == "[1., 2., 3.] km / s"
    assert len(caught) == 1
    assert "changes nothing" in str(caught[0].message)
