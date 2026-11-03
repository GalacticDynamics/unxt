"""Tests for `unxt`'s layer over the string-formatting engine.

The engine's own tests live in the `wadler-lindig-parts` distribution. What
stays here needs a quantity, a unit, a unit system or an array library -- plus
the registry invariants, which both suites check because the two registries
differ: this one has `unxt`'s axes and aliases in it.
"""

import subprocess
import sys
import textwrap

import jax
import numpy as np
import pytest
import wadler_lindig as wl

import unxt as u
from unxt._src.fmt import (
    ALIASES,
    AXES,
    MARKUPS,
    Axis,
    Spec,
    doc_part,
    doc_to_str,
    inert_axes,
    math_doc,
    parse_spec,
    parts_to_doc,
    parts_to_markup,
    pparts,
    pspec,
    pvalue,
    register_axis,
    register_markup,
    render,
    unregister_axis,
    unregister_markup,
    unwrap_math,
)

# ============================================================================
# The grammar's own invariants


def _keywords() -> set[str]:
    """The flat keyword namespace, derived from the axes the engine publishes.

    The engine's own index of it is private to the `wadler-lindig-parts`
    distribution; `Axis.keywords` is the public statement of the same thing,
    and this registry is the engine's axes plus `unxt`'s.
    """
    return {word for ax in AXES.values() for word in ax.keywords}


def test_keyword_namespace_is_pairwise_disjoint() -> None:
    """A keyword may never also be an alias.

    This is the invariant that lets the keyword run be order-independent with
    no content-sniffing: if the word alone decides which axis it sets, then
    ``"html-bare"`` and ``"bare-html"`` cannot differ. The engine's index is
    not ours to inspect across the distribution boundary, so derive the
    namespace from the axis declarations the engine does publish -- the real
    risk here is one of `unxt`'s own words colliding with an `ALIASES` entry.
    """
    assert not _keywords() & set(ALIASES)


@pytest.mark.parametrize("alias", sorted(ALIASES))
def test_every_alias_expands_to_a_valid_core_spec(alias: str) -> None:
    """An alias is sugar, never new meaning: it must parse as its expansion."""
    assert parse_spec(alias) == parse_spec(ALIASES[alias])


def test_every_axis_applies_to_some_layout() -> None:
    """An axis no layout accepts would be unreachable."""
    for name, ax in AXES.items():
        assert ax.layouts, name


def test_spec_defaults_are_the_grammar_defaults() -> None:
    """An all-omitted spec is exactly ``Spec.of()``."""
    assert parse_spec("product") == Spec.of()


_REPR_MIXIN_AT_CLASS_CREATION = textwrap.dedent("""
    {preamble}
    from wadler_lindig_parts import PPart, ReprMixin, pparts

    class Shown(ReprMixin):
        __repr_spec__ = "compact"  # `unxt`'s alias, for `unxt`'s `abbrev` axis

    @pparts.dispatch
    def _(obj: Shown, /, *, markup="text", **kw):
        return (PPart("name", "shown"),)

    print(repr(Shown()))
""")


def test_importing_unxt_registers_its_axes_before_any_class_is_built() -> None:
    """`ReprMixin` resolves `__repr_spec__` at class creation, across the seam.

    The engine is its own distribution now, so `unxt`'s axes and aliases reach
    its registry only because importing `unxt` imports `unxt._src.fmt`, which
    imports `axes`. A class whose spec names one is validated at the ``class``
    statement, so a lazier import would fail *there* rather than at first
    print. Both halves run in a fresh interpreter: this suite's own imports
    have already registered everything, which is exactly what would hide it.
    """
    with_unxt = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-c",
            _REPR_MIXIN_AT_CLASS_CREATION.format(preamble="import unxt"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert with_unxt.stdout.strip() == "shown"

    # Not vacuous: without that import the engine has never heard of "compact".
    without = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _REPR_MIXIN_AT_CLASS_CREATION.format(preamble="")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert without.returncode != 0
    assert "compact" in without.stderr


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
        ("latex", r"$[1.,~2.,~3.] \, \mathrm{m}$"),
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


def test_the_ipython_reprs_keep_the_operator() -> None:
    r"""They share the renderer but not `Spec`, and that is deliberate.

    The separator is the visible consequence -- ``mul`` in a notebook, ``bare``
    from a format spec. The flat layout is the other: a markup string takes no
    width breaks, which neither markup would honour anyway.
    """
    q = u.Q(1.0, "m")
    assert q._repr_html_() == "<span>1.</span> * <span>m</span>"
    assert pspec(q, "html") == "<span>1.</span> <span>m</span>"
    assert q._repr_latex_() == r"$1. \; \mathrm{m}$"
    assert pspec(q, "latex") == r"$1. \, \mathrm{m}$"

    wide = u.Q(np.arange(30.0), "Msun/pc3")
    assert "\n" not in wide._repr_html_()
    assert "\n" not in wide._repr_latex_()
    assert "\n" in pspec(wide, "html")


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

    A unit system has no elements to format. It must say so rather than
    quietly render something else.
    """
    with pytest.raises(TypeError, match="does not support a value format spec"):
        pspec(u.unitsystem("m", "s", "kg", "rad"), ".2f")


def test_unknown_markup_is_named() -> None:
    with pytest.raises(ValueError, match=r"unknown markup 'markdown'"):
        parts_to_markup(pparts(u.Q(1.0, "m")), markup="markdown")


@pytest.mark.parametrize("word", sorted(_keywords() | set(ALIASES)))
def test_grammar_words_are_not_valid_value_specs(word: str) -> None:
    """Claiming these words is strictly additive.

    If any were also a legal Python format spec, the scan rule would silently
    prefer the keyword reading and steal a meaning users already had.
    """
    for value in (3.14, 3, complex(1, 2), np.float32(1.5)):
        with pytest.raises((ValueError, TypeError)):
            format(value, word)


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


def test_rendered_markup_is_not_escaped_again() -> None:
    r"""Escaping unxt's own LaTeX would turn ``\mathrm`` into ``\textbackslash``."""
    assert pspec(u.Q(1.0, "m"), "latex-mul") == r"$1. \; \mathrm{m}$"


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


def test_an_explicit_axis_assignment_resolves_without_ambiguity() -> None:
    """``axis=word`` is always available, collision or not.

    It also says what a keyword actually does -- set an axis to a value -- so
    bare ``dim`` is shorthand for ``unit=dim``. A downstream package can write
    the explicit form from the start and be immune to a word it does not yet
    share.
    """
    q = u.Q([1.0, 2, 3], "m")
    assert pspec(q, "unit=dim") == pspec(q, "dim")
    assert pspec(q, "markup=latex-sep=mul") == pspec(q, "latex-mul")


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
        # Either explicit form still works, and unxt's is unchanged.
        assert pspec(q, "unit=dim") == "[1., 2., 3.] length"
        # Surgical: only the colliding token needs the prefix.
        assert pspec(q, "latex-mul-unit=dim") == "$[1.,~2.,~3.] \\; length$"
        # Every other word is untouched by the collision.
        assert pspec(q, "mul") == "[1., 2., 3.] * m"
    finally:
        unregister_axis("manifold")


@pytest.mark.parametrize("spec", [":>6", "=>6", "=6", "=^7"])
def test_an_unresolvable_assignment_is_free_text_not_an_error(spec: str) -> None:
    """A ``=`` in a value spec must not be read as an axis assignment.

    ``=`` is also an *align* character, but align only appears at position 0
    or 1, so ``=>6`` names no axis and falls through to the value spec exactly
    like any other non-keyword. Same for a ``:`` fill character, which the
    grammar no longer uses at all.
    """
    assert parse_spec(spec)["value"] == spec
    assert pspec(u.Q(3, "m"), spec).endswith(" m")


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


def test_latex_value_text_is_escaped():
    """A value's `%` or `_` must not reach LaTeX raw (comment / subscript)."""
    pct = f"{u.Q([0.25, 0.5], 'm'):latex-.1%}"
    assert r"25.0\%,~50.0\%" in pct
    assert r"weak\_f32" in f"{u.Q(1.0, 'm'):latex-type}"


def test_unit_system_pdoc_wraps_at_narrow_width():
    """`__pdoc__` offers a break per unit, and repr still round-trips."""
    usys = u.unitsystem("m", "s", "kg", "rad")
    wrapped = wl.pformat(usys, width=20)
    assert "\n" in wrapped
    assert eval(wrapped, {"unitsystem": u.unitsystem}) == usys  # noqa: S307
    assert repr(usys) == "unitsystem(['m', 's', 'kg', 'rad'])"


@pytest.mark.parametrize("width", [10, 20, 40, 88])
@pytest.mark.parametrize("arr", [[1.0, 2.0, 3.0], [1.5, 2.5]])
def test_product_text_is_the_wl_layout_of_its_parts(width, arr):
    """Text product output is exactly the parts' WL doc laid out at `width`."""
    q = u.Q(arr, "m")
    assert render(q, parse_spec("mul"), width=width) == doc_to_str(
        parts_to_doc(pparts(q), sep="mul"), width
    )


def test_call_layout_is_wl_pformat_for_nested_dataclasses():
    """`call` adds nothing over `wadler_lindig` for a plain dataclass tree."""
    import dataclasses  # noqa: PLC0415

    @dataclasses.dataclass
    class Box:
        usys: object
        n: int

    box = Box(u.unitsystem("m", "s", "kg", "rad"), 3)
    for width in (20, 88):
        assert render(box, parse_spec("call"), width=width) == wl.pformat(
            box, width=width
        )


@pytest.mark.parametrize("width", [10, 20, 40])
def test_large_array_respects_width(width):
    text = render(u.Q(np.arange(30.0), "m"), parse_spec("mul"), width=width)
    assert max(map(len, text.splitlines())) <= width


def test_array_text_is_unchanged_when_it_fits():
    q = u.Q([1.5, 10.0], "m")
    assert f"{q:mul}" == "[ 1.5, 10. ] * m"
    assert f"{u.Q([[1.0, 2.0], [3.0, 4.0]], 'm'):mul}" == "[[1., 2.], [3., 4.]] * m"


@pytest.mark.parametrize(
    "arr",
    [
        np.array([1.5, 10.0]),
        np.array([1.0, 2.0, 3.0]),
        np.array([True, False]),
        np.array([1, 20]),
        np.array([1 + 2j, 3j]),
        np.array([np.nan, np.inf, 1.0]),
    ],
)
def test_1d_value_matches_the_old_numpy_rendering(arr):
    old = np.array2string(arr, separator=", ", max_line_width=10**9)
    assert f"{u.Q(arr, 'm'):mul}" == f"{old} * m"


@pytest.mark.parametrize(
    ("value", "mul", "latex"),
    [
        (1.0, "1. * m", r"$1. \, \mathrm{m}$"),
        ([], "[] * m", r"$[] \, \mathrm{m}$"),
        ([True, False], "[ True, False] * m", None),
        ([1, 2], "[1, 2] * m", None),
        ([1 + 2j], "[1.+2.j] * m", None),
        ([np.nan, np.inf], "[nan, inf] * m", None),
    ],
)
def test_pvalue_handles_every_shape_and_dtype(value, mul, latex):
    q = u.Q(value, "m")
    assert f"{q:mul}" == mul
    if latex is not None:
        assert f"{q:latex}" == latex
    for spec in ("html", "latex", "type", "array"):
        assert isinstance(f"{q:{spec}}", str)


def test_summarised_2d_array_respects_width():
    q = u.Q(np.arange(4000.0).reshape(2000, 2), "m")
    text = render(q, parse_spec("mul"), width=40)
    assert "..." in text
    assert max(map(len, text.splitlines())) <= 40


def test_bracket_in_a_string_element_is_not_structure():
    arr = np.array(["a]b", "c"])
    old = np.array2string(arr, separator=", ", max_line_width=10**9)
    assert f"{u.quantity.StaticQuantity(arr, 'm'):mul}" == f"{old} * m"
    assert old == "['a]b', 'c']"


@pytest.mark.parametrize(
    "arr",
    [
        np.array(["a]b", "c [d", "e f"]),
        np.array([[1, 2], "x]"], dtype=object),
        np.array([b"x]", b"y"]),
    ],
)
def test_non_numeric_values_are_never_truncated(arr):
    from unxt._src.fmt import pvalue  # noqa: PLC0415

    old = np.array2string(arr, separator=", ", max_line_width=10**9)
    assert doc_to_str(pvalue(arr)) == old


def test_bracket_fill_in_value_spec_does_not_crash():
    from unxt._src.fmt import pvalue  # noqa: PLC0415

    a = np.array([1.0, 2.0])
    old = np.array2string(
        a,
        separator=", ",
        formatter={"all": lambda v: format(v, "[>5")},
        max_line_width=10**9,
    )
    assert doc_to_str(pvalue(a, value_spec="[>5")) == old


def test_large_array_summarises_past_the_threshold():
    text = f"{u.Q(np.arange(2000.0), 'm'):mul}"
    assert "..." in text


def test_value_elements_are_escaped_for_the_markup():
    assert r"25.0\%,~50.0\%" in f"{u.Q([0.25, 0.5], 'm'):latex-.1%}"
    assert "&lt;" in parts_to_markup(
        (doc_part("value", wl.TextDoc("&lt;"), "markup"),), markup="html"
    )


def test_tracer_value_renders_a_summary_in_every_markup():
    seen = []

    @jax.jit
    def f(x):
        q = u.Q(x, "m")
        seen.extend(f"{q:{s}}" for s in ("mul", "html", "latex"))
        return x

    f(np.arange(3.0))
    assert all("f32[3]" in s or "f64[3]" in s for s in seen)


@pytest.mark.parametrize("markup", ["text", "html", "latex"])
@pytest.mark.parametrize("spec", ["", "bare", "mul"])
def test_markup_strings_equal_the_flat_layout_of_the_unified_doc(markup, spec):
    q = u.Q([1.0, 2.0], "m")
    parts = pparts(q, markup=markup)
    sep = spec or None
    flat = doc_to_str(parts_to_doc(parts, markup=markup, sep=sep), 10**9)
    assert parts_to_markup(parts, markup=markup, sep=sep) == flat


@pytest.mark.parametrize("width", [10, 20, 30, 40, 88])
def test_html_lays_out_exactly_like_text(width):
    """``<span>`` is zero-width, so HTML breaks exactly where plain text does.

    This is the property that lets one `parts_to_doc` serve every markup: strip
    the wrappers and the layout is byte-identical. While wrappers were billed
    as columns it was false at every width but the widest.
    """
    q = u.Q(np.arange(30.0), "m")
    out = render(q, parse_spec("html"), width=width)
    stripped = out.replace("<span>", "").replace("</span>", "")
    assert stripped == render(q, parse_spec("text"), width=width)


UNITS_WITH_LATEX = ["m", "Jy", "deg", "m/s2", "km/s", "Msun/pc3", "mas/yr", "1/m"]


@pytest.mark.parametrize("unit", UNITS_WITH_LATEX)
def test_math_doc_reproduces_the_fragment_exactly(unit):
    """Zero-width billing must not cost a single character of output.

    This is a change to the layout alone, so the flat rendering has to stay
    byte-identical to what astropy handed over.
    """
    frag = unwrap_math(u.unit(unit).to_string("latex"))
    assert doc_to_str(math_doc(frag), 10**9) == frag


def _narrowest_flat(q, markup):
    """The narrowest width at which ``q`` still renders on one line."""
    spec = parse_spec(markup)
    return next(w for w in range(2, 140) if "\n" not in render(q, spec, width=w))


@pytest.mark.parametrize("unit", UNITS_WITH_LATEX)
def test_latex_does_not_bill_its_markup_as_columns(unit):
    r"""LaTeX lays out by its visible glyphs, not by its source length.

    ``\mathrm{\frac{m}{s^{2}}}`` is 24 characters rendering as about three
    columns; billed in full it pushed the break point tens of columns out, so
    ``latex-width=`` broke lines that the same quantity's text form fitted
    comfortably. LaTeX may now legitimately come out *narrower* than the text
    spelling -- a fraction stacks, and ``solMass / pc3`` is 13 columns of text
    against roughly four of ``\frac{M_{\odot}}{pc^{3}}``.

    The one allowed column of slack is real rather than markup: the separator
    ``\,`` is set off by a space on each side where text has a single space.
    """
    q = u.Q([1.0, 2.0, 3.0], unit)
    assert _narrowest_flat(q, "latex") <= _narrowest_flat(q, "text") + 1


def test_latex_width_breaks_at_the_same_scale_as_text():
    """The concrete regression: this fitted in text but broke in LaTeX.

    ``m / s2`` needs 19 columns as text. LaTeX billed its markup too and so
    held out to 40, which is why a width in between broke one and not the
    other.
    """
    q = u.Q([1.0, 2.0, 3.0], "m/s2")
    assert "\n" not in format(q, "latex-width=20")
    assert "\n" not in format(q, "text-width=20")
    assert "\n" in format(q, "latex-width=12")


_GOOD_ROW = {"_content": "[{}]", "wrap": "{}", "vsep": ", ", "escape": None}


def test_register_markup_validates_and_extends_the_markup_axis():
    register_markup("brackets", _GOOD_ROW)
    try:
        assert parse_spec("brackets")["markup"] == "brackets"
        assert f"{u.Q(1.0, 'm'):brackets}" == "[1.] * [m]"
        with pytest.raises(ValueError, match="already"):
            register_markup("brackets", _GOOD_ROW)
        register_markup("brackets", _GOOD_ROW, replace=True)
        with pytest.raises(ValueError, match="missing"):
            register_markup("broken", {"wrap": "{}"})
    finally:
        unregister_markup("brackets")
    assert "brackets" not in MARKUPS
    assert "brackets" not in _keywords()


def test_width_and_indent_in_the_spec() -> None:
    q = u.Q(np.arange(30.0), "m")
    wide = f"{q:mul-width=200}"
    narrow = f"{q:mul-width=20}"
    assert "\n" not in wide
    assert max(map(len, narrow.splitlines())) <= 20
    two = f"{q:mul-width=20-indent=2}".splitlines()
    four = f"{q:mul-width=20-indent=4}".splitlines()
    assert two != four

    def lead(ln: str) -> int:
        return len(ln) - len(ln.lstrip())

    # continuation lines indent two more spaces under indent=4 than indent=2
    assert len(two) == len(four)
    assert {lead(b) - lead(a) for a, b in zip(two[1:-1], four[1:-1], strict=True)} == {
        2
    }


def test_width_is_rejected_when_not_a_number() -> None:
    """Non-numeric text is not a width: it falls through to the value spec."""
    with pytest.raises(ValueError, match="invalid format spec"):
        f"{u.Q(1.0, 'm'):width=wide}"


def test_spec_width_overrides_the_argument_and_defaults_do_not_clobber() -> None:
    q = u.Q(np.arange(30.0), "m")
    assert "\n" in render(q, parse_spec("mul"), width=20)
    assert "\n" not in render(q, parse_spec("mul-width=200"), width=20)
    # indent=None leaves the caller's indent alone
    assert render(q, Spec.of(layout="call"), indent=4) == render(
        q, Spec.of(layout="call", indent=4)
    )


def test_call_layout_honours_width() -> None:
    q = u.Q(np.arange(6.0), "m")
    assert "\n" in f"{q:call-width=20}"
    assert "\n" not in f"{q:call-width=200}"


@pytest.mark.parametrize("digit", ["\u00b2", "\u0663"])
def test_non_ascii_digits_are_not_a_count(digit: str) -> None:
    with pytest.raises(ValueError, match="invalid format spec"):
        f"{u.Q(1.0, 'm'):width={digit}}"


def test_zero_width_and_indent_render() -> None:
    q = u.Q(np.arange(3.0), "m")
    assert "1." in f"{q:mul-width=0}"
    assert "1." in f"{q:mul-indent=0}"


def test_hand_built_spec_without_width_or_indent_renders() -> None:
    q = u.Q(1.0, "m")
    spec = Spec({k: v for k, v in Spec.of().items() if k not in ("width", "indent")})
    assert render(q, spec) == render(q, Spec.of())


def test_long_array_exact_layout_at_default_width():
    """A long 1-D array flows across lines; it is not one element per line."""
    out = f"{u.Q(np.arange(30.0), 'm'):mul}"
    assert out == (
        "[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14.,"
        " 15., 16.,\n"
        "   17., 18., 19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29.] *\n"
        "  m"
    )


@pytest.mark.parametrize("spec", ["mul-width=40", "bare-width=40", "width=40-.2f"])
@pytest.mark.parametrize(
    "value",
    [
        np.arange(30.0),
        np.arange(60.0).reshape(2, 30),
        np.arange(4000.0).reshape(2000, 2),
    ],
)
def test_array_lines_respect_width(spec, value):
    out = f"{u.Q(value, 'm'):{spec}}"
    assert max(map(len, out.splitlines())) <= 40
    assert len(out.splitlines()) < 400


@pytest.mark.parametrize(
    "value",
    [
        np.array([1.5, 10.0]),
        np.array([1.0, 2.0, 3.0]),
        np.array([]),
        np.array(1.5),
        np.arange(10.0),
    ],
)
def test_short_1d_arrays_are_byte_identical_to_numpy(value):
    expected = np.array2string(value, separator=", ", max_line_width=10**9)
    assert f"{u.Q(value, 'm'):mul}" == f"{expected} * m"


@pytest.mark.parametrize("name", ["bfloat16", "float8_e4m3fn", "int4"])
def test_ml_dtype_arrays_take_the_parser_path(name):
    ml_dtypes = pytest.importorskip("ml_dtypes")
    arr = np.arange(40).astype(getattr(ml_dtypes, name))
    out = f"{u.Q(arr, 'm'):mul-width=40}"
    assert max(map(len, out.splitlines())) <= 40
    assert len(out.splitlines()) > 1


def test_bfloat16_jax_array_wraps():
    import jax.numpy as jnp  # noqa: PLC0415

    out = f"{u.Q(jnp.arange(40, dtype=jnp.bfloat16), 'm'):mul-width=40}"
    assert max(map(len, out.splitlines())) <= 40


def test_flat_fallback_has_no_trailing_space():
    out = doc_to_str(pvalue(np.ones((2, 2), dtype=object)), 88)
    assert all(line == line.rstrip() for line in out.splitlines())
    assert out.startswith("[[1, 1],\n [1, 1]]")


def test_unit_system_follows_markup_specs():
    """A unit system renders under html/latex like any other type."""
    usys = u.unitsystem("m", "s", "kg", "rad")
    assert f"{usys:html}".startswith("<span>unitsystem</span>(<span>m</span>")
    assert f"{usys:latex}".startswith("$")
    assert f"{usys:compact}" == "unitsystem(m, s, kg, rad)"


def test_short_arrays_unwraps_a_static_value() -> None:
    """``StaticValue`` is not an array, so the kind hook declines it.

    The wrapper is then suppressed and the inner NumPy array renders without
    the ``(numpy)`` kind suffix.
    """
    assert pspec(u.StaticQuantity([1.0, 2.0], "m"), "type-mul") == "f64[2] * m"


def test_pparts_shim_exports_the_build_by_hand_tools():
    """A downstream package can build a doc from its own parts, no `_src`."""
    import unxt._pparts as shim  # noqa: PLC0415

    for name in (
        "parts_to_doc",
        "parts_to_markup",
        "doc_to_str",
        "pvalue",
        "doc_part",
        "pparts_to_pdoc",
        "ReprMixin",
        "register_markup",
        "unregister_axis",
        "unregister_alias",
        "unregister_markup",
    ):
        assert name in shim.__all__
        assert callable(getattr(shim, name))


def test_array_doc_rejects_text_after_the_closing_bracket():
    """Trailing text is not numpy's structure: callers fall back on IndexError."""
    from unxt._src.fmt.axes import _SENTINEL, _array_doc  # noqa: PLC0415

    with pytest.raises(IndexError):
        _array_doc(f"[1{_SENTINEL}2] x", vsep=", ", escape=str)
