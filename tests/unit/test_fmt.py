"""Tests for the `unxt._fmt` string-formatting engine."""

import re

import jax
import numpy as np
import pytest
import wadler_lindig as wl

import unxt as u
from unxt._fmt import (
    FORMAT_PRESETS,
    MARKUPS,
    PGroup,
    PPart,
    doc_to_str,
    parts_to_doc,
    parts_to_markup,
    pparts,
    pspec,
)
from unxt._src.fmt import REQUIRED_MARKUP_KEYS

# ============================================================================
# Presets


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("compact", "Q([1., 2., 3.], unit='m')"),
        ("full", "Quantity(Array([1., 2., 3.], dtype=float32), unit='m')"),
        ("short", "f32[3] * m"),
        ("mul", "[1., 2., 3.] * m"),
        ("bare", "[1., 2., 3.] m"),
        ("latex", r"$[1.,~2.,~3.] \; \mathrm{m}$"),
        ("html", "<span>[1., 2., 3.]</span> * <span>m</span>"),
    ],
)
def test_preset_renders_quantity(spec: str, expected: str) -> None:
    """Every preset produces its documented string for an array quantity."""
    assert pspec(u.Q([1.0, 2, 3], "m"), spec) == expected


def test_every_preset_is_reachable_for_a_quantity() -> None:
    """No preset raises for a plain quantity.

    Guards the defect where the preset kwargs did not fit the consumers'
    signatures and five of eight presets raised ``TypeError``.
    """
    q = u.Q([1.0, 2, 3], "m")
    for spec in FORMAT_PRESETS:
        assert isinstance(pspec(q, spec), str)


@pytest.mark.parametrize("spec", ["mul", "bare", "latex", "html", "short"])
def test_dimensionless_has_no_unit_fragment(spec: str) -> None:
    r"""A dimensionless quantity drops the separator and the unit.

    ``Unit("").to_string("latex")`` is ``$\mathrm{}$``, which is truthy once
    the ``$`` are stripped, so deciding emptiness on the LaTeX form emitted a
    phantom ``\mathrm{}``.
    """
    out = pspec(u.Q([1.0, 2], ""), spec)
    assert "*" not in out
    assert r"\mathrm{}" not in out


def test_empty_spec_is_not_a_preset() -> None:
    """``""`` must never be a preset key.

    A format spec may use ``:`` as its fill character, and an empty-string
    preset would make ``f"{q::>10}"`` parse as a preset plus a spec, silently
    dropping the fill character.
    """
    assert "" not in FORMAT_PRESETS


@pytest.mark.parametrize("name", sorted(FORMAT_PRESETS))
def test_preset_names_are_not_valid_value_specs(name: str) -> None:
    """Registering these names is strictly additive."""
    for value in (3.14, 3, complex(1, 2), np.float32(1.5)):
        with pytest.raises((ValueError, TypeError)):
            format(value, name)


# ============================================================================
# Errors


def test_unknown_spec_names_the_type_and_the_presets() -> None:
    """A bad spec raises `ValueError`, not NumPy's opaque `TypeError`."""
    with pytest.raises(ValueError, match=r"invalid format spec 'nonsense'"):
        pspec(u.Q(3.14, "m"), "nonsense")


def test_valid_spec_failing_for_array_reasons_keeps_its_TypeError() -> None:  # noqa: N802
    """``.2f`` on an array is a *valid* spec NumPy rejects.

    Translating it to ``ValueError: invalid format spec`` would be a lie and
    would break downstream ``except TypeError`` handlers.
    """
    with pytest.raises(TypeError, match="unsupported format string"):
        pspec(u.Q([1.5, 2.5], "m"), ".2f")


def test_unknown_markup_is_named() -> None:
    with pytest.raises(ValueError, match=r"unknown markup 'markdown'"):
        parts_to_markup(pparts(u.Q(1.0, "m")), markup="markdown")


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
    assert pspec(u.Q(1.0, "m"), "latex") == r"$1. \; \mathrm{m}$"


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
    assert pspec(u.Q(1.0, "m"), "short") == "weak_f32[] * m"


def test_short_arrays_unwraps_a_static_value() -> None:
    """``StaticValue`` is not an array, so the kind hook declines it.

    The wrapper is then suppressed and the inner NumPy array renders without
    the ``(numpy)`` kind suffix.
    """
    assert pspec(u.StaticQuantity([1.0, 2.0], "m"), "short") == "f64[2] * m"
