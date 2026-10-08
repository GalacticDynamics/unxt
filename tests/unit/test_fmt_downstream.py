"""A type outside unxt, using only the public `pparts` surface (no numpy)."""

import dataclasses

import pytest
import wadler_lindig as wl

from unxt._src.fmt import (
    PPart,
    ReprMixin,
    pparts,
    pspec,
    register_alias,
    unregister_alias,
)


@dataclasses.dataclass(repr=False)
class Version(ReprMixin):
    """A two-part version number."""

    major: int
    minor: int
    __repr_spec__ = "bare"


@pparts.dispatch
def _(obj: Version, /, *, markup="text", **kw):
    return (
        PPart("major", str(obj.major)),
        PPart("dot", ".", "sep"),
        PPart("minor", str(obj.minor)),
    )


def test_every_route_comes_from_one_pparts():
    v = Version(1, 2)
    assert repr(v) == "1.2"
    assert str(v) == "1.2"
    assert f"{v}" == "1.2"
    assert f"{v:html}" == "<span>1</span>.<span>2</span>"
    assert f"{v:latex}" == "$1.2$"
    assert v._repr_html_() == "<span>1</span>.<span>2</span>"
    assert v._repr_latex_() == "$1.2$"
    assert wl.pformat(v) == "1.2"
    assert pspec(v, "") == str(v)


def test_bad_repr_spec_fails_at_class_creation():
    with pytest.raises(ValueError, match="invalid format spec"):

        class Bad(ReprMixin):
            __repr_spec__ = "nonsense-axis"


@pytest.mark.parametrize(
    "spec", ["bare", "mul", "html", ".2f", "mul-.2f", "call", "d", "x", "s"]
)
def test_good_repr_specs_are_accepted(spec):
    class Ok(ReprMixin):
        __repr_spec__ = spec

    assert Ok._repr_spec_parsed is not None


def test_an_alias_must_exist_before_the_class():
    with pytest.raises(ValueError, match="invalid format spec"):

        class Early(ReprMixin):
            __repr_spec__ = "later_alias"

    register_alias("later_alias", "bare")  # registered after: too late for Early
    try:

        class Late(ReprMixin):
            __repr_spec__ = "later_alias"

        assert Late._repr_spec_parsed["sep"] == "bare"
    finally:
        unregister_alias("later_alias")


def test_spec_is_inherited_and_overridable():
    class Mid(ReprMixin):
        __repr_spec__ = "mul"

    class Leaf(Mid):
        pass

    class Over(Mid):
        __repr_spec__ = "bare"

    assert Leaf._repr_spec_parsed["sep"] == "mul"
    assert Over._repr_spec_parsed["sep"] == "bare"


def test_dataclass_repr_true_would_shadow_the_mixin():
    """Documented hazard: @dataclass (repr=True) wins over an inherited __repr__."""

    @dataclasses.dataclass
    class Shadowed(ReprMixin):
        x: int

    assert "Shadowed(x=1)" in repr(Shadowed(1))  # dataclass repr, not the mixin's


def test_pformat_matches_repr_for_product_spec():
    @dataclasses.dataclass(repr=False)
    class Pt(ReprMixin):
        x: int

    @pparts.dispatch
    def _(obj: Pt, /, *, markup="text", **kw):
        return (PPart("x", str(obj.x)), PPart("u", "m", "unit"))

    assert wl.pformat(Pt(3)) == repr(Pt(3))


@pytest.mark.parametrize("spec", ["product", "call"])
def test_missing_pparts_is_a_clear_error_not_recursion(spec):
    @dataclasses.dataclass(repr=False)
    class N(ReprMixin):
        """Forgot to register pparts."""

        x: int
        __repr_spec__ = spec

    msg = "N subclasses ReprMixin but registers no pparts"
    for call in (repr, str, lambda o: f"{o:mul}", wl.pformat):
        with pytest.raises(TypeError, match=msg):
            call(N(1))


def test_pdoc_applies_the_class_spec():
    @dataclasses.dataclass(repr=False)
    class B(ReprMixin):
        """Bare layout with a mul separator part."""

        x: int
        __repr_spec__ = "bare"

    @pparts.dispatch
    def _(obj: B, /, *, markup="text", **kw):
        return (
            PPart("value", str(obj.x)),
            PPart("mul", " * ", "sep"),
            PPart("unit", "m"),
        )

    assert repr(B(3)) == "3 m"
    assert wl.pformat(B(3)) == "3 m"
    assert wl.pformat([B(3)]) == repr([B(3)]) == "[3 m]"
