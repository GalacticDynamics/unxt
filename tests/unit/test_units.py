"""Tests for `unxt.units` -- the `unit` constructor and `AbstractUnit` alias."""

import astropy.units as apyu
import pytest
from plum import NotFoundLookupError

import unxt as u
from unxt.units import AbstractUnit


class TestUnitFunctionUnits:
    """`unit()` accepts astropy function units (mag / dex / dB).

    Regression: `AbstractUnit` was `Unit | UnitBase | CompositeUnit`, none of
    which is a `FunctionUnitBase`. So `unit("mag(AB)")` built a `MagUnit` via
    `apyu.Unit(...)` but failed its own `-> AbstractUnit` return annotation
    (`TypeCheckError` under beartype, `TypeError` otherwise), and passing a
    function-unit object raised `NotFoundLookupError`. The `APYUnits` interop
    alias already listed `FunctionUnitBase`, so the type surface claimed support
    the constructor did not provide.
    """

    @pytest.mark.parametrize(
        ("string", "expected"),
        [
            ("mag(AB)", "mag(AB)"),
            ("dex(cm/s2)", "dex(cm / s2)"),
            ("dB(mW)", "dB(mW)"),
        ],
    )
    def test_from_string(self, string: str, expected: str) -> None:
        """A function-unit string round-trips to the astropy function unit."""
        got = u.unit(string)
        assert isinstance(got, apyu.FunctionUnitBase)
        assert got == apyu.Unit(expected)

    def test_from_object_is_identity(self) -> None:
        """A function-unit object passes straight through."""
        mag = apyu.Unit("mag(AB)")
        assert u.unit(mag) is mag

    def test_is_abstract_unit(self) -> None:
        """A function unit is an instance of the (widened) `AbstractUnit`."""
        assert isinstance(u.unit("mag(AB)"), AbstractUnit)

    def test_function_unit_flows_downstream(self) -> None:
        """A function unit works end-to-end, not just in the constructor.

        Widening the *type* would be hollow if the rest of the machinery choked
        on it -- so pin the whole path: dimension, construction, and a
        round-trip through ``ustrip``.
        """
        mag = u.unit("mag(AB)")
        assert u.dimension_of(mag) is not None  # does not raise
        q = u.Q(1.0, mag)
        assert q.unit == mag
        assert float(u.ustrip(mag, q)) == 1.0


class TestUnitStructuredIsNotYetSupported:
    """`StructuredUnit` is intentionally still rejected -- see the alias comment.

    Unlike function units it has no single physical dimension, so accepting it
    would build a ``Quantity`` whose ``dimension_of`` raises. Until that is
    designed, ``unit()`` should reject it *cleanly* at the boundary rather than
    return a half-usable object. This test pins that boundary so a future
    widening is a deliberate, tested change.
    """

    def test_structured_unit_rejected(self) -> None:
        """`unit()` has no dispatch for a `StructuredUnit`, so it raises."""
        su = apyu.StructuredUnit((apyu.m, apyu.s))
        with pytest.raises(NotFoundLookupError):
            u.unit(su)
