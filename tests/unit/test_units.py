"""Tests for `unxt.units` -- the `unit` constructor and `AbstractUnit` alias."""

import astropy.units as apyu
import pytest
from plum import NotFoundLookupError

import unxt as u
from unxt.units import AbstractUnit


class TestUnitFunctionUnits:
    """`unit()` accepts astropy function units (mag / dex / dB).

    Regression: `AbstractUnit` excluded `FunctionUnitBase`, so `unit("mag(AB)")`
    failed its own return annotation and a function-unit object was unmatched.
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
        """A function unit works end-to-end: dimension, construction, ustrip."""
        mag = u.unit("mag(AB)")
        assert u.dimension_of(mag) is not None  # does not raise
        q = u.Q(1.0, mag)
        assert q.unit == mag
        assert float(u.ustrip(mag, q)) == 1.0


class TestUnitStructuredIsNotYetSupported:
    """`StructuredUnit` is intentionally rejected -- see the alias comment.

    It has no single dimension, so accepting it would build a `Quantity` whose
    `dimension_of` raises. This pins the boundary until that is designed.
    """

    def test_structured_unit_rejected(self) -> None:
        """`unit()` has no dispatch for a `StructuredUnit`, so it raises."""
        su = apyu.StructuredUnit((apyu.m, apyu.s))
        with pytest.raises(NotFoundLookupError):
            u.unit(su)
