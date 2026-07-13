"""Tests for ParametricQuantity printing with wadler-lindig."""

import wadler_lindig as wl
from unxts.parametric import PQ


class TestParametricShortName:
    """Short-name / pretty-printing behavior specific to ``ParametricQuantity``."""

    def test_parametricquantity_short_name(self):
        """``ParametricQuantity`` (``PQ``) has short_name 'PQ'."""
        assert hasattr(PQ, "short_name")
        assert PQ.short_name == "PQ"

    def test_uses_full_name_by_default(self):
        """The parametric class uses its full name by default."""
        pq = PQ([1, 2, 3], "m")
        pq_result = wl.pformat(pq)
        assert pq_result.startswith("ParametricQuantity")
        assert not pq_result.startswith("PQ(")

    def test_use_short_name_true(self):
        """``ParametricQuantity`` uses its short name 'PQ'."""
        pq = PQ([1, 2, 3], "m")
        result = wl.pformat(pq, use_short_name=True)
        assert result.startswith("PQ(")
        assert "unit='m'" in result

    def test_use_short_name_with_include_params(self):
        """The parametric class shows its dimension parameter."""
        pq = PQ([1, 2, 3], "m")
        pq_result = wl.pformat(pq, use_short_name=True, include_params=True)
        assert pq_result.startswith("PQ['length']")
