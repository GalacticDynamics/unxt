"""Tests for QuantityMatrix printing with wadler-lindig."""

import jax.numpy as jnp
import wadler_lindig as wl
from unxts.linalg import QM, QuantityMatrix


class TestQuantityMatrixShortName:
    """Short-name / pretty-printing behavior specific to ``QuantityMatrix``."""

    def test_quantitymatrix_short_name(self):
        """``QuantityMatrix`` (``QM``) has short_name 'QM'."""
        assert hasattr(QuantityMatrix, "short_name")
        assert QuantityMatrix.short_name == "QM"
        assert QM.short_name == "QM"

    def test_uses_full_name_by_default(self):
        """``QuantityMatrix`` uses its full name by default."""
        qv = QuantityMatrix(jnp.array([1.0, 2.0, 3.0]), unit=("m", "s", "kg"))
        result = wl.pformat(qv)
        assert result.startswith("QuantityMatrix")
        assert not result.startswith("QM(")

    def test_use_short_name_true(self):
        """``QuantityMatrix`` uses its short name 'QM'."""
        qv = QuantityMatrix(jnp.array([1.0, 2.0, 3.0]), unit=("m", "s", "kg"))
        result = wl.pformat(qv, use_short_name=True)
        assert result.startswith("QM(")
        assert "unit='(m, s, kg)'" in result
