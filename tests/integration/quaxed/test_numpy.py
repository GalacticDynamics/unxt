# pylint: disable=import-error, too-many-lines
# ruff:noqa: E402

"""Test the Array API."""

import pytest

import quaxed.numpy as jnp

from unxt import Quantity

# =============================================================================
# Constants


def test_allclose():
    """Test `e`."""
    q = Quantity(100.0, "m")

    match = "Physical type mismatch."
    with pytest.raises(ValueError, match=match):
        assert jnp.allclose(q, Quantity(0.1, "km"))

    # Need the `atol` argument.
    assert jnp.allclose(q, Quantity(0.1, "km"), atol=Quantity(1e-6, "m"))
