# pylint: disable=import-error, too-many-lines
# ruff:noqa: E402

"""Test the Array API."""

import equinox as eqx
import pytest

import quaxed.numpy as qnp

from unxt import Quantity

# =============================================================================
# Constants


def test_allclose():
    """Test `e`."""
    q = Quantity(100.0, "m")

    match = "Cannot add a non-quantity and quantity"
    with pytest.raises(eqx.EquinoxTracetimeError, match=match):
        assert qnp.allclose(q, Quantity(0.1, "km"))

    # Need the `atol` argument.
    assert qnp.allclose(q, Quantity(0.1, "km"), atol=Quantity(1e-6, "m"))
