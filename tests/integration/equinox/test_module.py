"""Tests."""

import equinox as eqx

import unxt as u


def test_quantity_as_module_field():
    """Test a Quantity as an `equinox.Module` field."""

    class TestModule(eqx.Module):
        """Test module."""

        field: u.Q = u.Q(1.0, "m")

    assert TestModule().field == u.Q(1.0, "m")
