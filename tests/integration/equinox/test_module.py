"""Tests."""

import equinox as eqx

import unxt as u


def quantity_as_module_field():
    """Test a Quantity as a `equinox.Module` field."""

    class TestModule(eqx.Module):
        """Test module."""

        field: u.Quantity = u.Quantity(1.0, "m")

    assert TestModule().field == u.Quantity(1.0, "m")
