"""Tests for unxts.interop.matplotlib converter registration."""

import matplotlib.units
import unxts.interop.matplotlib as uimpl

import unxt as u


def test_converter_registered_on_import():
    assert u.quantity.AbstractQuantity in matplotlib.units.registry
    assert isinstance(
        matplotlib.units.registry[u.quantity.AbstractQuantity], uimpl.UnxtConverter
    )


def test_converter_strips_units():
    converter = uimpl.UnxtConverter()
    q = u.Q([1.0, 2.0, 3.0], "km")
    converted = list(converter.convert(q, u.unit("m"), axis=None))
    assert converted == [1000.0, 2000.0, 3000.0]
