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


def test_default_units_for_iterable_of_quantities():
    """`default_units` reports the unit for a list/tuple of quantities.

    Regression: the iterable branch probed pint's ``.units`` attribute, but
    unxt quantities expose ``.unit`` -- so it returned ``None`` and matplotlib
    could not resolve the axis unit (crashing e.g. ``ax.set_xlim(Q, Q)``).
    """
    converter = uimpl.UnxtConverter()
    # Bare quantity already worked.
    assert converter.default_units(u.Q(1.0, "m"), None) == u.unit("m")
    # A list / tuple of quantities must resolve the same unit.
    assert converter.default_units([u.Q(1.0, "m"), u.Q(2.0, "m")], None) == u.unit("m")
    assert converter.default_units((u.Q(1.0, "m"),), None) == u.unit("m")
    # A plain (unitless) value still yields None.
    assert converter.default_units(5.0, None) is None


def test_axisinfo_tolerates_none_unit():
    """`axisinfo(None, ...)` returns a bare AxisInfo rather than dereferencing it."""
    converter = uimpl.UnxtConverter()
    info = converter.axisinfo(None, None)
    assert isinstance(info, matplotlib.units.AxisInfo)
