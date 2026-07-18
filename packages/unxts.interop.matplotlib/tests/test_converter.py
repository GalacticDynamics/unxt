"""Tests for unxts.interop.matplotlib converter registration."""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.units
import numpy as np
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


def test_default_units_from_list_of_scalar_quantities():
    """Matplotlib splits array-likes into lists of scalar quantities."""
    converter = uimpl.UnxtConverter()
    q = u.Q([-1.0, 1.0, -1.0, 1.0], "mas")
    assert converter.default_units([q[0], q[1]], None) == u.unit("mas")


def test_axisinfo_handles_none_unit():
    """Matplotlib may query axisinfo before a unit is set on the axis."""
    converter = uimpl.UnxtConverter()
    info = converter.axisinfo(None, None)
    assert isinstance(info, matplotlib.units.AxisInfo)


def test_imshow_with_quantity_extent():
    """End-to-end: a Quantity ``extent`` should plot and label the axis."""
    fig, ax = plt.subplots()
    try:
        ax.imshow(np.zeros((4, 4)), extent=u.Q([-8.0, 8.0, -8.0, 8.0], "mas"))
        fig.canvas.draw()
        assert ax.get_xlim() == (-8.0, 8.0)
        assert "mas" in ax.get_xlabel()
    finally:
        plt.close(fig)
