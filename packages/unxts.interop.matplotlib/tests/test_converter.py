"""Tests for unxts.interop.matplotlib converter registration."""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.units
import numpy as np
import pytest
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


def test_default_units_from_a_quantity():
    """A quantity carries its own unit; no unwrapping needed."""
    converter = uimpl.UnxtConverter()
    assert converter.default_units(u.Q([1.0, 2.0], "mas"), None) == u.unit("mas")


def test_default_units_of_a_bare_scalar_is_none():
    """A plain scalar has no unit and is not iterable."""
    converter = uimpl.UnxtConverter()
    assert converter.default_units(1.0, None) is None


def test_setup_can_be_disabled_and_re_enabled():
    """Disabling removes the converter from matplotlib's registry."""
    try:
        uimpl.setup_matplotlib_support_for_unxt(enable=False)
        assert u.quantity.AbstractQuantity not in matplotlib.units.registry
    finally:
        uimpl.setup_matplotlib_support_for_unxt()
    assert u.quantity.AbstractQuantity in matplotlib.units.registry


def test_unit_format_default_latex_inline():
    """Default unit_format produces latex_inline format labels."""
    converter = uimpl.UnxtConverter()
    unit = u.unit("m/s")
    info = converter.axisinfo(unit, None)
    # latex_inline format should produce something like $\mathrm{m\,s^{-1}}$
    assert info.label == unit.to_string("latex_inline")
    assert "$" in info.label
    assert "mathrm" in info.label


def test_unit_format_custom_format():
    """Custom unit_format respects the specified format."""
    converter = uimpl.UnxtConverter(unit_format="latex")
    unit = u.unit("m/s")
    info = converter.axisinfo(unit, None)
    # latex format should produce something like $\mathrm{\frac{m}{s}}$
    assert info.label == unit.to_string("latex")
    assert "frac" in info.label


def test_unit_format_console_format():
    """unit_format='console' produces plain text format."""
    converter = uimpl.UnxtConverter(unit_format="console")
    unit = u.unit("m/s")
    info = converter.axisinfo(unit, None)
    # console format should produce plain text like "m / s"
    assert info.label == unit.to_string("console")
    assert "$" not in info.label


def test_axisinfo_kw_deprecated():
    """Using axisinfo_kw triggers a deprecation warning."""
    with pytest.warns(DeprecationWarning, match=r"axisinfo_kw.*deprecated"):
        converter = uimpl.UnxtConverter(axisinfo_kw={"format": "latex"})
    # Verify the format was correctly applied
    assert converter.unit_format == "latex"


def test_axisinfo_kw_backwards_compatibility():
    """axisinfo_kw={'format': ...} still works and produces correct output."""
    with pytest.warns(DeprecationWarning, match=r"axisinfo_kw.*deprecated"):
        converter = uimpl.UnxtConverter(axisinfo_kw={"format": "latex"})

    unit = u.unit("m/s")
    info = converter.axisinfo(unit, None)
    # Should use the format from axisinfo_kw
    assert info.label == unit.to_string("latex")
    assert "frac" in info.label


def test_axisinfo_kw_empty_dict():
    """axisinfo_kw={} uses the default unit_format."""
    with pytest.warns(DeprecationWarning, match=r"axisinfo_kw.*deprecated"):
        converter = uimpl.UnxtConverter(axisinfo_kw={})
    # Should still use default unit_format since no "format" key
    assert converter.unit_format == "latex_inline"
