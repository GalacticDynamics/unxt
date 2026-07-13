"""Tests for deprecated aliases kept for the v1 -> v2 transition."""

import pytest

import unxt as u


def test_barequantity_is_deprecated_alias():
    """`BareQuantity` warns and resolves to the new default `Quantity`."""
    with pytest.warns(DeprecationWarning, match="renamed to `Quantity`"):
        from unxt.quantity import BareQuantity  # noqa: PLC0415

    assert BareQuantity is u.Quantity


def test_barequantity_attribute_access_warns():
    """Attribute access (not just import) also warns."""
    import unxt.quantity as uq  # noqa: PLC0415

    with pytest.warns(DeprecationWarning, match="renamed to `Quantity`"):
        cls = uq.BareQuantity
    assert cls is u.Quantity


def test_barequantity_not_in_public_all():
    """The deprecated name is not advertised via __all__ / star-import."""
    import unxt.quantity as uq  # noqa: PLC0415

    assert "BareQuantity" not in uq.__all__


def test_unknown_attribute_raises():
    import unxt.quantity as uq  # noqa: PLC0415

    with pytest.raises(AttributeError, match="NotAThing"):
        _ = uq.NotAThing
