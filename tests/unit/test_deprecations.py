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


def test_barequantity_top_level_is_deprecated_alias():
    """`from unxt import BareQuantity` warns and resolves to `Quantity`.

    The top-level shim must match `unxt.quantity`'s (which was already handled).
    """
    with pytest.warns(DeprecationWarning, match="renamed to `Quantity`"):
        from unxt import BareQuantity  # noqa: PLC0415

    assert BareQuantity is u.Quantity


def test_barequantity_top_level_attribute_access_warns():
    """Attribute access on the top-level module also warns."""
    with pytest.warns(DeprecationWarning, match="renamed to `Quantity`"):
        cls = u.BareQuantity
    assert cls is u.Quantity


def test_unknown_attribute_raises():
    import unxt.quantity as uq  # noqa: PLC0415

    with pytest.raises(AttributeError, match="NotAThing"):
        _ = uq.NotAThing


def test_unknown_top_level_attribute_raises():
    with pytest.raises(AttributeError, match="NotAThing"):
        _ = u.NotAThing
