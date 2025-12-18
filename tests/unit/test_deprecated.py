"""Test that `UncheckedQuantity` functions but is deprecated."""

import pytest

import quaxed.numpy as jnp

from unxt.quantity import BareQuantity, UncheckedQuantity


def test_unchecked_quantity_deprecated():
    with pytest.warns(DeprecationWarning, match="Use `BareQuantity` instead."):
        UncheckedQuantity(1, "m")


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_uncheckedquantity_works_like_barequantity():
    # Test creation
    uq1 = UncheckedQuantity(1, "m")
    bq1 = BareQuantity(1, "m")
    assert uq1.value == bq1.value
    assert uq1.unit == bq1.unit

    # Test addition
    uq2 = UncheckedQuantity(2, "m")
    bq2 = BareQuantity(2, "m")
    addu = uq1 + uq2
    addb = bq1 + bq2
    assert addu.value == addb.value
    assert addu.unit == addb.unit

    # Test multiplication
    mulu = uq1 * 2
    mulb = bq1 * 2
    assert mulu.value == mulb.value
    assert mulu.unit == mulb.unit

    # Test numpy operations
    addu = jnp.add(uq1, uq2)
    addb = jnp.add(bq1, bq2)
    assert jnp.array_equal(addu.value, addb.value)
    assert addu.unit == addb.unit
