"""Tests."""

import re

import jax.numpy as jnp
import jax.random as jr
import pytest
from quax.examples import lora

import unxt as u


def test_lora_array_as_quantity_value():
    lora_array = lora.LoraArray(jnp.asarray([[1.0, 2, 3]]), rank=1, key=jr.key(0))
    with pytest.warns(
        UserWarning, match=re.escape("'quax.ArrayValue' subclass 'LoraArray'")
    ):
        quantity = u.Quantity(lora_array, "m")

    assert quantity.value is lora_array
    assert quantity.unit == "m"
