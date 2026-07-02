"""Smoke tests for the unxts.hypothesis public API."""

import unxts.hypothesis
from hypothesis import given

import unxt as u


def test_all_symbols_present():
    for name in unxts.hypothesis.__all__:
        assert hasattr(unxts.hypothesis, name), f"unxts.hypothesis missing: {name}"


def test_version_is_a_string():
    assert isinstance(unxts.hypothesis.__version__, str)


@given(q=unxts.hypothesis.quantities(unit="km/s"))
def test_a_strategy_produces_a_quantity(q):
    assert q.unit == u.unit("km/s")
