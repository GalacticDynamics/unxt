"""Smoke tests for the unxts.parametric public API."""

import unxts.parametric


def test_all_symbols_present():
    for name in unxts.parametric.__all__:
        assert hasattr(unxts.parametric, name), f"unxts.parametric missing: {name}"
