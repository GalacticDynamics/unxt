"""Smoke tests for the unxts.hypothesis public API."""

import unxts.hypothesis


def test_all_symbols_present():
    for name in unxts.hypothesis.__all__:
        assert hasattr(unxts.hypothesis, name), f"unxts.hypothesis missing: {name}"


def test_version_is_a_string():
    assert isinstance(unxts.hypothesis.__version__, str)
