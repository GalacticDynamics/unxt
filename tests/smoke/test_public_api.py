"""Smoke tests for the unxt public API."""

from importlib.metadata import version

import unxt as u


def test_all_symbols_present():
    for name in u.__all__:
        assert hasattr(u, name), f"unxt missing: {name}"


def test_version_exposed():
    assert u.__version__ == version("unxt")
