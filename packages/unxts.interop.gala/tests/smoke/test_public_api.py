"""Smoke tests for the unxts.interop.gala public API."""

from importlib.metadata import version

import unxts.interop.gala as uig


def test_all_symbols_present():
    for name in uig.__all__:
        assert hasattr(uig, name), f"unxts.interop.gala missing: {name}"


def test_version_exposed():
    assert uig.__version__ == version("unxts.interop.gala")
