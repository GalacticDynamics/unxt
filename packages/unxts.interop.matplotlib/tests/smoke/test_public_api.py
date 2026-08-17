"""Smoke tests for the unxts.interop.matplotlib public API."""

from importlib.metadata import version

import unxts.interop.matplotlib as uim


def test_all_symbols_present():
    for name in uim.__all__:
        assert hasattr(uim, name), f"unxts.interop.matplotlib missing: {name}"


def test_version_exposed():
    assert uim.__version__ == version("unxts.interop.matplotlib")
