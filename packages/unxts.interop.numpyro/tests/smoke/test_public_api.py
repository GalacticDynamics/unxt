"""Smoke tests for the unxts.interop.numpyro public API."""

import unxts.interop.numpyro as unp


def test_all_symbols_present():
    for name in unp.__all__:
        assert hasattr(unp, name), f"unxts.interop.numpyro missing: {name}"


def test_version():
    assert isinstance(unp.__version__, str)
