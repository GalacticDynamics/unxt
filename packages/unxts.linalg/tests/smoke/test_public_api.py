"""Public-API surface tests for `unxts.linalg`."""

from importlib.metadata import version

import unxts.linalg as ul


def test_all_symbols_importable():
    for name in ul.__all__:
        assert hasattr(ul, name), f"unxts.linalg missing: {name}"


def test_version_exposed():
    assert "__version__" in ul.__all__
    assert ul.__version__ == version("unxts.linalg")
