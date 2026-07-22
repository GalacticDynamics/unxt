"""Smoke tests for the unxts.parametric public API."""

import importlib.resources

import unxts.parametric


def test_all_symbols_present():
    for name in unxts.parametric.__all__:
        assert hasattr(unxts.parametric, name), f"unxts.parametric missing: {name}"


def test_exposes_version():
    """The package must expose a non-empty ``__version__`` string.

    Every sibling ``unxts.*`` package exposes ``__version__``; this one did not.
    """
    assert isinstance(unxts.parametric.__version__, str)
    assert unxts.parametric.__version__
    assert "__version__" in unxts.parametric.__all__


def test_ships_py_typed_marker():
    """The package must ship a ``py.typed`` marker (it declares ``Typing :: Typed``)."""
    marker = importlib.resources.files("unxts.parametric") / "py.typed"
    assert marker.is_file()
