"""Test the package itself."""

import contextlib
import importlib
import importlib.metadata

import pytest

import unxt as u
from unxt import setup_package


def test_version():
    """Test version."""
    assert importlib.metadata.version("unxt") == u.__version__


def test_experimental_is_public_importable_module():
    """``unxt.experimental`` is a real, importable public module.

    Regression test: ``experimental`` must be listed in ``unxt.__all__`` and
    ``import unxt.experimental`` must succeed (not just ``from unxt import
    experimental``), which requires a genuine ``unxt/experimental.py`` module
    rather than a mere top-level attribute.
    """
    assert "experimental" in u.__all__

    experimental = importlib.import_module("unxt.experimental")
    assert experimental is u.experimental
    assert set(experimental.__all__) == {"grad", "hessian", "jacfwd", "where"}


@pytest.mark.parametrize(
    ("env", "expected"),
    [("False", False), ("None", None), ("beartype.beartype", "beartype.beartype")],
)
def test_runtime_typechecker_from_env(monkeypatch, env, expected):
    """``UNXT_ENABLE_RUNTIME_TYPECHECKING`` selects the jaxtyping typechecker."""
    monkeypatch.setenv("UNXT_ENABLE_RUNTIME_TYPECHECKING", env)
    try:
        module = importlib.reload(setup_package)
        assert expected == module.RUNTIME_TYPECHECKER
        # `False` means "no hook"; anything else installs a jaxtyping one.
        hook = module.install_import_hook("unxt")
        assert isinstance(hook, contextlib.nullcontext) is (expected is False)
    finally:
        monkeypatch.undo()
        importlib.reload(setup_package)
