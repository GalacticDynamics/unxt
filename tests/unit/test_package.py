"""Test the package itself."""

import importlib
import importlib.metadata

import unxt as u


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
