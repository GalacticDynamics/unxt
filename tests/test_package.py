"""Test the package itself."""

import importlib.metadata

import jax_quantity as m


def test_version():
    """Test version."""
    assert importlib.metadata.version("jax_quantity") == m.__version__
