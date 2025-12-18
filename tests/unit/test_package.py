"""Test the package itself."""

import importlib.metadata

import unxt as u


def test_version():
    """Test version."""
    assert importlib.metadata.version("unxt") == u.__version__
