"""Test the package itself."""

import importlib.metadata

import unxt as m


def test_version():
    """Test version."""
    assert importlib.metadata.version("unxt") == m.__version__
