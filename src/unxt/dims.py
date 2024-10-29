"""Dimensions in JAX.

This module provides a way to work with dimensions in JAX.

The main features are:

- `unxt.dims.dimensions`: A function to construct the dimensions of an object.
- `unxt.dims.dimensions_of`: A function to get the dimensions of an object.

"""

__all__ = ["dimensions", "dimensions_of"]

from jaxtyping import install_import_hook

from .setup_package import RUNTIME_TYPECHECKER

with install_import_hook("unxt", RUNTIME_TYPECHECKER):
    from ._src.dimensions.core import dimensions, dimensions_of

# Clean up the namespace
del install_import_hook
