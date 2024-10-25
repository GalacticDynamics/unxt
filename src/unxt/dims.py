"""Dimensions."""

__all__ = ["dimensions", "dimensions_of"]

from jaxtyping import install_import_hook

from .setup_package import RUNTIME_TYPECHECKER

with install_import_hook("unxt", RUNTIME_TYPECHECKER):
    from ._src.dimensions.core import dimensions, dimensions_of

# Clean up the namespace
del install_import_hook
