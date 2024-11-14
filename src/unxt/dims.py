"""Working with dimensions.

The main features are:

- ``unxt.dims.dimension``: a function to construct a dimension object.
- ``unxt.dims.dimension_of``: a function to get the dimensions of an object.

"""

__all__ = ["dimension", "dimension_of"]

from jaxtyping import install_import_hook

from .setup_package import RUNTIME_TYPECHECKER

with install_import_hook("unxt", RUNTIME_TYPECHECKER):
    from ._src.dimensions.core import dimension, dimension_of

# Clean up the namespace
del install_import_hook
