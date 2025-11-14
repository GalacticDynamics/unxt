"""Working with dimensions.

The main features are:

- ``unxt.dims.dimension``: a function to construct a dimension object.
- ``unxt.dims.dimension_of``: a function to get the dimensions of an object.

"""

__all__ = ("dimension", "dimension_of", "AbstractDimension")

from .setup_package import install_import_hook

with install_import_hook("unxt.dims"):
    from ._src.dimensions import AbstractDimension, dimension, dimension_of

# Clean up the namespace
del install_import_hook
