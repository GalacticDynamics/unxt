"""Working with units.

The main features are:

- ``unxt.units.unit``: a function to construct units.
- ``unxt.units.unit_of``: a function to get the units of an object.

"""

__all__ = ["unit", "unit_of"]

from jaxtyping import install_import_hook

from .setup_package import RUNTIME_TYPECHECKER

with install_import_hook("unxt", RUNTIME_TYPECHECKER):
    from ._src.units.core import unit, unit_of

# Clean up the namespace
del install_import_hook, RUNTIME_TYPECHECKER
