"""Working with units.

The main features are:

- ``unxt.units.unit``: a function to construct units.
- ``unxt.units.unit_of``: a function to get the units of an object.

"""

__all__ = ("unit", "unit_of", "AbstractUnit")

from .setup_package import install_import_hook

with install_import_hook("unxt.units"):
    from ._src.units import AbstractUnit, unit, unit_of

# Clean up the namespace
del install_import_hook
