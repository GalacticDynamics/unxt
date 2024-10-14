"""unxt: Quantities in JAX.

Copyright (c) 2023 Galactic Dynamics. All rights reserved.
"""

from jaxtyping import install_import_hook

from .setup_package import RUNTIME_TYPECHECKER

with install_import_hook("unxt", RUNTIME_TYPECHECKER):
    from . import quantity, unitsystems
    from ._src.dimensions.core import dimensions, dimensions_of
    from ._src.units.core import units, units_of
    from ._version import version as __version__
    from .quantity import *
    from .unitsystems import AbstractUnitSystem, unitsystem, unitsystem_of

from ._src import (
    experimental,  # noqa: F401
)

# isort: split
from . import _interop  # noqa: F401  # register interop

__all__ = [
    "__version__",
    # dimensions
    "dimensions",
    "dimensions_of",
    # units
    "units",
    "units_of",
    # units systems
    "unitsystems",  # module
    "AbstractUnitSystem",  # base class
    "unitsystem",  # convenience constructor
    "unitsystem_of",  # get the unit system
]
__all__ += quantity.__all__

# Clean up the namespace
del install_import_hook
