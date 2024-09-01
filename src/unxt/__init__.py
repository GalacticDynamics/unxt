"""unxt: Quantities in JAX.

Copyright (c) 2023 Galactic Dynamics. All rights reserved.
"""

from . import unitsystems
from ._unxt import (
    experimental,  # noqa: F401
    quantity,
)
from ._unxt.quantity import *
from ._unxt.units import units, units_of
from ._version import version as __version__
from .unitsystems import AbstractUnitSystem, unitsystem, unitsystem_of

# isort: split
from . import _interop  # noqa: F401  # register interop

__all__ = [
    "__version__",
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

# Clean up namespace
del quantity
