"""unxt: Quantities in JAX.

Copyright (c) 2023 Galactic Dynamics. All rights reserved.
"""

from . import unitsystems
from ._unxt import (
    experimental,  # noqa: F401
    quantity,
)
from ._unxt.quantity import *
from ._version import version as __version__
from .unitsystems import AbstractUnitSystem, unitsystem

# isort: split
from . import _interop  # noqa: F401  # register interop

__all__ = [
    "__version__",
    # units systems
    "unitsystems",  # module
    "AbstractUnitSystem",  # base class
    "unitsystem",  # convenience constructor
]
__all__ += quantity.__all__

# Clean up namespace
del quantity
