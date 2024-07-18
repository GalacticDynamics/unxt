"""unxt: Quantities in JAX.

Copyright (c) 2023 Galactic Dynamics. All rights reserved.
"""

from . import (
    _quantity,
    experimental,  # noqa: F401
    unitsystems,
)
from ._optional_deps import HAS_ASTROPY, HAS_GALA
from ._quantity import *
from ._version import version as __version__
from .unitsystems import AbstractUnitSystem, UnitSystem, unitsystem

# Register interoperability
# isort: split
if HAS_ASTROPY:
    from ._interop import unxt_interop_astropy as _
if HAS_GALA:
    from ._interop import unxt_interop_gala as _  # type: ignore[no-redef]  # noqa: F401

__all__ = [
    "__version__",
    # units systems
    "unitsystems",  # module
    "AbstractUnitSystem",  # base class
    "UnitSystem",  # main user-facing class
    "unitsystem",  # convenience constructor
]
__all__ += _quantity.__all__

# Clean up namespace
del HAS_ASTROPY, HAS_GALA, _quantity
