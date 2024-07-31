"""unxt: Quantities in JAX.

Copyright (c) 2023 Galactic Dynamics. All rights reserved.
"""

from . import (
    _quantity,
    experimental,  # noqa: F401
    unitsystems,
)
from ._optional_deps import HAS_ASTROPY, HAS_GALA, HAS_MATPLOTLIB
from ._quantity import *
from ._version import version as __version__
from .unitsystems import AbstractUnitSystem, UnitSystem, unitsystem

# Register interoperability
# isort: split
if HAS_ASTROPY:
    from ._interop import unxt_interop_astropy

    del unxt_interop_astropy
if HAS_GALA:
    from ._interop import unxt_interop_gala

    del unxt_interop_gala
if HAS_MATPLOTLIB:
    from ._interop import unxt_interop_mpl as interop_mpl

    interop_mpl.setup_matplotlib_support_for_unxt(enable=True)

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
del HAS_ASTROPY, HAS_GALA, HAS_MATPLOTLIB, _quantity
