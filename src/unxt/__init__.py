"""``unxt``: Quantities in JAX.

``unxt`` is a library for working with physical quantities in JAX, supporting
JAX's autodiff and JIT compilation, and easy integration with existing codes. If
you're seeing this then you're in the main module of the ``unxt`` package, where
we provide exports for the main functionality of the library. Sub-modules are
available for more specialized functionality, such as unit systems and
experimental features.

Note that `unxt` uses multiple-dispatch to provide a flexible and extensible
interface. In the docs you'll see the function signatures without type
annotation and then subsections for specific function dispatches based on the
type annotations. However dispatches registered from other modules may not be
included in the rendered docs. To see all the dispatches execute ``<func or
class>.methods`` in an interactive Python session. For more information on
multiple-dispatch see the [`plum`](https://beartype.github.io/plum/intro.html)
documentation.

-----

"""

__all__ = (
    "__version__",
    # dimensions
    "dims",  # module
    "AbstractDimension",  # dimension type
    "dimension",  # convenience constructor
    "dimension_of",  # get the dimensions
    # units
    "units",  # module
    "AbstractUnit",  # unit type
    "unit",
    "unit_of",
    # units systems
    "unitsystems",  # module
    "AbstractUnitSystem",  # base class
    "unitsystem",  # convenience constructor
    "unitsystem_of",  # get the unit system
    # quantities
    "quantity",  # module
    "Quantity",  # main class
    "Q",  # convenience alias
    "AbstractQuantity",  # base class
    "Angle",  # angular quantity
    "uconvert",  # convert units
    "ustrip",  # strip units
    "is_unit_convertible",  # check if units can be converted
)

from .setup_package import install_import_hook

with install_import_hook("unxt"):
    from . import dims, quantity, units, unitsystems
    from ._version import version as __version__
    from .dims import AbstractDimension, dimension, dimension_of
    from .quantity import (
        AbstractQuantity,
        Angle,
        Q,
        Quantity,
        is_unit_convertible,
        uconvert,
        ustrip,
    )
    from .units import AbstractUnit, unit, unit_of
    from .unitsystems import AbstractUnitSystem, unitsystem, unitsystem_of

from ._src import experimental  # noqa: F401

# isort: split
from . import _interop  # noqa: F401  # register interop

# Clean up the namespace
del install_import_hook
