"""Systems of units.

A unit system is a collection of units that are used together. In a unit system
there are base units and derived units. Base units are the fundamental units of
the system and derived units are constructed from the base units. For example,
in the SI system, the base units are the meter, kilogram, second, ampere,
kelvin, mole, and candela. Derived units are constructed from these base units,
for example, the newton is a derived unit of force.

`unxt` provides powerful tools for defining and working with unit systems. Unit
systems can be statically defined (useful for many tools and development
environments) or dynamically defined (useful for interactive environments and
Python's general dynamism). Unit systems can be extended, compared, and used for
decomposing units on quantities. There are many more features and tools for
working with unit systems in `unxt`.

"""

from ._src.units.system import base, builtin, compare, core, flags, realizations, utils
from ._src.units.system.base import *  # noqa: F403
from ._src.units.system.builtin import *  # noqa: F403
from ._src.units.system.compare import *  # noqa: F403
from ._src.units.system.core import *  # noqa: F403
from ._src.units.system.flags import *  # noqa: F403
from ._src.units.system.realizations import *  # noqa: F403
from ._src.units.system.utils import *  # noqa: F403

__all__: list[str] = []
__all__ += base.__all__
__all__ += core.__all__
__all__ += builtin.__all__
__all__ += realizations.__all__
__all__ += compare.__all__
__all__ += utils.__all__
__all__ += flags.__all__

# Clean up namespace
del base, builtin, compare, core, utils, realizations, flags
