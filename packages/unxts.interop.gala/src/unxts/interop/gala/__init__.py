"""gala integration for unxt.

This is the canonical package (unxts.interop.gala). It provides conversions
between `gala.units.UnitSystem` and `unxt.unitsystems.AbstractUnitSystem`.
"""

__all__: tuple[str, ...] = ()

from ._src import unitsystems as _unitsystems  # noqa: F401 -- registers dispatches
from ._version import version as __version__  # noqa: F401
