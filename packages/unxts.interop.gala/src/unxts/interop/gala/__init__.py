"""gala integration for unxt.

This is the canonical package (unxts.interop.gala). It provides conversions
between `gala.units.UnitSystem` and `unxt.unitsystems.AbstractUnitSystem`,
exposed as the public functions `convert_gala_unitsystem_to_unxt_unitsystem`
and `convert_unxt_unitsystem_to_gala_unitsystem`.

Prefer `plum.convert` over calling these functions directly -- importing
this module registers them with `plum` as conversion methods, so
`plum.convert(usys, AbstractUnitSystem)` and
`plum.convert(usys, gala.units.UnitSystem)` are the idiomatic way to convert
between the two unit system types.
"""

__all__ = (
    "convert_gala_unitsystem_to_unxt_unitsystem",
    "convert_unxt_unitsystem_to_gala_unitsystem",
)

from ._src import (
    convert_gala_unitsystem_to_unxt_unitsystem,
    convert_unxt_unitsystem_to_gala_unitsystem,
)
from ._version import version as __version__  # noqa: F401
