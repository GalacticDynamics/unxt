"""Hypothesis strategies for unxt.

This is the canonical package (unxts.hypothesis). The legacy ``unxt_hypothesis``
package continues to work via a thin backward-compatible shim.
"""

__all__ = (
    "__version__",
    "DIMENSION_NAMES",
    "derived_units",
    "units",
    "unitsystems",
    "quantities",
    "wrap_to",
    "angles",
    "named_dimensions",
)

from ._src.dimensions import DIMENSION_NAMES, named_dimensions
from ._src.quantities import angles, quantities, wrap_to
from ._src.units import derived_units, units
from ._src.unitsystems import unitsystems
from ._version import version as __version__
