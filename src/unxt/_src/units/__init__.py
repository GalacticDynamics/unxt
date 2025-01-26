"""Units module.

Copyright (c) 2023 Galactic Dynamics. All rights reserved.
"""

__all__ = ["unit", "unit_of", "AstropyUnits"]

from .api import (
    AstropyUnits,  # TODO: remove
    unit,
    unit_of,
)

# Register dispatches
# isort: split
from . import register_dispatches  # noqa: F401
