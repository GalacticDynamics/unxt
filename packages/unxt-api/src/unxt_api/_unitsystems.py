"""Unit systems API for unxt.

Copyright (c) 2023 Galactic Dynamics. All rights reserved.
"""

__all__ = ("unitsystem_of",)

from typing import Any

import plum


@plum.dispatch.abstract
def unitsystem_of(obj: Any, /) -> Any:
    """Return the unit system of an object."""
