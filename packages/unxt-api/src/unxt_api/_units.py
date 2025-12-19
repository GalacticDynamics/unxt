"""Units API for unxt.

Copyright (c) 2023 Galactic Dynamics. All rights reserved.
"""

__all__ = ("unit", "unit_of")

from typing import Any

import plum


@plum.dispatch.abstract
def unit(obj: Any, /) -> Any:
    """Construct the units from a units object."""


@plum.dispatch.abstract
def unit_of(obj: Any, /) -> Any:
    """Return the units of an object."""
