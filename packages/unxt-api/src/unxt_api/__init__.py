"""Backward-compatible shim for unxt_api.

The canonical package is now ``unxts.api`` (install: ``pip install unxts.api``).
This package re-exports the complete public API and will be maintained long-term.
No changes are required in code that imports ``unxt_api``.
"""

from unxts.api import (
    __version__,
    dimension,
    dimension_of,
    is_unit_convertible,
    uconvert,
    uconvert_value,
    unit,
    unit_of,
    unitsystem_of,
    ustrip,
    wrap_to,
)

__all__ = (
    "__version__",
    "dimension",
    "dimension_of",
    "uconvert_value",
    "uconvert",
    "ustrip",
    "is_unit_convertible",
    "wrap_to",
    "unit",
    "unit_of",
    "unitsystem_of",
)
