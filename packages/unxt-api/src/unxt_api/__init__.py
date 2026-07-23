"""Backward-compatible shim for unxt_api.

The canonical package is now ``unxts.api`` (install: ``pip install unxts.api``).
This package re-exports the complete public API and will be maintained long-term.
No changes are required in code that imports ``unxt_api``.
"""

from importlib.metadata import version as _dist_version

from unxts.api import (
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

#: Version of the ``unxt-api`` distribution (not ``unxts.api``).
__version__ = _dist_version("unxt-api")

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
