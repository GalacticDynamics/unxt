"""Backward-compatible shim for unxt_hypothesis.

The canonical package is now ``unxts.hypothesis``
(install: ``pip install unxts.hypothesis``).
This package re-exports the complete public API and will be maintained long-term.
No changes are required in code that imports ``unxt_hypothesis``.
"""

from unxts.hypothesis import (
    DIMENSION_NAMES,
    angles,
    derived_units,
    named_dimensions,
    quantities,
    units,
    unitsystems,
    wrap_to,
)

__all__ = (
    "DIMENSION_NAMES",
    "derived_units",
    "units",
    "unitsystems",
    "quantities",
    "wrap_to",
    "angles",
    "named_dimensions",
)
