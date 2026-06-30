"""xarray integration for unxt.

Copyright (c) 2023 Galactic Dynamics. All rights reserved.
"""

__all__ = (
    "attach_units",
    "extract_unit_attributes",
    "extract_units",
    "strip_units",
)

# Import accessors to register them with xarray
from unxt_xarray._src import accessors  # noqa: F401
from unxt_xarray._src.conversion import (
    attach_units,
    extract_unit_attributes,
    extract_units,
    strip_units,
)
