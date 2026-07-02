"""xarray integration for unxt.

This is the canonical package (unxts.interop.xarray).
The legacy ``unxt-xarray`` package is deprecated; migrate to
``unxts.interop.xarray``.
"""

__all__ = (
    "attach_units",
    "extract_unit_attributes",
    "extract_units",
    "strip_units",
)

from ._src import accessors as _accessors  # noqa: F401 -- registers .unxt accessor
from ._src.conversion import (
    attach_units,
    extract_unit_attributes,
    extract_units,
    strip_units,
)
from ._version import version as __version__  # noqa: F401
