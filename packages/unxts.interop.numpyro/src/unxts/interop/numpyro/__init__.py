"""NumPyro integration for unxt.

This is the canonical package (unxts.interop.numpyro).
"""

from ._src import patches as _patches
from ._version import version as __version__  # noqa: F401

_patches.install()

__all__ = ()
