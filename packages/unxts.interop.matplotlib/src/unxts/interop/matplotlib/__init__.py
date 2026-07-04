"""matplotlib integration for unxt.

This is the canonical package (unxts.interop.matplotlib). It registers a
`matplotlib.units.ConversionInterface` so `unxt.Quantity` objects can be
plotted directly with `matplotlib`.
"""

__all__ = ("UnxtConverter", "setup_matplotlib_support_for_unxt")

from ._src.converter import UnxtConverter, setup_matplotlib_support_for_unxt
from ._version import version as __version__  # noqa: F401

setup_matplotlib_support_for_unxt(enable=True)
