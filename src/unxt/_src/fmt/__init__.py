"""String formatting: `wadler_lindig_parts` plus `unxt`'s layer over it.

The engine lives in `wadler_lindig_parts`, which knows nothing about `unxt`;
`axes` teaches it the axes `unxt` needs. Importing this package registers those
axes, so the grammar is complete by the time anything formats -- which is why
every module in `unxt` reaches the engine through here rather than importing
`wadler_lindig_parts` directly.

Only the package's flat public surface is re-exported: it has no
private-but-documented tier, so nothing here reaches for an underscore name
across the distribution boundary.
"""

import wadler_lindig_parts as _wlp
from wadler_lindig_parts import *

from . import axes as _axes
from .axes import *

# The engine's own example types are its documentation, not `unxt`'s surface.
del examples

__all__ = [  # noqa: PLE0604
    *(name for name in _wlp.__all__ if name != "examples"),
    *_axes.__all__,
]
