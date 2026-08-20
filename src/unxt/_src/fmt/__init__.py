"""String formatting: a domain-agnostic engine plus `unxt`'s layer over it.

`engine` is self-contained and knows nothing about `unxt`; `axes` teaches it
the axes `unxt` needs. Importing this package registers those axes, so the
grammar is complete by the time anything formats. See `engine` for the seam.
"""

from . import axes as _axes, engine as _engine
from .axes import *
from .engine import *
from .engine import pspec

__all__ = [  # noqa: PLE0604
    *_engine.__all__,
    *_axes.__all__,
    "pspec",
]
