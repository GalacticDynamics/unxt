# pylint: disable=import-error

"""Copyright (c) 2023 Nathaniel Starkman. All rights reserved.

jax-quantity: Quantities in JAX
"""

from . import _core
from ._core import *
from ._version import version as __version__

# isort: split
# Register dispatches
from . import _register_dispatches, _register_primitives  # noqa: F401

__all__ = ["__version__"]
__all__ += _core.__all__
