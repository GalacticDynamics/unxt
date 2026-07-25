"""Quantities in JAX."""

from .angle import *
from .base import *
from .base_angle import *
from .flag import *
from .quantity import *
from .static_quantity import *
from .value import *

# isort: split
from .register_api import *
from .register_compare import *
from .register_conversions import *
from .register_dispatches import *
from .register_primitives import *
from .register_ufuncs import *

# isort: split
from ._deprecation import deprecated_getattr


def __getattr__(name: str) -> object:
    return deprecated_getattr(name, __name__, Quantity)  # noqa: F405
