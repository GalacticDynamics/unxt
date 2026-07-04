"""Quantities in JAX."""
# The ``BareQuantity`` deprecation shim in ``__getattr__`` imports ``warnings``
# lazily on purpose; silence the module-level lint for that intentional pattern.
# pylint: disable=import-outside-toplevel,redefined-outer-name

from .angle import *
from .base import *
from .base_angle import *
from .base_parametric import *
from .flag import *
from .parametric import *
from .quantity import *
from .static_quantity import *
from .value import *

# isort: split
from .register_api import *
from .register_conversions import *
from .register_dispatches import *
from .register_primitives import *
from .register_ufuncs import *


def __getattr__(name: str) -> object:
    if name == "BareQuantity":
        import warnings  # noqa: PLC0415

        warnings.warn(
            "`BareQuantity` has been renamed to `Quantity` and is now the "
            "default quantity class (unxt v2). The parametric class formerly "
            "named `Quantity` is now `ParametricQuantity`. `BareQuantity` "
            "will be removed in a future release.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return Quantity  # noqa: F405
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
