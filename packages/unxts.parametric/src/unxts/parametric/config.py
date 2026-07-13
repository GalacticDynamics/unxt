"""Public configuration API for ``unxts.parametric``."""
# The public re-export mirrors ``_src.config``'s ``__all__`` by design.
# pylint: disable=duplicate-code

__all__ = (
    "ParametricConfig",
    "ParametricQuantityReprConfig",
    "ParametricQuantityStrConfig",
    "config",
)

from ._src.config import (
    ParametricConfig,
    ParametricQuantityReprConfig,
    ParametricQuantityStrConfig,
    config,
)
