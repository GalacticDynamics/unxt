"""Dimensions module.

This is the private implementation of the dimensions module.

"""

__all__ = ["AbstractDimension", "dimension", "dimension_of"]

from .api import AbstractDimension, dimension, dimension_of

# Register the dispatches
# isort: split
from . import core  # noqa: F401
