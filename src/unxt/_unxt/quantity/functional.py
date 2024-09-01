"""Functional approach to Quantities."""

__all__ = ["to_units"]

from typing import Any

from jaxtyping import ArrayLike
from plum import dispatch

from .base import AbstractQuantity
from .core import Quantity
from unxt._unxt.typing_ext import Unit

# ============================================================================
# to_units


@dispatch.abstract
def to_units(value: Any, units: Any, /) -> AbstractQuantity:
    """Convert a value to a Quantity with the given units.

    Parameters
    ----------
    value : Any, positional-only
        The value to convert to a Quantity.
    units : Any, positional-only
        The units to convert the value to.

    Returns
    -------
    AbstractQuantity
        The value converted to a Quantity with the given units.

    """


@dispatch  # type: ignore[no-redef]
def to_units(value: AbstractQuantity, units: Unit | str, /) -> AbstractQuantity:
    """Convert a Quantity to the given units.

    Examples
    --------
    >>> from unxt import Quantity, to_units

    >>> q = Quantity(1, "m")
    >>> to_units(q, "cm")
    Quantity['length'](Array(100., dtype=float32, ...), unit='cm')

    """
    return value.to_units(units)


@dispatch  # type: ignore[no-redef]
def to_units(value: ArrayLike, units: Unit | str, /) -> Quantity:
    """Convert a value to a Quantity with the given units.

    Examples
    --------
    >>> from unxt import to_units

    >>> to_units(1, "m")
    Quantity['length'](Array(1, dtype=int32, ...), unit='m')

    """
    return Quantity.constructor(value, units)
