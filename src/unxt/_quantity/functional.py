"""Functional approach to Quantities."""

__all__ = ["to_units", "to_units_value"]

from typing import Any

from astropy.units import Quantity as AstropyQuantity
from jax.typing import ArrayLike
from plum import dispatch

from .base import AbstractQuantity
from .core import Quantity
from unxt._typing import Unit

# ============================================================================
# to_units


@dispatch.abstract
def to_units(value: Any, units: Unit | str, /) -> AbstractQuantity:
    """Convert a value to a Quantity with the given units.

    Parameters
    ----------
    value : Any, positional-only
        The value to convert to a Quantity.
    units : Unit, positional-only
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


# ---------------------------
# Compat


@dispatch  # type: ignore[no-redef]
def to_units(value: AstropyQuantity, units: Unit, /) -> Quantity:
    """Convert an Astropy Quantity to the given units.

    Examples
    --------
    >>> from unxt import to_units
    >>> import astropy.units as u

    >>> q = u.Quantity(1, "m")
    >>> to_units(q, "cm")
    Quantity['length'](Array(100., dtype=float32), unit='cm')

    """
    return Quantity.constructor(value, units)


# ============================================================================
# to_units_value


@dispatch.abstract
def to_units_value(value: Any, units: Unit | str, /) -> ArrayLike:
    """Convert a value to an array with the given units.

    Parameters
    ----------
    value : Any, positional-only
        The value to convert to an array.
    units : Unit | str, positional-only
        The units to convert the value to.

    Returns
    -------
    ArrayLike
        The value converted to an array with the given units.

    """


@dispatch  # type: ignore[no-redef]
def to_units_value(value: AbstractQuantity, units: Unit | str, /) -> ArrayLike:
    """Convert a Quantity to an array with the given units.

    Examples
    --------
    >>> from unxt import Quantity, to_units_value

    >>> q = Quantity(1, "m")
    >>> to_units_value(q, "cm")
    Array(100., dtype=float32, ...)

    """
    return value.to_units_value(units)


@dispatch  # type: ignore[no-redef]
def to_units_value(value: ArrayLike, units: Unit | str, /) -> ArrayLike:
    """Convert a value to an array with the given units.

    Examples
    --------
    >>> from unxt import to_units_value

    >>> to_units_value(1, "m")
    1

    """
    return value


# ---------------------------
# Compat


@dispatch  # type: ignore[no-redef]
def to_units_value(value: AstropyQuantity, units: Unit | str, /) -> ArrayLike:
    """Convert an Astropy Quantity to an array with the given units.

    Examples
    --------
    >>> from unxt import to_units_value
    >>> import astropy.units as u

    >>> q = u.Quantity(1, "m")
    >>> to_units_value(q, "cm")
    np.float64(100.0)

    """
    return value.to_value(units)
