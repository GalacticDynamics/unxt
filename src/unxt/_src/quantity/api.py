"""Functional approach to Quantities."""

__all__ = ["uconvert", "ustrip"]

from typing import Any

from plum import dispatch

# ===================================================================
# Convert units


@dispatch.abstract  # type: ignore[misc]
def uconvert(u: Any, x: Any, /) -> Any:
    """Convert the quantity to the specified units.

    Examples
    --------
    >>> from unxt import Quantity, uconvert, units

    >>> q = Quantity(1, "km")
    >>> uconvert(units("m"), q)
    Quantity['length'](Array(1000., dtype=float32, ...), unit='m')

    >>> uconvert("m", q)
    Quantity['length'](Array(1000., dtype=float32, ...), unit='m')

    For further examples, see the other method dispatches.

    """
    raise NotImplementedError  # pragma: no cover


# ===================================================================


@dispatch.abstract  # type: ignore[misc]
def ustrip(u: Any, x: Any, /) -> Any:
    """Strip the units from the quantity, first converting if necessary.

    Examples
    --------
    >>> from unxt import Quantity, uconvert, units

    >>> q = Quantity(1, "km")
    >>> ustrip(units("m"), q)
    Array(1000., dtype=float32, ...)

    >>> ustrip("m", q)
    Array(1000., dtype=float32, ...)

    For further examples, see the other method dispatches.

    """
    raise NotImplementedError  # pragma: no cover
