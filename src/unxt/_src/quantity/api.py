"""Functional approach to Quantities."""

__all__ = ["uconvert", "ustrip", "is_unit_convertible"]

from typing import Any

from plum import dispatch


@dispatch.abstract
def uconvert(u: Any, x: Any, /) -> Any:
    """Convert the quantity to the specified units.

    Examples
    --------
    >>> import unxt as u

    >>> q = u.Quantity(1, "km")
    >>> u.uconvert(u.unit("m"), q)
    Quantity['length'](Array(1000., dtype=float32, ...), unit='m')

    >>> u.uconvert("m", q)
    Quantity['length'](Array(1000., dtype=float32, ...), unit='m')

    For further examples, see the other method dispatches.

    """
    raise NotImplementedError  # pragma: no cover


@dispatch.abstract
def ustrip(*args: Any) -> Any:
    """Strip the units from the quantity, first converting if necessary.

    Examples
    --------
    >>> import unxt as u

    >>> q = u.Quantity(1, "km")
    >>> ustrip(u.unit("m"), q)
    Array(1000., dtype=float32, ...)

    >>> u.ustrip("m", q)
    Array(1000., dtype=float32, ...)

    For further examples, see the other method dispatches.

    """
    raise NotImplementedError  # pragma: no cover


@dispatch.abstract
def is_unit_convertible(to_unit: Any, from_: Any, /) -> bool:
    """Check if the units are convertible."""
    return False
