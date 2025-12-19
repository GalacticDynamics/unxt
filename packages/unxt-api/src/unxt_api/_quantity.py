"""Quantity API for unxt.

Copyright (c) 2023 Galactic Dynamics. All rights reserved.
"""

__all__ = ("uconvert", "ustrip", "is_unit_convertible", "wrap_to")

from typing import Any

import plum


@plum.dispatch.abstract
def uconvert(u: Any, x: Any, /) -> Any:
    """Convert the quantity to the specified units.

    Examples
    --------
    >>> import unxt as u

    >>> q = u.Quantity(1, "km")
    >>> u.uconvert(u.unit("m"), q)
    Quantity(Array(1000., dtype=float32, ...), unit='m')

    >>> u.uconvert("m", q)
    Quantity(Array(1000., dtype=float32, ...), unit='m')

    For further examples, see the other method dispatches.

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
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


@plum.dispatch.abstract
def is_unit_convertible(to_unit: Any, from_: Any, /) -> bool:
    """Check if the units are convertible."""
    return False


@plum.dispatch.abstract
def wrap_to(x: Any, min: Any, max: Any, /) -> Any:
    """Wrap to the range [min, max).

    Examples
    --------
    >>> import unxt as u
    >>>
    >>> angle = u.Angle(370, "deg")
    >>> min, max = u.Quantity(0, "deg"), u.Quantity(360, "deg")
    >>>
    >>> u.quantity.wrap_to(angle, min, max)
    Angle(Array(10, dtype=int32, ...), unit='deg')
    >>>
    >>> u.quantity.wrap_to(angle, min=min, max=max)
    Angle(Array(10, dtype=int32, ...), unit='deg')

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch
def wrap_to(x: Any, /, *, min: Any, max: Any) -> Any:
    """Wrap to the range [min, max)."""
    return wrap_to(x, min, max)  # redirect to the standard method
