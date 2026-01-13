"""Quantity API for unxt.

Copyright (c) 2023 Galactic Dynamics. All rights reserved.
"""

__all__ = ("uconvert_value", "uconvert", "ustrip", "is_unit_convertible", "wrap_to")

from typing import Any

import plum


@plum.dispatch.abstract
def uconvert_value(uto: Any, ufrom: Any, x: Any, /) -> Any:
    """Convert the value from specified units to specified units.

    General signature: ``(to_unit, from_unit, value) -> converted_value``.
    Other signatures are defined via method dispatch.
    See ``uconvert_value.methods`` for details.

    Examples
    --------
    >>> import unxt as u  # implements `unxt_api.uconvert_value`

    >>> u.uconvert_value(u.unit("m"), u.unit("km"), 1)
    1000.0

    >>> u.uconvert_value("m", "km", 1)
    1000.0

    For further examples, see the other method dispatches.

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def uconvert(u: Any, x: Any, /) -> Any:
    """Convert the quantity to the specified units.

    General signature: ``(to_unit, quantity) -> converted_quantity``.
    Other signatures are defined via method dispatch.
    See ``uconvert.methods`` for details.

    Internally, {func}`unxt_api.uconvert` often calls to
    {func}`unxt_api.uconvert_value` to perform the numerical conversion on the
    Quantity's value.

    Examples
    --------
    >>> import unxt as u  # implements `unxt_api.uconvert`

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

    General signature: ``(to_unit, quantity) -> value_array``.
    Other signatures are defined via method dispatch.
    See ``ustrip.methods`` for details.

    Examples
    --------
    >>> import unxt as u  # implements `unxt_api.ustrip`

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
    """Check if the units are convertible.

    General signature is ``(to_unit, from_unit) -> bool``.
    Other signatures are defined via method dispatch.
    See ``is_unit_convertible.methods`` for details.

    Examples
    --------
    >>> import unxt as u  # implements `unxt_api.is_unit_convertible`
    >>> u.is_unit_convertible(u.unit("m"), u.unit("km"))
    True

    >>> u.is_unit_convertible(u.unit("m"), u.unit("s"))
    False

    """
    return False


@plum.dispatch.abstract
def wrap_to(x: Any, min: Any, max: Any, /) -> Any:
    """Wrap to the range [min, max).

    General signature: ``(x, min, max) -> wrapped_x``.
    Other signatures are defined via method dispatch.
    See ``wrap_to.methods`` for details.

    Examples
    --------
    >>> import unxt as u
    >>>
    >>> angle = u.Angle(370, "deg")
    >>> min, max = u.Q(0, "deg"), u.Q(360, "deg")
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
