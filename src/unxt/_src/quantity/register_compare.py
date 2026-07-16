"""Comparison operations for quantities.

`==` on a `StaticValue`-backed quantity is *unit-blind* (equal only when the unit
labels match -- so it is safe as a `jax.jit` ``static_argnames`` key). This module
provides :func:`equivalent`, the unit-*aware* "same physical quantity" check.
"""

__all__ = ("equivalent",)

from typing import Any

import numpy as np
from plum import dispatch

import unxt_api as uapi
from .base import AbstractQuantity
from .value import StaticValue


@dispatch  # type: ignore[misc]
def equivalent(a: AbstractQuantity, b: AbstractQuantity, /) -> Any:
    """Whether two quantities are physically equal, accounting for units.

    This is the unit-*aware* counterpart to ``==`` (which is unit-blind for
    `StaticValue`-backed quantities -- see `AbstractQuantity.__eq__`). The result
    mirrors ``==``'s shape: a scalar `bool` for `StaticValue`-backed operands, and
    an element-wise dimensionless `Quantity` of booleans for array-backed ones.
    Quantities with incompatible dimensions are never equivalent (and this never
    raises).

    Examples
    --------
    >>> import numpy as np
    >>> import unxt as u
    >>> from unxt.quantity import StaticValue

    Physically-equal static quantities in different units are *equivalent*, even
    though unit-blind ``==`` reports ``False``:

    >>> a = u.Q(StaticValue(np.array([1.0, 2.0])), "m")
    >>> b = u.Q(StaticValue(np.array([0.001, 0.002])), "km")
    >>> a == b
    False
    >>> u.equivalent(a, b)
    True
    >>> a.is_equivalent(b)
    True

    Array-backed quantities compare element-wise (unit-aware):

    >>> u.equivalent(u.Q([1.0, 2.0], "m"), u.Q([0.001, 0.009], "km"))
    Quantity(Array([ True, False], dtype=bool), unit='')

    Incompatible dimensions are never equivalent:

    >>> u.equivalent(u.Q(1.0, "m"), u.Q(1.0, "s"))
    False

    """
    # Incompatible dimensions are never equivalent (and must not raise).
    if not uapi.is_unit_convertible(b.unit, a.unit):
        return False
    # ``StaticValue``-backed ``==`` is unit-blind, so do the unit-aware compare
    # explicitly to keep the scalar-bool result.
    if isinstance(a.value, StaticValue) and isinstance(b.value, StaticValue):
        converted = uapi.uconvert_value(a.unit, b.unit, b.value.array)
        return bool(np.array_equal(a.value.array, converted))
    # Array-backed ``==`` is already unit-aware -> element-wise result.
    return a == b
