# pylint: disable=import-error

"""unxt: Quantities in JAX.

Copyright (c) 2023 Galactic Dynamics. All rights reserved.
"""

__all__ = ["type_unparametrized"]

from plum.parametric import ParametricTypeMeta

from .base import AbstractQuantity


# TODO: upstream to `plum`
def type_unparametrized(q: AbstractQuantity) -> type[AbstractQuantity]:
    """Return the non-parametric type of a Quantity.

    :mod:`plum.parametric` produces parametric subtypes of Quantity.  This
    function can be used to get the original, non-parametric, Quantity type.

    Examples
    --------
    >>> from unxt import UncheckedQuantity, Quantity

    >>> q = UncheckedQuantity(1, "m")
    >>> q
    UncheckedQuantity(Array(1, dtype=int32, weak_type=True), unit='m')

    >>> type_unparametrized(q)
    <class 'unxt...UncheckedQuantity'>

    >>> q = Quantity(1, "m")
    >>> q
    Quantity['length'](Array(1, dtype=int32, weak_type=True), unit='m')

    >>> type_unparametrized(q)
    <class 'unxt...Quantity'>

    This is different from `type` for parametric types.

    >>> type(q)
    <class 'unxt...Quantity[PhysicalType('length')]'>

    """
    typ = type(q)
    return typ.mro()[1] if isinstance(typ, ParametricTypeMeta) else typ
