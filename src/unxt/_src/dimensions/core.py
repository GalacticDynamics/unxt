"""Units objects in JAX.

Copyright (c) 2023 Galactic Dynamics. All rights reserved.
"""

__all__: list[str] = []

from typing import Any

import astropy.units as apyu
from plum import dispatch

from .api import AbstractDimension

# ===================================================================
# Construct the dimensions


@dispatch
def dimension(obj: AbstractDimension, /) -> AbstractDimension:
    """Construct dimension from a dimension object.

    Examples
    --------
    >>> import unxt as u
    >>> import astropy.units as apyu

    >>> length = apyu.get_physical_type("length")
    >>> length
    PhysicalType('length')

    >>> u.dimension(length) is length
    True

    """
    return obj


@dispatch
def dimension(obj: str, /) -> AbstractDimension:
    """Construct dimension from a string.

    Examples
    --------
    >>> from unxt.dims import dimension
    >>> dimension("length")
    PhysicalType('length')

    """
    return apyu.get_physical_type(obj)


# ===================================================================
# Get the dimension


@dispatch
def dimension_of(obj: Any, /) -> None:
    """Most objects have no dimension.

    Examples
    --------
    >>> from unxt.dims import dimension_of

    >>> print(dimension_of(1))
    None

    >>> print(dimension_of("length"))
    None

    """
    return None  # noqa: RET501


@dispatch
def dimension_of(obj: AbstractDimension, /) -> AbstractDimension:
    """Return the dimension of the given units.

    Examples
    --------
    >>> from unxt.dims import dimension, dimension_of

    >>> dimension_of(dimension("length"))
    PhysicalType('length')

    """
    return obj
