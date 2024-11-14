"""Units objects in JAX.

Copyright (c) 2023 Galactic Dynamics. All rights reserved.
"""

__all__ = ["dimension", "dimension_of"]

from typing import Any, TypeAlias

import astropy.units as apyu
from astropy.units import Unit
from plum import dispatch

AbstractUnits: TypeAlias = apyu.UnitBase | Unit
AbstractDimension: TypeAlias = apyu.PhysicalType


# ===================================================================
# Construct the dimensions


@dispatch.abstract
def dimension(obj: Any, /) -> AbstractDimension:
    """Construct the dimension.

    .. note::

        This function uses multiple dispatch. Dispatches made in other modules
        may not be included in the rendered docs. To see the full range of
        options, execute ``unxt.dims.dimension.methods`` in an interactive
        Python session.

    """


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


@dispatch.abstract
def dimension_of(obj: Any, /) -> AbstractDimension:
    """Return the dimension of the given units.

    .. note::

        This function uses multiple dispatch. Dispatches made in other modules
        may not be included in the rendered docs. To see the full range of
        options, execute ``unxt.dimension_of.methods`` in an interactive Python
        session.

    """


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
