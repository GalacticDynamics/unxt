"""Units objects in JAX.

Copyright (c) 2023 Galactic Dynamics. All rights reserved.
"""

__all__ = ["dimensions", "dimensions_of"]

from typing import Any, TypeAlias

import astropy.units as u
from astropy.units import Unit
from plum import dispatch

AbstractUnits: TypeAlias = u.UnitBase | Unit
AbstractDimensions: TypeAlias = u.PhysicalType


# ===================================================================
# Construct the dimensions


@dispatch.abstract
def dimensions(obj: Any, /) -> AbstractDimensions:
    """Construct the dimensions.

    .. note::

        This function uses multiple dispatch. Dispatches made in other modules
        may not be included in the rendered docs. To see the full range of
        options, execute ``unxt.dims.dimensions.methods`` in an interactive
        Python session.

    """


@dispatch
def dimensions(obj: AbstractDimensions, /) -> AbstractDimensions:
    """Construct dimensions from a dimensions object.

    Examples
    --------
    >>> from astropy import units as u
    >>> from unxt.dims import dimensions

    >>> length = u.get_physical_type("length")
    >>> length
    PhysicalType('length')

    >>> dimensions(length) is length
    True

    """
    return obj


@dispatch
def dimensions(obj: str, /) -> AbstractDimensions:
    """Construct dimensions from a string.

    Examples
    --------
    >>> from unxt.dims import dimensions
    >>> dimensions("length")
    PhysicalType('length')

    """
    return u.get_physical_type(obj)


# ===================================================================
# Get the dimensions


@dispatch.abstract
def dimensions_of(obj: Any, /) -> AbstractDimensions:
    """Return the dimensions of the given units.

    .. note::

        This function uses multiple dispatch. Dispatches made in other modules
        may not be included in the rendered docs. To see the full range of
        options, execute ``unxt.dimensions_of.methods`` in an interactive Python
        session.

    """


@dispatch
def dimensions_of(obj: Any, /) -> None:
    """Most objects have no dimensions.

    Examples
    --------
    >>> from unxt.dims import dimensions_of

    >>> print(dimensions_of(1))
    None

    >>> print(dimensions_of("length"))
    None

    """
    return None  # noqa: RET501


@dispatch
def dimensions_of(obj: AbstractDimensions, /) -> AbstractDimensions:
    """Return the dimensions of the given units.

    Examples
    --------
    >>> from astropy import units as u
    >>> from unxt.dims import dimensions, dimensions_of

    >>> dimensions_of(dimensions("length"))
    PhysicalType('length')

    """
    return obj
