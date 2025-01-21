"""Units objects in JAX.

Copyright (c) 2023 Galactic Dynamics. All rights reserved.
"""

__all__ = ["dimension", "dimension_of"]

from typing import Any, TypeAlias

import astropy.units as apyu
from plum import dispatch

AbstractDimension: TypeAlias = apyu.PhysicalType


@dispatch.abstract
def dimension(obj: Any, /) -> AbstractDimension:
    """Construct the dimension.

    .. note::

        This function uses multiple dispatch. Dispatches made in other modules
        may not be included in the rendered docs. To see the full range of
        options, execute ``unxt.dims.dimension.methods`` in an interactive
        Python session.

    """


@dispatch.abstract
def dimension_of(obj: Any, /) -> AbstractDimension:
    """Return the dimension of the given units.

    .. note::

        This function uses multiple dispatch. Dispatches made in other modules
        may not be included in the rendered docs. To see the full range of
        options, execute ``unxt.dimension_of.methods`` in an interactive Python
        session.

    """
