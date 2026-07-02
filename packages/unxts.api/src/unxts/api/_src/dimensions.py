"""Dimension API for unxt.

Copyright (c) 2023 Galactic Dynamics. All rights reserved.
"""

__all__ = ("dimension", "dimension_of")

from typing import Any

import plum


@plum.dispatch.abstract
def dimension(obj: Any, /) -> Any:
    """Construct the dimension.

    .. note::

        This function uses multiple dispatch. Dispatches made in other modules
        may not be included in the rendered docs. To see the full range of
        options, execute ``unxt.dims.dimension.methods`` in an interactive
        Python session.

    """


@plum.dispatch.abstract
def dimension_of(obj: Any, /) -> Any:
    """Return the dimension of the given units.

    .. note::

        This function uses multiple dispatch. Dispatches made in other modules
        may not be included in the rendered docs. To see the full range of
        options, execute ``unxt.dimension_of.methods`` in an interactive Python
        session.

    """
