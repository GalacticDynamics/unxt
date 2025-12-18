"""Unit system utils."""

__all__: tuple[str, ...] = ()

import re
from typing import Any

from plum import dispatch

from zeroth import zeroth

from . import builtin_dimensions as bdims
from unxt.dims import AbstractDimension, dimension_of
from unxt.units import AbstractUnit

# ------------------------------------


@dispatch.abstract
def parse_dimlike_name(pt: Any, /) -> str:
    """Get the dimension name from the object."""
    raise NotImplementedError  # pragma: no cover


@dispatch
def parse_dimlike_name(pt: str, /) -> str:
    """Return the dimension name.

    Note that this does not check for the existence of that dimension.

    Examples
    --------
    >>> from unxt._src.unitsystems.utils import parse_dimlike_name

    >>> parse_dimlike_name("length")
    'length'

    >>> parse_dimlike_name("not real")
    'not_real'

    >>> try:
    ...     parse_dimlike_name("*62")
    ... except ValueError as e:
    ...     print(e)
    Input contains non-letter characters

    """
    # A regex search to match anything that's not a letter or a whitespace.
    if re.search(r"[^a-zA-Z_ ]", pt):
        msg = "Input contains non-letter characters"
        raise ValueError(msg)

    return pt.replace(" ", "_")


@dispatch
def parse_dimlike_name(pt: AbstractDimension, /) -> str:
    """Return the dimension name from a dimension.

    Examples
    --------
    >>> import unxt as u
    >>> from unxt._src.unitsystems.utils import parse_dimlike_name

    >>> parse_dimlike_name(u.dimension("length"))
    'length'

    >>> parse_dimlike_name(u.dimension("speed"))
    'speed'

    """
    # Note: this is not deterministic b/c ``_physical_type`` is a set
    #       that's why the `if` statement is needed.
    match pt:
        case bdims.speed:
            out = "speed"
        case _:
            out = parse_dimlike_name(zeroth(pt._physical_type))  # noqa: SLF001
    return out


@dispatch
def parse_dimlike_name(pt: AbstractUnit, /) -> str:
    """Return the dimension name from a unit.

    Examples
    --------
    >>> import unxt as u
    >>> from unxt._src.unitsystems.utils import parse_dimlike_name

    >>> parse_dimlike_name(u.unit("km"))
    'length'

    """
    return parse_dimlike_name(dimension_of(pt))
