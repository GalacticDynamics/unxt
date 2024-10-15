"""Unit system utils."""

__all__: list[str] = []

import re
from typing import Any

from astropy.units import PhysicalType as Dimension
from plum import dispatch

from . import builtin_dimensions as bdims
from unxt._src.dimensions.core import dimensions_of
from unxt._src.typing_ext import Unit

# ------------------------------------


@dispatch.abstract
def get_dimension_name(pt: Any, /) -> str:
    """Get the dimension name from the object."""
    raise NotImplementedError  # pragma: no cover


@dispatch  # type: ignore[no-redef]
def get_dimension_name(pt: str, /) -> str:
    """Return the dimension name.

    Note that this does not check for the existence of that dimension.

    Examples
    --------
    >>> from unxt._src.units.system.utils import get_dimension_name

    >>> get_dimension_name("length")
    'length'

    >>> get_dimension_name("not real")
    'not_real'

    >>> try: get_dimension_name("*62")
    ... except ValueError as e: print(e)
    Input contains non-letter characters

    """
    # A regex search to match anything that's not a letter or a whitespace.
    if re.search(r"[^a-zA-Z_ ]", pt):
        msg = "Input contains non-letter characters"
        raise ValueError(msg)

    return pt.replace(" ", "_")


@dispatch  # type: ignore[no-redef]
def get_dimension_name(pt: Dimension, /) -> str:
    """Return the dimension name from a dimension.

    Examples
    --------
    >>> from unxt._src.units.system.utils import get_dimension_name
    >>> import astropy.units as u
    >>> get_dimension_name(u.get_physical_type("length"))
    'length'

    >>> get_dimension_name(u.get_physical_type("speed"))
    'speed'

    """
    # Note: this is not deterministic b/c ``_physical_type`` is a set
    #       that's why the `if` statement is needed.
    match pt:
        case bdims.speed:
            out = "speed"
        case _:
            out = get_dimension_name(next(iter(pt._physical_type)))  # noqa: SLF001
    return out


@dispatch  # type: ignore[no-redef]
def get_dimension_name(pt: Unit, /) -> str:
    """Return the dimension name from a unit.

    Examples
    --------
    >>> from unxt._src.units.system.utils import get_dimension_name
    >>> import astropy.units as u
    >>> get_dimension_name(u.km)
    'length'

    """
    return get_dimension_name(dimensions_of(pt))
