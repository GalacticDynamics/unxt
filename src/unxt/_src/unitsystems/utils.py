"""Unit system utils."""

__all__: tuple[str, ...] = ()

import re
from typing import Any

from plum import dispatch

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
    # A dimension can carry several equivalent physical-type aliases (e.g.
    # ``speed`` / ``velocity``), held in an unordered ``_physical_type``. Pick
    # the alphabetically-first alias so the name -- and hence the field order
    # and class identity of any dynamically generated unit system -- is
    # deterministic across runs and ``PYTHONHASHSEED`` (``min`` also happens to
    # yield ``"speed"`` over ``"velocity"``).
    return parse_dimlike_name(min(pt._physical_type))  # noqa: SLF001


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
