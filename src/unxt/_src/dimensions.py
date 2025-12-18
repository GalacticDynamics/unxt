"""Dimensions module.

This is the private implementation of the dimensions module.

"""

__all__ = ("AbstractDimension", "dimension", "dimension_of")

import importlib.metadata
from typing import Any, NoReturn, TypeAlias

import astropy.units as apyu
from packaging.version import Version, parse as parse_version
from plum import dispatch

AbstractDimension: TypeAlias = apyu.PhysicalType


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


@dispatch
def dimension_of(obj: type, /) -> NoReturn:
    """Get the dimension of a type.

    Examples
    --------
    >>> import unxt as u

    >>> try:
    ...     u.dimension_of(u.quantity.BareQuantity)
    ... except ValueError as e:
    ...     print(e)
    Cannot get the dimension of <class 'unxt._src.quantity.unchecked.BareQuantity'>.

    """
    msg = f"Cannot get the dimension of {obj}."
    raise ValueError(msg)


# ===================================================================
# COMPAT

ASTROPY_LT_71 = parse_version(importlib.metadata.version("astropy")) < Version("7.1")


@dispatch
def name_of(dim: AbstractDimension, /) -> str:
    """Name of a dimension.

    Examples
    --------
    >>> import unxt as u

    >>> name_of(u.dimension("length"))
    'length'

    >>> name_of(u.dimension("speed"))
    'speed'

    >>> name_of(u.dimension("mass density"))
    'mass density'

    """
    if dim == "unknown":
        ptid = dim._unit._physical_type_id  # noqa: SLF001
        name = " ".join(
            f"{unit}{power}" if power != 1 else unit for unit, power in ptid
        )

    elif ASTROPY_LT_71:
        name = dim._name_string_as_ordered_set().split("'")[1]  # noqa: SLF001
    else:
        name = dim._physical_type[0]  # noqa: SLF001

    return name
