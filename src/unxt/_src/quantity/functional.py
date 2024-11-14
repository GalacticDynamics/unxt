"""Functional approach to Quantities."""

__all__: list[str] = []


from jaxtyping import Array
from plum import dispatch

from .base import AbstractQuantity
from unxt._src.dimensions.core import dimension_of
from unxt._src.units.core import AbstractUnits, unit
from unxt._src.units.system.base import AbstractUnitSystem

# ===================================================================
# Convert units


@dispatch
def uconvert(ustr: str, x: AbstractQuantity, /) -> AbstractQuantity:
    """Convert the quantity to the specified units.

    Examples
    --------
    >>> from unxt import Quantity, units

    >>> x = Quantity(1000, "m")
    >>> uconvert("km", x)
    Quantity['length'](Array(1., dtype=float32, ...), unit='km')

    """
    return uconvert(unit(ustr), x)


@dispatch
def uconvert(usys: AbstractUnitSystem, x: AbstractQuantity, /) -> AbstractQuantity:
    """Strip the units from the quantity.

    Examples
    --------
    >>> from unxt import Quantity, units
    >>> from unxt.unitsystems import galactic

    >>> q = Quantity(1e17, "km")
    >>> uconvert(galactic, q)
    Quantity['length'](Array(3.2407792, dtype=float32, ...), unit='kpc')

    """
    return uconvert(usys[dimension_of(x)], x)


# ===================================================================
# Strip units


@dispatch
def ustrip(u: AbstractUnits, x: AbstractQuantity, /) -> Array:
    """Strip the units from the quantity.

    Examples
    --------
    >>> import unxt as u

    >>> q = u.Quantity(1000, "m")
    >>> u.ustrip(u.unit("km"), q)
    Array(1., dtype=float32, ...)

    """
    return uconvert(u, x).value


@dispatch
def ustrip(u: str, x: AbstractQuantity, /) -> Array:
    """Strip the units from the quantity.

    Examples
    --------
    >>> import unxt as u

    >>> q = u.Quantity(1000, "m")
    >>> u.ustrip("km", q)
    Array(1., dtype=float32, ...)

    """
    return uconvert(unit(u), x).value


@dispatch
def ustrip(u: AbstractUnitSystem, x: AbstractQuantity, /) -> Array:
    """Strip the units from the quantity.

    Examples
    --------
    >>> import unxt as u
    >>> from unxt.unitsystems import galactic

    >>> q = u.Quantity(1e17, "km")
    >>> u.ustrip(galactic, q)
    Array(3.2407792, dtype=float32, weak_type=True)

    """
    return ustrip(u[dimension_of(x)], x)
