"""Functional approach to Quantities."""

__all__: list[str] = []


from typing import Any

from astropy.units import UnitConversionError
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


# ============================================================================


@dispatch
def is_unit_convertible(to_unit: Any, from_: Any, /) -> bool:
    """Check if a unit can be converted to another unit.

    Parameters
    ----------
    to_unit : Any
        The unit to convert to. Converted to a unit object using `unxt.unit`.
    from_ : Any
        The unit to convert from. Converted to a unit object using `unxt.unit`,
        Note this means it also support `Quantity` objects and many others.

    Examples
    --------
    >>> from unxt import is_unit_convertible
    >>> is_unit_convertible("cm", "m")
    True

    >>> is_unit_convertible("m", "Gyr")
    False

    """
    to_u = unit(to_unit)
    from_u = unit(from_)
    try:
        from_u.to(to_u)
    except UnitConversionError:
        return False
    return True


@dispatch
def is_unit_convertible(to_unit: Any, from_: AbstractQuantity, /) -> bool:
    """Check if a Quantity can be converted to another unit.

    Examples
    --------
    >>> from unxt import Quantity, is_unit_convertible
    >>> q = Quantity(1, "m")

    >>> is_unit_convertible("cm", q)
    True

    >>> is_unit_convertible("Gyr", q)
    False

    """
    return is_unit_convertible(to_unit, from_.unit)
