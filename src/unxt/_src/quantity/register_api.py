"""Functional approach to Quantities."""

__all__: list[str] = []


from typing import Any

import equinox as eqx
from astropy.units import UnitConversionError
from jaxtyping import Array
from plum import dispatch

from .api import ustrip
from .base import AbstractQuantity
from .quantity import Quantity
from unxt._src.units import AstropyUnits
from unxt.dims import AbstractDimension, dimension_of
from unxt.units import unit
from unxt.unitsystems import AbstractUnitSystem

# ===================================================================
# Get dimensions


@dispatch
def dimension_of(obj: AbstractQuantity, /) -> AbstractDimension:
    """Return the dimension of a quantity.

    Examples
    --------
    >>> import unxt as u
    >>> q = u.Quantity(1, "m")
    >>> u.dimension_of(q)
    PhysicalType('length')

    """
    return dimension_of(obj.unit)


@dispatch
def dimension_of(obj: type[Quantity], /) -> AbstractDimension:
    """Return the dimension of a quantity.

    Examples
    --------
    >>> import unxt as u

    >>> try:
    ...     u.dimension_of(u.Quantity)
    ... except Exception as e:
    ...     print(e)
    can only get dimensions from parametrized Quantity -- Quantity[dim].

    >>> u.dimension_of(u.Quantity["length"])
    PhysicalType('length')

    """
    obj = eqx.error_if(
        obj,
        not hasattr(obj, "_type_parameter"),
        "can only get dimensions from parametrized Quantity -- Quantity[dim].",
    )
    return obj._type_parameter  # noqa: SLF001


# ===================================================================
# Get units


@dispatch
def unit_of(obj: AbstractQuantity, /) -> AstropyUnits:
    """Return the units of an object.

    Examples
    --------
    >>> from unxt import unit_of, Quantity
    >>> q = Quantity(1, "m")
    >>> unit_of(q)
    Unit("m")

    """
    return obj.unit


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
def ustrip(x: AbstractQuantity, /) -> Array:
    """Strip the units from the quantity.

    Examples
    --------
    >>> import unxt as u

    >>> q = u.Quantity(1000, "m")
    >>> u.ustrip(q)
    Array(1000, dtype=int32, weak_type=True)

    >>> u.ustrip(q) is q.value
    True

    """
    return x.value


@dispatch
def ustrip(u: AstropyUnits, x: AbstractQuantity, /) -> Array:
    """Strip the units from the quantity.

    Examples
    --------
    >>> import unxt as u

    >>> q = u.Quantity(1000, "m")
    >>> u.ustrip(u.unit("km"), q)
    Array(1., dtype=float32, ...)

    """
    return ustrip(uconvert(u, x))


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
    return ustrip(unit(u), x)


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
