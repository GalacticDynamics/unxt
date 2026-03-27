"""Functional approach to Quantities."""

__all__: tuple[str, ...] = ()

from dataclasses import replace
from typing import Any

import equinox as eqx
import numpy as np
from astropy.units import UnitConversionError
from jaxtyping import Array, ArrayLike
from plum import dispatch

import unxt_api as uapi
from .base import AbstractQuantity
from .base_angle import AbstractAngle
from .base_parametric import AbstractParametricQuantity
from .static_quantity import StaticQuantity
from .value import StaticValue
from unxt.dims import AbstractDimension
from unxt.units import AbstractUnit
from unxt.unitsystems import AbstractUnitSystem

# ===================================================================
# Get dimensions


@dispatch
def dimension_of(obj: AbstractQuantity, /) -> AbstractDimension:
    """Return the dimension of a quantity.

    Examples
    --------
    >>> import unxt as u
    >>> q = u.Q(1, "m")
    >>> u.dimension_of(q)
    PhysicalType('length')

    """
    return uapi.dimension_of(obj.unit)


@dispatch
def dimension_of(obj: type[AbstractParametricQuantity], /) -> AbstractDimension:
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
        f"can only get dimensions from parametrized {obj.__name__} "
        f"-- {obj.__name__}[dim].",
    )
    return obj._type_parameter  # noqa: SLF001


@dispatch
def dimension_of(obj: type[AbstractAngle], /) -> AbstractDimension:
    """Get the dimension of an angle class.

    Examples
    --------
    >>> import unxt as u

    >>> u.dimension_of(u.Angle)
    PhysicalType('angle')

    """
    return uapi.dimension("angle")


# ===================================================================
# Get units


@dispatch
def unit_of(obj: AbstractQuantity, /) -> AbstractUnit:
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
# Convert units for values


@dispatch
def uconvert_value(
    tousys: AbstractUnitSystem, ufrom: AbstractUnit, x: ArrayLike, /
) -> ArrayLike:
    """Convert the value from units to a unitsystem's preferred units.

    Examples
    --------
    >>> import unxt as u
    >>> u.uconvert_value(u.unitsystems.galactic, u.unit("km"), 1e17)  # kpc
    3.2407792894443648

    >>> u.unitsystems.galactic[u.dimension("length")]  # checking the units
    Unit("kpc")

    """
    return uapi.uconvert_value(tousys[uapi.dimension_of(ufrom)], ufrom, x)


@dispatch
def uconvert_value(
    tousys: AbstractUnitSystem, ufrom: str, x: ArrayLike, /
) -> ArrayLike:
    """Convert the value from units to a unitsystem's preferred units.

    Examples
    --------
    >>> import unxt as u
    >>> u.uconvert_value(u.unitsystems.galactic, "km", 1e17)  # in kpc
    3.2407792894443648

    >>> u.unitsystems.galactic[u.dimension("length")]  # checking the units
    Unit("kpc")

    """
    return uapi.uconvert_value(tousys, uapi.unit(ufrom), x)


@dispatch
def uconvert_value(uto: str, ufrom: str, x: ArrayLike, /) -> ArrayLike:
    """Convert the value to the specified units.

    Examples
    --------
    >>> import unxt as u

    >>> u.uconvert_value("m", "km", 1)
    1000.0

    """
    return uapi.uconvert_value(uapi.unit(uto), uapi.unit(ufrom), x)


# **NOTE:** we also add convenience dispatches for `uconvert_value` so that
# users can use the lower-level function with Quantity and not break their code.


@dispatch
def uconvert_value(uto: Any, ufrom: Any, x: AbstractQuantity, /) -> AbstractQuantity:
    """Convert the quantity to the specified units.

    This is a convenience dispatch so that users can use the lower-level
    function with Quantity and not break their code.
    This dispatch simply calls `uconvert`, checking first that the units are
    convertible.

    Examples
    --------
    >>> import unxt as u

    >>> q = u.Q(1, "km")
    >>> u.uconvert_value("m", "km", q)
    Quantity(Array(1000., dtype=float32, ...), unit='m')

    """
    x = eqx.error_if(
        x,
        not uapi.is_unit_convertible(ufrom, x.unit),
        f"Cannot convert from {x.unit} to {ufrom}",
    )
    return uapi.uconvert(uto, x)


# ===================================================================
# Convert units for quantities


@dispatch
def uconvert(ustr: str, x: AbstractQuantity, /) -> AbstractQuantity:
    """Convert the quantity to the specified units.

    Examples
    --------
    >>> from unxt import Quantity, units

    >>> x = Quantity(1000, "m")
    >>> uconvert("km", x)
    Quantity(Array(1., dtype=float32, ...), unit='km')

    """
    return uconvert(uapi.unit(ustr), x)


@dispatch
def uconvert(usys: AbstractUnitSystem, x: AbstractQuantity, /) -> AbstractQuantity:
    """Convert the quantity to the specified units.

    Examples
    --------
    >>> from unxt import Quantity, units
    >>> from unxt.unitsystems import galactic

    >>> q = Quantity(1e17, "km")
    >>> uconvert(galactic, q)
    Quantity(Array(3.2407792, dtype=float32, ...), unit='kpc')

    """
    return uconvert(usys[uapi.dimension_of(x)], x)


# ===================================================================
# Strip units


@dispatch
def ustrip(x: StaticQuantity, /) -> np.ndarray:
    """Strip the units from a static quantity."""
    return x.value.array


@dispatch
def ustrip(x: AbstractQuantity, /) -> Array | np.ndarray:
    """Strip the units from the quantity.

    Examples
    --------
    >>> import unxt as u

    >>> q = u.Q(1000, "m")
    >>> u.ustrip(q)
    Array(1000, dtype=int32, weak_type=True)

    >>> u.ustrip(q) is q.value
    True

    """
    v = x.value
    return v.array if isinstance(v, StaticValue) else v


@dispatch
def ustrip(u: AbstractUnit, x: AbstractQuantity, /) -> Array | np.ndarray:
    """Strip the units from the quantity.

    Examples
    --------
    >>> import unxt as u

    >>> q = u.Q(1000, "m")
    >>> u.ustrip(u.unit("km"), q)
    Array(1., dtype=float32, ...)

    """
    return uapi.ustrip(uapi.uconvert(u, x))


@dispatch
def ustrip(u: str, x: AbstractQuantity, /) -> Array | np.ndarray:
    """Strip the units from the quantity.

    Examples
    --------
    >>> import unxt as u

    >>> q = u.Q(1000, "m")
    >>> u.ustrip("km", q)
    Array(1., dtype=float32, ...)

    """
    return uapi.ustrip(uapi.unit(u), x)


@dispatch
def ustrip(u: AbstractUnitSystem, x: AbstractQuantity, /) -> Array | np.ndarray:
    """Strip the units from the quantity.

    Examples
    --------
    >>> import unxt as u
    >>> from unxt.unitsystems import galactic

    >>> q = u.Q(1e17, "km")
    >>> u.ustrip(galactic, q)
    Array(3.2407792, dtype=float32, weak_type=True)

    """
    return uapi.ustrip(u[uapi.dimension_of(x)], x)


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
    to_u = uapi.unit(to_unit)
    from_u = uapi.unit(from_)
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


# ============================================================================


@dispatch
def wrap_to(
    angle: AbstractQuantity,
    min: AbstractQuantity,
    max: AbstractQuantity,
    /,
) -> AbstractQuantity:
    """Wrap to the range [min, max).

    Examples
    --------
    >>> import unxt as u

    >>> angle = u.Angle(370, "deg")
    >>> u.quantity.wrap_to(angle, min=u.Q(0, "deg"), max=u.Q(360, "deg"))
    Angle(Array(10, dtype=int32, ...), unit='deg')

    """
    minv, maxv = min.ustrip(angle.unit), max.ustrip(angle.unit)
    value = ((angle.value - minv) % (maxv - minv)) + minv
    return replace(angle, value=value)
