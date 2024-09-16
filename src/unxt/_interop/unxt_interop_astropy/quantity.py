"""Unitsystem compatibility."""

__all__: list[str] = []

from typing import Any, TypeAlias

import astropy.units as apyu
from astropy.coordinates import Angle as AstropyAngle, Distance as AstropyDistance
from astropy.units import Quantity as AstropyQuantity
from plum import conversion_method, dispatch, type_unparametrized as type_up

import quaxed.numpy as jnp
from dataclassish import field_items, replace

from unxt.dims import dimension_of
from unxt.quantity import AbstractQuantity, Quantity, UncheckedQuantity

# ============================================================================
# AbstractQuantity

# -----------------
# Constructor


@AbstractQuantity.from_.dispatch
def from_(
    cls: type[AbstractQuantity], value: AstropyQuantity, /, **kwargs: Any
) -> AbstractQuantity:
    """Construct a `Quantity` from another `Quantity`.

    The `value` is converted to the new `unit`.

    Examples
    --------
    >>> import unxt as u
    >>> import astropy.units as apyu

    >>> u.Quantity.from_(apyu.Quantity(1, "m"))
    Quantity['length'](Array(1., dtype=float32), unit='m')

    """
    return cls(jnp.asarray(value.value, **kwargs), value.unit)


@AbstractQuantity.from_.dispatch  # type: ignore[no-redef]
def from_(
    cls: type[AbstractQuantity], value: AstropyQuantity, unit: Any, /, **kwargs: Any
) -> AbstractQuantity:
    """Construct a `Quantity` from another `Quantity`.

    The `value` is converted to the new `unit`.

    Examples
    --------
    >>> import unxt as u
    >>> import astropy.units as apyu

    >>> u.Quantity.from_(apyu.Quantity(1, "m"), "cm")
    Quantity['length'](Array(100., dtype=float32), unit='cm')

    """
    return cls(jnp.asarray(value.to_value(unit), **kwargs), unit)


# -----------------
# Conversion Methods


@conversion_method(type_from=AbstractQuantity, type_to=AstropyQuantity)  # type: ignore[misc]
def convert_unxt_quantity_to_astropy_quantity(
    q: AbstractQuantity, /
) -> AstropyQuantity:
    """Convert a `unxt.AbstractQuantity` to a `astropy.units.Quantity`.

    Examples
    --------
    >>> import unxt as u
    >>> from astropy.units import Quantity as AstropyQuantity
    >>> from plum import convert

    >>> convert(u.Quantity(1.0, "cm"), AstropyQuantity)
    <Quantity 1. cm>

    """
    return AstropyQuantity(q.value, q.unit)


@conversion_method(type_from=AbstractQuantity, type_to=AstropyDistance)  # type: ignore[misc]
def convert_unxt_quantity_to_astropy_distance(
    q: AbstractQuantity, /
) -> AstropyDistance:
    """Convert a `unxt.AbstractQuantity` to a `astropy.coordinates.Distance`.

    Examples
    --------
    >>> import unxt as u
    >>> from astropy.coordinates import Distance as AstropyDistance
    >>> from plum import convert

    >>> convert(u.Quantity(1.0, "cm"), AstropyDistance)
    <Distance 1. cm>

    """
    return AstropyDistance(q.value, q.unit)


@conversion_method(type_from=AbstractQuantity, type_to=AstropyAngle)  # type: ignore[misc]
def convert_unxt_quantity_to_astropy_angle(q: AbstractQuantity, /) -> AstropyAngle:
    """Convert a `unxt.quantity.AbstractQuantity` to a `astropy.coordinates.Angle`.

    Examples
    --------
    >>> import unxt as u
    >>> from astropy.coordinates import Angle as AstropyAngle
    >>> from plum import convert

    >>> convert(u.Quantity(1.0, "radian"), AstropyAngle)
    <Angle 1. rad>

    """
    return AstropyAngle(q.value, q.unit)


# ============================================================================
# Quantity


@conversion_method(type_from=AstropyQuantity, type_to=Quantity)  # type: ignore[misc]
def convert_astropy_quantity_to_unxt_quantity(q: AstropyQuantity, /) -> Quantity:
    """Convert a `astropy.units.Quantity` to a `unxt.Quantity`.

    Examples
    --------
    >>> from astropy.units import Quantity as AstropyQuantity
    >>> from plum import convert
    >>> from unxt import Quantity

    >>> convert(AstropyQuantity(1.0, "cm"), Quantity)
    Quantity['length'](Array(1., dtype=float32), unit='cm')

    """
    return Quantity(q.value, q.unit)


# ============================================================================
# UncheckedQuantity


@conversion_method(type_from=AstropyQuantity, type_to=UncheckedQuantity)  # type: ignore[misc]
def convert_astropy_quantity_to_unxt_uncheckedquantity(
    q: AstropyQuantity, /
) -> UncheckedQuantity:
    """Convert a `astropy.units.Quantity` to a `unxt.UncheckedQuantity`.

    Examples
    --------
    >>> from astropy.units import Quantity as AstropyQuantity
    >>> from plum import convert
    >>> from unxt.quantity import UncheckedQuantity

    >>> convert(AstropyQuantity(1.0, "cm"), UncheckedQuantity)
    UncheckedQuantity(Array(1., dtype=float32), unit='cm')

    """
    return UncheckedQuantity(q.value, q.unit)


###############################################################################


AstropyUnit: TypeAlias = (
    apyu.UnitBase | apyu.Unit | apyu.FunctionUnitBase | apyu.StructuredUnit
)


@dispatch  # type: ignore[misc]
def uconvert(unit: AstropyUnit, x: AbstractQuantity, /) -> AbstractQuantity:
    """Convert the quantity to the specified units.

    Examples
    --------
    >>> import astropy.units as apyu
    >>> import unxt as u

    >>> x = u.Quantity(1000, "m")
    >>> u.uconvert(u.unit("km"), x)
    Quantity['length'](Array(1., dtype=float32, ...), unit='km')

    >>> x = u.Quantity([1, 2, 3], "Kelvin")
    >>> with apyu.add_enabled_equivalencies(apyu.temperature()):
    ...     y = x.uconvert("deg_C")
    >>> y
    Quantity['temperature'](Array([-272.15, -271.15, -270.15], dtype=float32, ...), unit='deg_C')

    >>> x = Quantity([1, 2, 3], "radian")
    >>> with apyu.add_enabled_equivalencies(apyu.dimensionless_angles()):
    ...     y = x.uconvert("")
    >>> y
    Quantity['dimensionless'](Array([1., 2., 3.], dtype=float32, ...), unit='')

    """  # noqa: E501
    # Hot-path: if no unit conversion is necessary
    if x.unit == unit:
        return x

    # Compute the value. Used in all subsequent branches.
    value = x.unit.to(unit, x.value)

    # If the dimensions are the same, we can just replace the value and unit.
    if dimension_of(x.unit) == dimension_of(unit):
        return replace(x, value=value, unit=unit)

    # If the dimensions are different, we need to create a new quantity since
    # the dimensions can be part of the type. This won't work if the Quantity
    # type itself needs to be changed, e.g. `coordinax.Angle` ->
    # `unxt.Quantity`. These cases are handled separately, in other dispatches.
    fs = dict(field_items(x))
    fs["value"] = value
    fs["unit"] = unit

    return type_up(x)(**fs)
