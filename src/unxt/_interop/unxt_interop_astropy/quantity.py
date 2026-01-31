"""Quantity compatibility."""

__all__: tuple[str, ...] = ()

import dataclasses
from typing import Any, NoReturn

import plum
from astropy.coordinates import Angle as AstropyAngle, Distance as AstropyDistance
from astropy.units import Quantity as AstropyQuantity
from jaxtyping import ArrayLike

import dataclassish as dc
import quaxed.numpy as jnp

import unxt_api as uapi
from .custom_types import APYUnits
from unxt.quantity import AbstractQuantity, AllowValue, BareQuantity, Quantity, ustrip

# ============================================================================
# Value Converter


@plum.dispatch
def convert_to_quantity_value(obj: AstropyQuantity, /) -> NoReturn:
    """Disallow conversion of `AstropyQuantity` to a value.

    >>> import astropy.units as apyu
    >>> from unxt.quantity import convert_to_quantity_value

    >>> try:
    ...     convert_to_quantity_value(apyu.Quantity(1, "m"))
    ... except TypeError as e:
    ...     print(e)
    Cannot convert 'Quantity' to a value.
    For a Quantity, use the `.from_` constructor instead.

    """
    msg = (
        f"Cannot convert {type(obj).__name__!r} to a value. "
        "For a Quantity, use the `.from_` constructor instead."
    )
    raise TypeError(msg)


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
    Quantity(Array(1., dtype=float32), unit='m')

    """
    u = uapi.unit_of(value)
    value = jnp.asarray(uapi.ustrip(u, value), **kwargs)
    return cls(value, u)


@AbstractQuantity.from_.dispatch
def from_(
    cls: type[AbstractQuantity], value: AstropyQuantity, u: Any, /, **kwargs: Any
) -> AbstractQuantity:
    """Construct a `Quantity` from another `Quantity`.

    The `value` is converted to the new `unit`.

    Examples
    --------
    >>> import unxt as u
    >>> import astropy.units as apyu

    >>> u.Quantity.from_(apyu.Quantity(1, "m"), "cm")
    Quantity(Array(100., dtype=float32), unit='cm')

    """
    u = uapi.unit(u)
    value = jnp.asarray(uapi.ustrip(u, value), **kwargs)
    return cls(value, u)


# -----------------
# Conversion Methods


@plum.conversion_method(type_from=AbstractQuantity, type_to=AstropyQuantity)  # type: ignore[arg-type]
def convert_unxt_quantity_to_astropy_quantity(
    q: AbstractQuantity, /
) -> AstropyQuantity:
    """Convert a `unxt.AbstractQuantity` to a `astropy.units.Quantity`.

    Examples
    --------
    >>> import unxt as u
    >>> from astropy.units import Quantity as AstropyQuantity
    >>> from plum import convert

    >>> convert(u.Q(1.0, "cm"), AstropyQuantity)
    <Quantity 1. cm>

    """
    u = uapi.unit_of(q)
    return AstropyQuantity(uapi.ustrip(u, q), u)


@plum.conversion_method(type_from=AbstractQuantity, type_to=AstropyDistance)  # type: ignore[arg-type]
def convert_unxt_quantity_to_astropy_distance(
    q: AbstractQuantity, /
) -> AstropyDistance:
    """Convert a `unxt.AbstractQuantity` to a `astropy.coordinates.Distance`.

    Examples
    --------
    >>> import unxt as u
    >>> from astropy.coordinates import Distance as AstropyDistance
    >>> from plum import convert

    >>> convert(u.Q(1.0, "cm"), AstropyDistance)
    <Distance 1. cm>

    """
    u = uapi.unit_of(q)
    return AstropyDistance(uapi.ustrip(u, q), u)


@plum.conversion_method(type_from=AbstractQuantity, type_to=AstropyAngle)  # type: ignore[arg-type]
def convert_unxt_quantity_to_astropy_angle(q: AbstractQuantity, /) -> AstropyAngle:
    """Convert a `unxt.quantity.AbstractQuantity` to a `astropy.coordinates.Angle`.

    Examples
    --------
    >>> import unxt as u
    >>> from astropy.coordinates import Angle as AstropyAngle
    >>> from plum import convert

    >>> convert(u.Q(1.0, "rad"), AstropyAngle)
    <Angle 1. rad>

    """
    u = uapi.unit_of(q)
    return AstropyAngle(uapi.ustrip(u, q), u)


# ============================================================================
# Quantity


@plum.conversion_method(type_from=AstropyQuantity, type_to=Quantity)
def convert_astropy_quantity_to_unxt_quantity(q: AstropyQuantity, /) -> Quantity:
    """Convert a `astropy.units.Quantity` to a `unxt.Quantity`.

    Examples
    --------
    >>> from astropy.units import Quantity as AstropyQuantity
    >>> from plum import convert
    >>> from unxt import Quantity

    >>> convert(AstropyQuantity(1.0, "cm"), Quantity)
    Quantity(Array(1., dtype=float32), unit='cm')

    """
    u = uapi.unit_of(q)
    return Quantity(uapi.ustrip(u, q), u)


# ============================================================================
# BareQuantity


@plum.conversion_method(type_from=AstropyQuantity, type_to=BareQuantity)  # type: ignore[arg-type]
def convert_astropy_quantity_to_unxt_barequantity(
    q: AstropyQuantity, /
) -> BareQuantity:
    """Convert a `astropy.units.Quantity` to a `unxt.BareQuantity`.

    Examples
    --------
    >>> from astropy.units import Quantity as AstropyQuantity
    >>> from plum import convert
    >>> from unxt.quantity import BareQuantity

    >>> convert(AstropyQuantity(1.0, "cm"), BareQuantity)
    BareQuantity(Array(1., dtype=float32), unit='cm')

    """
    u = uapi.unit_of(q)
    return BareQuantity(uapi.ustrip(u, q), u)


###############################################################################


@plum.dispatch
def uconvert_value(uto: APYUnits, ufrom: APYUnits, x: ArrayLike, /) -> ArrayLike:
    """Convert the value to the specified units.

    Examples
    --------
    >>> import astropy.units as apyu
    >>> import unxt as u

    >>> u.uconvert_value(apyu.Unit("km"), apyu.Unit("m"), 1000)
    1.0

    """
    # Hot-path: if no unit conversion is necessary
    if ufrom == uto:
        return x

    # Compute the new value.
    return ufrom.to(uto, x)


@plum.dispatch
def uconvert(u: APYUnits, x: AbstractQuantity, /) -> AbstractQuantity:
    """Convert the quantity to the specified units.

    Examples
    --------
    >>> import astropy.units as apyu
    >>> import unxt as u

    >>> x = u.Quantity(1000, "m")
    >>> u.uconvert(u.unit("km"), x)
    Quantity(Array(1., dtype=float32, ...), unit='km')

    >>> x = u.Quantity([1, 2, 3], "Kelvin")
    >>> with apyu.add_enabled_equivalencies(apyu.temperature()):
    ...     y = x.uconvert("deg_C")
    >>> y
    Quantity(Array([-272.15, -271.15, -270.15], dtype=float32, ...), unit='deg_C')

    >>> x = Quantity([1, 2, 3], "radian")
    >>> with apyu.add_enabled_equivalencies(apyu.dimensionless_angles()):
    ...     y = x.uconvert("")
    >>> y
    Quantity(Array([1., 2., 3.], dtype=float32, ...), unit='')

    """
    # Hot-path: if no unit conversion is necessary
    if x.unit == u:
        return x

    value = x.unit.to(u, uapi.ustrip(x))

    # If the dimensions are the same, we can just replace the value and unit.
    if uapi.dimension_of(x.unit) == uapi.dimension_of(u):
        return dc.replace(x, value=value, unit=u)

    # If the dimensions are different, we need to create a new quantity since
    # the dimensions can be part of the type. This won't work if the Quantity
    # type itself needs to be changed, e.g. `unxt.Angle` -> `unxt.Quantity`.
    # These cases are handled separately, in other dispatches.
    fs = dict(dc.field_items(x))  # pylint: disable=unreachable
    fs["value"] = value
    fs["unit"] = u

    return plum.type_unparametrized(x)(**fs)


# ============================================================================


@plum.dispatch
def ustrip(u: Any, x: AstropyQuantity) -> Any:
    """Strip the units from the quantity.

    Examples
    --------
    >>> import astropy.units as apyu
    >>> import unxt as u

    >>> x = apyu.Quantity(1000, "m")
    >>> float(u.ustrip(u.unit("m"), x))
    1000.0

    """
    return x.to_value(u)


@plum.dispatch  # TODO: type annotate by value
def ustrip(flag: type[AllowValue], u: Any, x: AstropyQuantity, /) -> Any:
    """Strip the units from a quantity.

    Examples
    --------
    >>> import unxt as u
    >>> q = u.Quantity(1000, "m")
    >>> u.ustrip(AllowValue, "km", q)
    Array(1., dtype=float32, ...)

    """
    return uapi.ustrip(u, x)


# ============================================================================


@plum.dispatch
def fields(obj: AbstractQuantity, /) -> tuple[dataclasses.Field, ...]:
    """Return the fields of a quantity.

    Examples
    --------
    >>> import dataclassish as dc
    >>> import unxt as u

    >>> q = u.Quantity(1, "m")
    >>> dc.fields(q)
    (Field(name='value',...), Field(name='unit',...))

    """
    value_field = dataclasses.Field(
        dataclasses.MISSING,
        dataclasses.MISSING,
        True,  # noqa: FBT003
        False,  # noqa: FBT003
        False,  # noqa: FBT003
        False,  # noqa: FBT003
        {},
        False,  # noqa: FBT003
    )
    value_field.name = "value"
    unit_field = dataclasses.Field(
        dataclasses.MISSING,
        dataclasses.MISSING,
        True,  # noqa: FBT003
        False,  # noqa: FBT003
        False,  # noqa: FBT003
        False,  # noqa: FBT003
        {},
        False,  # noqa: FBT003
    )
    unit_field.name = "unit"
    return (value_field, unit_field)
