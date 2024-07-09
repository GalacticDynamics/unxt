"""Unitsystem compatibility."""

__all__: list[str] = []

from typing import Any

from astropy.coordinates import Angle as AstropyAngle, Distance as AstropyDistance
from astropy.units import Quantity as AstropyQuantity, UnitBase as Unit
from jaxtyping import ArrayLike
from plum import conversion_method, dispatch

import quaxed.array_api as xp

from unxt import (  # type: ignore[attr-defined]
    AbstractQuantity,
    Distance,
    DistanceModulus,
    Parallax,
    Quantity,
    UncheckedQuantity,
)

# ============================================================================
# AbstractQuantity

# -----------------
# Constructor


@AbstractQuantity.constructor._f.register  # noqa: SLF001
def constructor(
    cls: type[AbstractQuantity], value: AstropyQuantity, /, **kwargs: Any
) -> AbstractQuantity:
    """Construct a `Quantity` from another `Quantity`.

    The `value` is converted to the new `unit`.

    Examples
    --------
    >>> import astropy.units as u
    >>> from unxt import Quantity

    >>> Quantity.constructor(u.Quantity(1, "m"))
    Quantity['length'](Array(1., dtype=float32), unit='m')

    """
    return cls(xp.asarray(value.value, **kwargs), value.unit)


@AbstractQuantity.constructor._f.register  # type: ignore[no-redef] # noqa: SLF001
def constructor(
    cls: type[AbstractQuantity], value: AstropyQuantity, unit: Any, /, **kwargs: Any
) -> AbstractQuantity:
    """Construct a `Quantity` from another `Quantity`.

    The `value` is converted to the new `unit`.

    Examples
    --------
    >>> import astropy.units as u
    >>> from unxt import Quantity

    >>> Quantity.constructor(u.Quantity(1, "m"), "cm")
    Quantity['length'](Array(100., dtype=float32), unit='cm')

    """
    return cls(xp.asarray(value.to_value(unit), **kwargs), unit)


# -----------------
# Conversion Methods


@conversion_method(type_from=AbstractQuantity, type_to=AstropyQuantity)  # type: ignore[misc]
def convert_unxt_quantity_to_astropy_quantity(
    q: AbstractQuantity, /
) -> AstropyQuantity:
    """Convert a `unxt.AbstractQuantity` to a `astropy.units.Quantity`.

    Examples
    --------
    >>> from astropy.units import Quantity as AstropyQuantity
    >>> from plum import convert
    >>> from unxt import Quantity

    >>> convert(Quantity(1.0, "cm"), AstropyQuantity)
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
    >>> from astropy.coordinates import Distance
    >>> from plum import convert
    >>> from unxt import Quantity

    >>> convert(Quantity(1.0, "cm"), Distance)
    <Distance 1. cm>

    """
    return AstropyDistance(q.value, q.unit)


@conversion_method(type_from=AbstractQuantity, type_to=AstropyAngle)  # type: ignore[misc]
def convert_unxt_quantity_to_astropy_angle(q: AbstractQuantity, /) -> AstropyAngle:
    """Convert a `unxt.AbstractQuantity` to a `astropy.coordinates.Angle`.

    Examples
    --------
    >>> from astropy.coordinates import Angle
    >>> from plum import convert
    >>> from unxt import Quantity

    >>> convert(Quantity(1.0, "radian"), Angle)
    <Angle 1. rad>

    """
    return AstropyAngle(q.value, q.unit)


# ---------------------------
# to_units


@dispatch  # type: ignore[misc]
def to_units(value: AstropyQuantity, units: Unit, /) -> Quantity:
    """Convert an Astropy Quantity to the given units.

    Examples
    --------
    >>> from unxt import to_units
    >>> import astropy.units as u

    >>> q = u.Quantity(1, "m")
    >>> to_units(q, "cm")
    Quantity['length'](Array(100., dtype=float32), unit='cm')

    """
    return Quantity.constructor(value, units)


# ---------------------------
# to_units_value


@dispatch  # type: ignore[misc]
def to_units_value(value: AstropyQuantity, units: Unit | str, /) -> ArrayLike:
    """Convert an Astropy Quantity to an array with the given units.

    Examples
    --------
    >>> from unxt import to_units_value
    >>> import astropy.units as u

    >>> q = u.Quantity(1, "m")
    >>> to_units_value(q, "cm")
    np.float64(100.0)

    """
    return value.to_value(units)


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
    >>> from unxt import UncheckedQuantity

    >>> convert(AstropyQuantity(1.0, "cm"), UncheckedQuantity)
    UncheckedQuantity(Array(1., dtype=float32), unit='cm')

    """
    return UncheckedQuantity(q.value, q.unit)


# ============================================================================
# Distance


@conversion_method(type_from=AstropyQuantity, type_to=Distance)  # type: ignore[misc]
def convert_astropy_quantity_to_unxt_distance(q: AstropyQuantity, /) -> Distance:
    """Convert a `astropy.units.Quantity` to a `unxt.Distance`.

    Examples
    --------
    >>> from astropy.units import Quantity as AstropyQuantity
    >>> from plum import convert
    >>> from unxt import Distance

    >>> convert(AstropyQuantity(1.0, "cm"), Distance)
    Distance(Array(1., dtype=float32), unit='cm')

    """
    return Distance(q.value, q.unit)


# ============================================================================
# Parallax


@conversion_method(type_from=AstropyQuantity, type_to=Parallax)  # type: ignore[misc]
def convert_astropy_quantity_to_unxt_parallax(q: AstropyQuantity, /) -> Parallax:
    """Convert a `astropy.units.Quantity` to a `unxt.Parallax`.

    Examples
    --------
    >>> from astropy.units import Quantity as AstropyQuantity
    >>> from plum import convert
    >>> from unxt import Parallax

    >>> convert(AstropyQuantity(1.0, "radian"), Parallax)
    Parallax(Array(1., dtype=float32), unit='rad')

    """
    return Parallax(q.value, q.unit)


# ============================================================================
# Distance Modulus


@conversion_method(type_from=AstropyQuantity, type_to=DistanceModulus)  # type: ignore[misc]
def convert_astropy_quantity_to_unxt_distmod(q: AstropyQuantity, /) -> DistanceModulus:
    """Convert a `astropy.units.Quantity` to a `unxt.DistanceModulus`.

    Examples
    --------
    >>> from astropy.units import Quantity as AstropyQuantity
    >>> from plum import convert
    >>> from unxt import DistanceModulus

    >>> convert(AstropyQuantity(1.0, "mag"), DistanceModulus)
    DistanceModulus(Array(1., dtype=float32), unit='mag')

    """
    return DistanceModulus(q.value, q.unit)
