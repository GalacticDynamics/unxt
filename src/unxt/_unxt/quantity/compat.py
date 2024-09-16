"""Compatibility for Quantity."""

__all__: list[str] = []

from jaxtyping import Array
from plum import conversion_method, dispatch

from dataclassish import replace

from .base import AbstractQuantity
from .core import Quantity
from .distance import Distance, DistanceModulus, Parallax
from .fast import UncheckedQuantity
from unxt._unxt.dimensions.core import AbstractDimensions
from unxt._unxt.units.core import AbstractUnits, units
from unxt._unxt.units.system.base import AbstractUnitSystem

# ===================================================================
# Get dimensions


@dispatch  # type: ignore[misc]
def dimensions_of(obj: AbstractQuantity, /) -> AbstractDimensions:
    """Return the dimensions of a quantity.

    Examples
    --------
    >>> from unxt import dimensions_of, Quantity
    >>> q = Quantity(1, "m")
    >>> dimensions_of(q)
    PhysicalType('length')

    """
    return dimensions_of(obj.unit)


# ===================================================================
# Get units


@dispatch  # type: ignore[misc]
def units_of(obj: AbstractQuantity, /) -> AbstractUnits:
    """Return the units of an object.

    Examples
    --------
    >>> from unxt import units_of, Quantity
    >>> q = Quantity(1, "m")
    >>> units_of(q)
    Unit("m")

    """
    return obj.unit


# ===================================================================
# Convert units


@dispatch
def uconvert(u: AbstractUnits, x: AbstractQuantity, /) -> AbstractQuantity:
    """Convert the quantity to the specified units.

    Examples
    --------
    >>> from unxt import Quantity, units

    >>> x = Quantity(1000, "m")
    >>> uconvert(units("km"), x)
    Quantity['length'](Array(1., dtype=float32, ...), unit='km')

    """
    # Hot-path: if no unit conversion is necessary
    if x.unit == u:
        return x

    # TODO: jaxpr units so we can understand them at trace time.
    # Hot-path: if in tracing mode
    # if isinstance(x.value, jax.core.Tracer) and not can_convert_unit(x.unit, u):
    #     return x.value

    # Astropy correction factor
    # TODO: this only works with multiplicative unit conversions
    factor = x.unit.to(u)

    return replace(x, value=x.value * factor, unit=u)


@dispatch
def uconvert(u: str, x: AbstractQuantity, /) -> AbstractQuantity:
    """Convert the quantity to the specified units.

    Examples
    --------
    >>> from unxt import Quantity, units

    >>> x = Quantity(1000, "m")
    >>> uconvert("km", x)
    Quantity['length'](Array(1., dtype=float32, ...), unit='km')

    """
    return uconvert(units(u), x)


@dispatch
def uconvert(u: AbstractUnitSystem, x: AbstractQuantity, /) -> AbstractQuantity:
    """Strip the units from the quantity.

    Examples
    --------
    >>> from unxt import Quantity, units
    >>> from unxt.unitsystems import galactic

    >>> q = Quantity(1e17, "km")
    >>> uconvert(galactic, q)
    Quantity['length'](Array(3.2407792, dtype=float32, ...), unit='kpc')

    """
    return uconvert(u[dimensions_of(x)], x)


# ===================================================================
# Strip units


@dispatch
def ustrip(u: AbstractUnits, x: AbstractQuantity, /) -> Array:
    """Strip the units from the quantity.

    Examples
    --------
    >>> from unxt import Quantity, units

    >>> q = Quantity(1000, "m")
    >>> ustrip(units("km"), q)
    Array(1., dtype=float32, ...)

    """
    return uconvert(u, x).value


@dispatch
def ustrip(u: str, x: AbstractQuantity, /) -> Array:
    """Strip the units from the quantity.

    Examples
    --------
    >>> from unxt import Quantity

    >>> q = Quantity(1000, "m")
    >>> ustrip("km", q)
    Array(1., dtype=float32, ...)

    """
    return uconvert(units(u), x).value


@dispatch
def ustrip(u: AbstractUnitSystem, x: AbstractQuantity, /) -> Array:
    """Strip the units from the quantity.

    Examples
    --------
    >>> from unxt import Quantity
    >>> from unxt.unitsystems import galactic

    >>> q = Quantity(1e17, "km")
    >>> ustrip(galactic, q)
    Array(3.2407792, dtype=float32, weak_type=True)

    """
    return ustrip(u[dimensions_of(x)], x)


#####################################################################
# Conversion


@conversion_method(type_from=AbstractQuantity, type_to=UncheckedQuantity)  # type: ignore[misc]
def _quantity_to_unchecked(q: AbstractQuantity, /) -> UncheckedQuantity:
    """Convert any quantity to an unchecked quantity.

    Examples
    --------
    >>> from plum import convert
    >>> from unxt import Quantity, UncheckedQuantity
    >>> q = Quantity(1, "m")
    >>> q
    Quantity['length'](Array(1, dtype=int32, ...), unit='m')

    The self-conversion doesn't copy the object:

    >>> q = UncheckedQuantity(1, "m")
    >>> convert(q, UncheckedQuantity) is q
    True

    """
    if isinstance(q, UncheckedQuantity):
        return q
    return UncheckedQuantity(q.value, q.unit)


@conversion_method(type_from=AbstractQuantity, type_to=Quantity)  # type: ignore[misc]
def _quantity_to_checked(q: AbstractQuantity, /) -> Quantity:
    """Convert any quantity to a checked quantity.

    Examples
    --------
    >>> from plum import convert
    >>> from unxt import Quantity, UncheckedQuantity
    >>> q = UncheckedQuantity(1, "m")
    >>> q
    UncheckedQuantity(Array(1, dtype=int32, ...), unit='m')

    >>> convert(q, Quantity)
    Quantity['length'](Array(1, dtype=int32, ...), unit='m')

    The self-conversion doesn't copy the object:

    >>> q = Quantity(1, "m")
    >>> convert(q, Quantity) is q
    True

    """
    if isinstance(q, Quantity):
        return q
    return Quantity(q.value, q.unit)


@conversion_method(type_from=AbstractQuantity, type_to=Distance)  # type: ignore[misc]
def _quantity_to_distance(q: AbstractQuantity, /) -> Distance:
    """Convert any quantity to a Distance.

    Examples
    --------
    >>> from plum import convert
    >>> from unxt import Quantity, Distance
    >>> q = UncheckedQuantity(1, "m")
    >>> q
    UncheckedQuantity(Array(1, dtype=int32, ...), unit='m')

    >>> convert(q, Distance)
    Distance(Array(1, dtype=int32, ...), unit='m')

    The self-conversion doesn't copy the object:

    >>> q = Distance(1, "m")
    >>> convert(q, Distance) is q
    True

    """
    if isinstance(q, Distance):
        return q
    return Distance(q.value, q.unit)


@conversion_method(type_from=AbstractQuantity, type_to=Parallax)  # type: ignore[misc]
def _quantity_to_parallax(q: AbstractQuantity, /) -> Parallax:
    """Convert any quantity to a Parallax.

    Examples
    --------
    >>> from plum import convert
    >>> from unxt import Quantity, Parallax
    >>> q = Quantity(1, "mas")
    >>> q
    Quantity['angle'](Array(1, dtype=int32, ...), unit='mas')

    >>> convert(q, Parallax)
    Parallax(Array(1, dtype=int32, weak_type=True), unit='mas')

    The self-conversion doesn't copy the object:

    >>> q = Parallax(1, "mas")
    >>> convert(q, Parallax) is q
    True

    """
    if isinstance(q, Parallax):
        return q
    return Parallax(q.value, q.unit)


@conversion_method(type_from=AbstractQuantity, type_to=DistanceModulus)  # type: ignore[misc]
def _quantity_to_distmod(q: AbstractQuantity, /) -> DistanceModulus:
    """Convert any quantity to a DistanceModulus.

    Examples
    --------
    >>> from plum import convert
    >>> from unxt import Quantity, DistanceModulus
    >>> q = Quantity(1, "mag")
    >>> q
    Quantity['dex'](Array(1, dtype=int32, ...), unit='mag')

    >>> convert(q, DistanceModulus)
    DistanceModulus(Array(1, dtype=int32, ...), unit='mag')

    The self-conversion doesn't copy the object:

    >>> q = DistanceModulus(1, "mag")
    >>> convert(q, DistanceModulus) is q
    True

    """
    if isinstance(q, DistanceModulus):
        return q
    return DistanceModulus(q.value, q.unit)
