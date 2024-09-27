"""Functional approach to Quantities."""

__all__: list[str] = []


from jaxtyping import Array
from plum import dispatch

from dataclassish import replace

from .base import AbstractQuantity
from unxt._src.dimensions.core import dimensions_of
from unxt._src.units.core import AbstractUnits, units
from unxt._src.units.system.base import AbstractUnitSystem

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
def uconvert(ustr: str, x: AbstractQuantity, /) -> AbstractQuantity:
    """Convert the quantity to the specified units.

    Examples
    --------
    >>> from unxt import Quantity, units

    >>> x = Quantity(1000, "m")
    >>> uconvert("km", x)
    Quantity['length'](Array(1., dtype=float32, ...), unit='km')

    """
    return uconvert(units(ustr), x)


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
    return uconvert(usys[dimensions_of(x)], x)


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
