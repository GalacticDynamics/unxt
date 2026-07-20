"""Astropy quantity compatibility for unxt quantity classes."""

__all__: tuple[str, ...] = ()

from typing import Any, NoReturn

import plum
from astropy.coordinates import Angle as AstropyAngle, Distance as AstropyDistance
from astropy.units import Quantity as AstropyQuantity
from jaxtyping import ArrayLike

import dataclassish as dc
import quaxed.numpy as jnp

import unxt_api as uapi
from .custom_types import APYUnits
from unxt.quantity import (
    AbstractQuantity,
    AllowValue,
    Quantity,
    ustrip,
)

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
    """Construct a quantity from an astropy `Quantity`.

    The `value` is converted to the new `unit`.

    Examples
    --------
    >>> import unxt as u
    >>> import astropy.units as apyu

    >>> u.Q.from_(apyu.Quantity(1, "m"))
    Quantity(Array(1., dtype=float32), unit='m')

    """
    u = uapi.unit_of(value)
    value = jnp.asarray(uapi.ustrip(u, value), **kwargs)
    return cls(value, u)


@AbstractQuantity.from_.dispatch
def from_(
    cls: type[AbstractQuantity], value: AstropyQuantity, u: Any, /, **kwargs: Any
) -> AbstractQuantity:
    """Construct a quantity from an astropy `Quantity`, converting to a target unit.

    The `value` is converted to the new `unit`.

    Examples
    --------
    >>> import unxt as u
    >>> import astropy.units as apyu

    >>> u.Q.from_(apyu.Quantity(1, "m"), "cm")
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


@plum.conversion_method(type_from=AstropyQuantity, type_to=AbstractQuantity)  # type: ignore[arg-type]
def convert_astropy_quantity_to_unxt_quantity(q: AstropyQuantity, /) -> Quantity:
    """Convert a `astropy.units.Quantity` to a `unxt.Quantity`.

    Registered against the abstract base: `plum` matches a conversion whose
    ``type_to`` is a supertype of the requested target, so this single method
    serves both ``convert(q, AbstractQuantity)`` and ``convert(q, Quantity)``
    (``Quantity`` is a subclass of `~unxt.quantity.AbstractQuantity`). Targeting
    the base also lets callers that only have `AbstractQuantity` in scope (e.g.
    arithmetic coercion in ``AbstractQuantity``) request the conversion without
    importing the concrete class.

    Examples
    --------
    >>> from astropy.units import Quantity as AstropyQuantity
    >>> from plum import convert
    >>> from unxt.quantity import AbstractQuantity, Quantity

    >>> convert(AstropyQuantity(1.0, "cm"), Quantity)
    Quantity(Array(1., dtype=float32), unit='cm')

    >>> convert(AstropyQuantity(1.0, "cm"), AbstractQuantity)
    Quantity(Array(1., dtype=float32), unit='cm')

    """
    u = uapi.unit_of(q)
    return Quantity(uapi.ustrip(u, q), u)


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
def uconvert_value(
    uto: APYUnits, ufrom: APYUnits, x: AstropyQuantity, /
) -> AstropyQuantity:
    """Convert an astropy quantity to the specified units.

    This is a convenience dispatch mirroring the `AbstractQuantity` one: users
    may pass a quantity to the lower-level value function and it keeps working.
    Like that dispatch, it checks that ``ufrom`` is convertible to the
    quantity's own unit, then defers to the quantity's own unit for the
    arithmetic -- returning a *relabelled quantity*, not a bare value.

    This dispatch is required: an astropy `~astropy.units.Quantity` subclasses
    `numpy.ndarray`, so it satisfies the ``x: ArrayLike`` annotation of the
    bare-value dispatch above and is not rejected even under `beartype`. That
    body's ``ufrom.to(uto, x)`` converts the magnitude but returns a quantity
    still carrying ``ufrom``'s unit, mislabelling the result (e.g.
    ``uconvert_value("m", "km", 1 km)`` gave ``1000.0 km``).

    Examples
    --------
    >>> import astropy.units as apyu
    >>> import unxt as u

    >>> u.uconvert_value(apyu.Unit("m"), apyu.Unit("km"), apyu.Quantity(1.0, "km"))
    <Quantity 1000. m>

    >>> u.uconvert_value(apyu.Unit("m"), apyu.Unit("m"), apyu.Quantity(5.0, "m"))
    <Quantity 5. m>

    """
    if not uapi.is_unit_convertible(ufrom, x.unit):
        msg = f"Cannot convert from {x.unit} to {ufrom}"
        raise ValueError(msg)

    return x.to(uto)


@plum.dispatch
def uconvert(u: APYUnits, x: AbstractQuantity, /) -> AbstractQuantity:
    """Convert the quantity to the specified units.

    Examples
    --------
    >>> import astropy.units as apyu
    >>> import unxt as u

    >>> x = u.Q(1000, "m")
    >>> u.uconvert(u.unit("km"), x)
    Quantity(Array(1., dtype=float32, ...), unit='km')

    >>> x = u.Q([1, 2, 3], "Kelvin")
    >>> with apyu.add_enabled_equivalencies(apyu.temperature()):
    ...     y = x.uconvert("deg_C")
    >>> y
    Quantity( Array([-272.15, -271.15, -270.15], dtype=float32, ...), unit='deg_C' )

    >>> x = u.Q([1, 2, 3], "radian")
    >>> with apyu.add_enabled_equivalencies(apyu.dimensionless_angles()):
    ...     y = x.uconvert("")
    >>> y
    Quantity(Array([1., 2., 3.], dtype=float32, ...), unit='')

    """
    # Hot-path: skip the conversion only when the unit is genuinely unchanged.
    # NB: astropy treats physically-equal units as ``==`` (e.g. ``J == m2 kg /
    # s2``), so ``x.unit == u`` alone would silently return ``x`` without
    # relabeling to the requested (equal but differently-named) unit. Require
    # the string forms to match too so the relabel still happens.
    #
    # Check identity first: astropy interns named units, so the common
    # "convert to the unit it already has" case hits ``is`` and skips two
    # ``to_string()`` calls (~116x the cost of the identity test). Identity
    # implies identical string forms, so this cannot skip a needed relabel --
    # the ``J`` vs ``m2 kg / s2`` case is ``==`` but *not* ``is``.
    if x.unit is u:
        return x
    if x.unit == u and x.unit.to_string() == u.to_string():
        return x

    value = x.unit.to(u, uapi.ustrip(x))

    # If the dimensions are the same, we can just replace the value and unit.
    if uapi.dimension_of(x.unit) == uapi.dimension_of(u):
        return dc.replace(x, value=value, unit=u)

    # If the dimensions are different, we need to create a new quantity since
    # the dimensions can be part of the type. This won't work if the quantity
    # type itself needs to be changed, e.g. `unxt.Angle` -> `unxt.Quantity`.
    # These cases are handled separately, in other dispatches.
    fs = dict(dc.field_items(x))
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
    >>> import astropy.units as apyu
    >>> import unxt as u
    >>> from unxt.quantity import AllowValue
    >>> q = apyu.Quantity(1000, "m")
    >>> float(u.ustrip(AllowValue, "km", q))
    1.0

    """
    return uapi.ustrip(u, x)


@plum.dispatch
def ustrip(flag: type[AllowValue], x: AstropyQuantity, /) -> Any:
    """Strip the units from an astropy quantity, allowing bare values through.

    The two-argument :class:`~unxt.quantity.AllowValue` form takes no target
    unit, so it returns the value in the quantity's own unit. This dispatch
    disambiguates ``ustrip(AllowValue, <astropy Quantity>)``, which otherwise
    matched both ``ustrip(type[AllowValue], Any)`` and
    ``ustrip(Any, AstropyQuantity)`` with neither dominating. It mirrors the
    ``(type[AllowValue], AbstractQuantity)`` form for unxt quantities.

    Examples
    --------
    >>> import astropy.units as apyu
    >>> import unxt as u
    >>> from unxt.quantity import AllowValue
    >>> q = apyu.Quantity(1000, "m")
    >>> float(u.ustrip(AllowValue, q))
    1000.0

    """
    return x.value
