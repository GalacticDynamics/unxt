"""Unitsystem compatibility."""

__all__: list[str] = []

from typing import Any, TypeAlias

import astropy.units as u
from astropy.coordinates import Angle as AstropyAngle, Distance as AstropyDistance
from astropy.units import Quantity as AstropyQuantity
from jaxtyping import Array
from packaging.version import Version
from plum import conversion_method, dispatch

import quaxed.numpy as jnp
from dataclassish import replace

from unxt import AbstractQuantity, Quantity, UncheckedQuantity
from unxt._interop.optional_deps import OptDeps

# ============================================================================
# AbstractQuantity

# -----------------
# Constructor


@AbstractQuantity.from_._f.register  # noqa: SLF001
def from_(
    cls: type[AbstractQuantity], value: AstropyQuantity, /, **kwargs: Any
) -> AbstractQuantity:
    """Construct a `Quantity` from another `Quantity`.

    The `value` is converted to the new `unit`.

    Examples
    --------
    >>> import astropy.units as u
    >>> from unxt import Quantity

    >>> Quantity.from_(u.Quantity(1, "m"))
    Quantity['length'](Array(1., dtype=float32), unit='m')

    """
    return cls(jnp.asarray(value.value, **kwargs), value.unit)


@AbstractQuantity.from_._f.register  # type: ignore[no-redef] # noqa: SLF001
def from_(
    cls: type[AbstractQuantity], value: AstropyQuantity, unit: Any, /, **kwargs: Any
) -> AbstractQuantity:
    """Construct a `Quantity` from another `Quantity`.

    The `value` is converted to the new `unit`.

    Examples
    --------
    >>> import astropy.units as u
    >>> from unxt import Quantity

    >>> Quantity.from_(u.Quantity(1, "m"), "cm")
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


###############################################################################


AstropyUnit: TypeAlias = u.UnitBase | u.Unit | u.FunctionUnitBase | u.StructuredUnit


if Version("7.0") <= OptDeps.ASTROPY.version:

    def _apy7_unit_to(self: AstropyUnit, other: AstropyUnit, value: Array, /) -> Array:
        return self.to(other, value)

else:

    def _apy7_unit_to(self: AstropyUnit, other: AstropyUnit, value: Array, /) -> Array:
        """Convert the value to the other unit."""
        # return self.get_converter(Unit(other), equivalencies)(value)
        # First see if it is just a scaling.
        try:
            scale = self._to(other)
        except u.UnitsError:
            pass
        else:
            return scale * value

        # if that doesn't work, maybe we can do it with equivalencies?
        try:
            return self._apply_equivalencies(
                self, other, self._normalize_equivalencies([])
            )(value)
        except u.UnitsError as exc:
            # Last hope: maybe other knows how to do it?
            # We assume the equivalencies have the unit itself as first item.
            # TODO: maybe better for other to have a `_back_converter` method?
            if hasattr(other, "equivalencies"):
                for funit, tunit, _, b in other.equivalencies:
                    if other is funit:
                        try:
                            converter = self.get_converter(tunit, [])
                        except Exception:  # noqa: BLE001, S110  # pylint: disable=W0718
                            pass
                        else:
                            return b(converter(value))

            raise exc  # noqa: TRY201


@dispatch  # type: ignore[misc]
def uconvert(unit: AstropyUnit, x: AbstractQuantity, /) -> AbstractQuantity:
    """Convert the quantity to the specified units.

    Examples
    --------
    >>> import astropy.units as u
    >>> from unxt import Quantity, units

    >>> x = Quantity(1000, "m")
    >>> uconvert(units("km"), x)
    Quantity['length'](Array(1., dtype=float32, ...), unit='km')

    >>> x = Quantity([1, 2, 3], "Kelvin")
    >>> with u.add_enabled_equivalencies(u.temperature()):
    ...     y = x.to(u.deg_C)
    >>> y
    Quantity['temperature'](Array([-272.15, -271.15, -270.15], dtype=float32), unit='deg_C')

    """  # noqa: E501
    # Hot-path: if no unit conversion is necessary
    if x.unit == unit:
        return x

    # TODO: jaxpr units so we can understand them at trace time.
    # Hot-path: if in tracing mode
    # if isinstance(x.value, jax.core.Tracer) and not is_unit_convertible(x.unit, u):
    #     return x.value

    return replace(x, value=_apy7_unit_to(x.unit, unit, x.value), unit=unit)
