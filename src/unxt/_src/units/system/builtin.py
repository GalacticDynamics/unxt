"""Built-in unit systems."""

from __future__ import annotations

__all__ = ["DimensionlessUnitSystem", "LTMAUnitSystem"]

from dataclasses import dataclass
from typing import Annotated, TypeAlias, final
from typing_extensions import override

import astropy.units as u
from astropy.units import dimensionless_unscaled

from . import builtin_dimensions as ud  # noqa: TCH001
from .base import AbstractUnitSystem
from unxt._src.utils import SingletonMixin

Unit: TypeAlias = u.UnitBase


@final
@dataclass(frozen=True, slots=True)
class DimensionlessUnitSystem(SingletonMixin, AbstractUnitSystem):
    """A unit system with only dimensionless units.

    This is a singleton class.

    Examples
    --------
    >>> from unxt.unitsystems import DimensionlessUnitSystem
    >>> dims1 = DimensionlessUnitSystem()
    >>> dims2 = DimensionlessUnitSystem()
    >>> dims1 is dims2
    True

    """

    #: The dimensionless unit.
    dimensionless: Annotated[Unit, ud.dimensionless] = dimensionless_unscaled

    def __repr__(self) -> str:
        return "DimensionlessUnitSystem()"

    @override
    def __str__(self) -> str:
        return self.__repr__()


@final
@dataclass(frozen=True, slots=True, repr=False)
class LTMAUnitSystem(AbstractUnitSystem):
    """Length, time, mass, angle unit system."""

    #: Units for the length dimension.
    length: Annotated[Unit, ud.length]

    #: Units for the time dimension.
    time: Annotated[Unit, ud.time]

    #: Units for the mass dimension.
    mass: Annotated[Unit, ud.mass]

    #: Units for the angle 'dimension'.
    angle: Annotated[Unit, ud.angle]

    def __repr__(self) -> str:
        fs = ", ".join(map(str, self.base_units))
        return f"unitsystem({fs})"


@final
@dataclass(frozen=True, slots=True)
class SIUnitSystem(SingletonMixin, AbstractUnitSystem):
    """SI unit system + angles.

    Note: this is not part of the public API! Use the `si` instance (realization) from
    `unxt.unitsystems` instead.

    Examples
    --------
    >>> from unxt import units
    >>> from unxt._src.units.system.builtin import SIUnitSystem
    >>> usys = SIUnitSystem(
    ...     length=units("meter"),
    ...     time=units("second"),
    ...     mass=units("kilogram"),
    ...     electric_current=units("ampere"),
    ...     temperature=units("Kelvin"),
    ...     amount=units("mole"),
    ...     luminous_intensity=units("candela"),
    ...     angle=units("radian"),
    ... )
    >>> usys
    unitsystem(m, kg, s, mol, A, K, cd, rad)

    """

    # Base SI dimensions
    #: Units for the length dimension.
    length: Annotated[Unit, ud.length]

    #: Units for the mass dimension.
    mass: Annotated[Unit, ud.mass]

    #: Units for the time dimension.
    time: Annotated[Unit, ud.time]

    #: Units for the amount of substance dimension.
    amount: Annotated[Unit, ud.amount]

    #: Units for the electric current dimension.
    electric_current: Annotated[Unit, ud.current]

    #: Units for the temperature dimension.
    temperature: Annotated[Unit, ud.temperature]

    #: Units for the luminous intensity dimension.
    luminous_intensity: Annotated[Unit, ud.luminous_intensity]

    # + angles
    #: Units for the angle 'dimension'.
    angle: Annotated[Unit, ud.angle]

    def __repr__(self) -> str:
        fs = ", ".join(map(str, self.base_units))
        return f"unitsystem({fs})"


@final
@dataclass(frozen=True, slots=True)
class CGSUnitSystem(SingletonMixin, AbstractUnitSystem):
    """CGS unit system + angles.

    Note: this is not part of the public API! Use the `cgs` instance (realization) from
    `unxt.unitsystems` instead.

    Examples
    --------
    >>> from unxt import units
    >>> from unxt._src.units.system.builtin import CGSUnitSystem
    >>> usys = CGSUnitSystem(
    ...     length=units("centimeter"),
    ...     time=units("second"),
    ...     mass=units("gram"),
    ...     angle=units("radian"),
    ...     force=units("dyne"),
    ...     energy=units("erg"),
    ...     pressure=units("barye"),
    ...     dynamic_viscosity=units("poise"),
    ...     kinematic_viscosity=units("stokes"),
    ... )
    >>> usys
    unitsystem(cm, g, s, dyn, erg, Ba, P, St, rad)

    """

    # Base CGS dimensions
    #: Units for the length dimension.
    length: Annotated[Unit, ud.length]

    #: Units for the mass dimension.
    mass: Annotated[Unit, ud.mass]

    #: Units for the time dimension.
    time: Annotated[Unit, ud.time]

    #: Units for the force dimension.
    force: Annotated[Unit, ud.force]

    #: Units for the energy dimension.
    energy: Annotated[Unit, ud.energy]

    #: Units for the pressure dimension.
    pressure: Annotated[Unit, ud.pressure]

    #: Units for the dynamic viscosity dimension.
    dynamic_viscosity: Annotated[Unit, ud.dynamic_viscosity]

    #: Units for the kinematic viscosity dimension.
    kinematic_viscosity: Annotated[Unit, ud.kinematic_viscosity]

    # + angles
    #: Units for the angle 'dimension'.
    angle: Annotated[Unit, ud.angle]

    def __repr__(self) -> str:
        fs = ", ".join(map(str, self.base_units))
        return f"unitsystem({fs})"
