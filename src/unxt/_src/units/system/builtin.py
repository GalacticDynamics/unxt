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

    length: Annotated[Unit, ud.length]
    time: Annotated[Unit, ud.time]
    mass: Annotated[Unit, ud.mass]
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
    >>> import astropy.units as u
    >>> from unxt._src.units.system.builtin import SIUnitSystem
    >>> usys = SIUnitSystem(
    ...     length=u.meter,
    ...     time=u.second,
    ...     mass=u.kilogram,
    ...     electric_current=u.ampere,
    ...     temperature=u.Kelvin,
    ...     amount=u.mole,
    ...     luminous_intensity=u.candela,
    ...     angle=u.radian,
    ... )
    >>> usys
    unitsystem(m, kg, s, mol, A, K, cd, rad)

    """

    # Base SI dimensions
    length: Annotated[Unit, ud.length]
    mass: Annotated[Unit, ud.mass]
    time: Annotated[Unit, ud.time]
    amount: Annotated[Unit, ud.amount]
    electric_current: Annotated[Unit, ud.current]
    temperature: Annotated[Unit, ud.temperature]
    luminous_intensity: Annotated[Unit, ud.luminous_intensity]
    # + angles
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
    >>> import astropy.units as u
    >>> from unxt._src.units.system.builtin import CGSUnitSystem
    >>> usys = CGSUnitSystem(
    ...     length=u.centimeter,
    ...     time=u.second,
    ...     mass=u.gram,
    ...     angle=u.radian,
    ...     force=u.dyne,
    ...     energy=u.erg,
    ...     pressure=u.barye,
    ...     dynamic_viscosity=u.poise,
    ...     kinematic_viscosity=u.stokes,
    ... )
    >>> usys
    unitsystem(cm, g, s, dyn, erg, Ba, P, St, rad)

    """

    # Base CGS dimensions
    length: Annotated[Unit, ud.length]
    mass: Annotated[Unit, ud.mass]
    time: Annotated[Unit, ud.time]
    force: Annotated[Unit, ud.force]
    energy: Annotated[Unit, ud.energy]
    pressure: Annotated[Unit, ud.pressure]
    dynamic_viscosity: Annotated[Unit, ud.dynamic_viscosity]
    kinematic_viscosity: Annotated[Unit, ud.kinematic_viscosity]
    # + angles
    angle: Annotated[Unit, ud.angle]

    def __repr__(self) -> str:
        fs = ", ".join(map(str, self.base_units))
        return f"unitsystem({fs})"
