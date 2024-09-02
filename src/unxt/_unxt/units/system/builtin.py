"""Built-in unit systems."""

from __future__ import annotations

__all__ = ["DimensionlessUnitSystem", "LTMAUnitSystem", "SIUnitSystem"]

from dataclasses import dataclass
from typing import Annotated, TypeAlias, final
from typing_extensions import override

import astropy.units as u
from astropy.units import dimensionless_unscaled

from . import builtin_dimensions as ud  # noqa: TCH001
from .base import AbstractUnitSystem
from unxt._unxt.utils import SingletonMixin

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
class SIUnitSystem(AbstractUnitSystem):
    """SI unit system + angles."""

    # Base SI dimensions
    length: Annotated[Unit, ud.length]
    time: Annotated[Unit, ud.time]
    amount: Annotated[Unit, ud.amount]
    electic_current: Annotated[Unit, ud.current]
    temperature: Annotated[Unit, ud.temperature]
    luminous_intensity: Annotated[Unit, ud.luminous_intensity]
    mass: Annotated[Unit, ud.mass]
    # + angles
    angle: Annotated[Unit, ud.angle]
