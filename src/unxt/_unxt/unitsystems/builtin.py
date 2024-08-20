"""Built-in unit systems."""

from __future__ import annotations

__all__ = ["DimensionlessUnitSystem", "LTMAUnitSystem", "LTMAVUnitSystem"]

from dataclasses import dataclass
from typing import Annotated, TypeAlias, final
from typing_extensions import override

import astropy.units as u
from astropy.units import dimensionless_unscaled

from . import builtin_dimensions as ud  # noqa: TCH001
from .base import AbstractUnitSystem

Unit: TypeAlias = u.UnitBase

_dimless_insts: dict[type[DimensionlessUnitSystem], DimensionlessUnitSystem] = {}


@final
@dataclass(frozen=True, slots=True)
class DimensionlessUnitSystem(AbstractUnitSystem):
    """A unit system with only dimensionless units."""

    dimensionless: Annotated[Unit, ud.dimensionless] = dimensionless_unscaled

    def __new__(cls) -> DimensionlessUnitSystem:
        # Check if instance already exists
        if cls in _dimless_insts:
            return _dimless_insts[cls]
        # Create new instance and cache it
        self = object.__new__(cls)
        _dimless_insts[cls] = self
        return self

    def __repr__(self) -> str:
        return "DimensionlessUnitSystem()"

    @override
    def __str__(self) -> str:
        return self.__repr__()


@final
@dataclass(frozen=True, slots=True)
class LTMAUnitSystem(AbstractUnitSystem):
    """Length, time, mass, angle unit system."""

    length: Annotated[Unit, ud.length]
    time: Annotated[Unit, ud.time]
    mass: Annotated[Unit, ud.mass]
    angle: Annotated[Unit, ud.angle]


@final
@dataclass(frozen=True, slots=True)
class LTMAVUnitSystem(AbstractUnitSystem):
    """Length, time, mass, angle, speed unit system."""

    length: Annotated[Unit, ud.length]
    time: Annotated[Unit, ud.time]
    mass: Annotated[Unit, ud.mass]
    angle: Annotated[Unit, ud.angle]
    speed: Annotated[Unit, ud.speed]
