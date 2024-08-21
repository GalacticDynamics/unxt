"""Realizations of unit systems."""
# pylint: disable=no-member

__all__ = [
    # unit system instance
    "galactic",
    "dimensionless",
    "solarsystem",
    # unit system alias
    "NAMED_UNIT_SYSTEMS",
]

import astropy.units as u

from .base import AbstractUnitSystem
from .builtin import DimensionlessUnitSystem, LTMAUnitSystem

# Dimensionless. This is a singleton.
dimensionless = DimensionlessUnitSystem()

# Galactic unit system
galactic = LTMAUnitSystem(
    length=u.kpc,
    time=u.Myr,
    mass=u.Msun,
    angle=u.radian,
    # preferred_units={speed: u.km / u.s},
)

# Solar system units
solarsystem = LTMAUnitSystem(length=u.au, time=u.yr, mass=u.Msun, angle=u.radian)


NAMED_UNIT_SYSTEMS: dict[str, AbstractUnitSystem] = {
    "galactic": galactic,
    "solarsystem": solarsystem,
    "dimensionless": dimensionless,
}
