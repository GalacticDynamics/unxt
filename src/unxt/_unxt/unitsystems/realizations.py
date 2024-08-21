"""Realizations of unit systems."""

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
from .builtin_dimensions import speed

# Dimensionless. This is a singleton.
dimensionless = DimensionlessUnitSystem()

# Galactic unit system
galactic = LTMAUnitSystem(  # pylint: disable=no-member
    u.kpc, u.Myr, u.Msun, u.radian, preferred_units={speed: u.km / u.s}
)

# Solar system units
solarsystem = LTMAUnitSystem(u.au, u.yr, u.Msun, u.radian)  # pylint: disable=no-member


NAMED_UNIT_SYSTEMS: dict[str, AbstractUnitSystem] = {
    "galactic": galactic,
    "solarsystem": solarsystem,
    "dimensionless": dimensionless,
}
