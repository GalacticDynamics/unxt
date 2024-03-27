"""Tools for representing systems of units using ``astropy.units``."""

__all__ = [
    # unit system instance
    "galactic",
    "dimensionless",
    "solarsystem",
    # unit system alias
    "NAMED_UNIT_SYSTEMS",
]


import astropy.units as u

from ._base import AbstractUnitSystem
from ._core import DimensionlessUnitSystem, UnitSystem

# define galactic unit system
galactic = UnitSystem(u.kpc, u.Myr, u.Msun, u.radian, u.km / u.s)  # pylint: disable=no-member

# solar system units
solarsystem = UnitSystem(u.au, u.M_sun, u.yr, u.radian)  # pylint: disable=no-member

# dimensionless
dimensionless = DimensionlessUnitSystem()


NAMED_UNIT_SYSTEMS: dict[str, AbstractUnitSystem] = {
    "galactic": galactic,
    "solarsystem": solarsystem,
    "dimensionless": dimensionless,
}
