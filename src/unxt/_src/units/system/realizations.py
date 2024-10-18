"""Realizations of unit systems."""
# pylint: disable=no-member

__all__ = [
    # unit system instance
    "galactic",
    "dimensionless",
    "solarsystem",
    "si",
    # unit system alias
    "NAMED_UNIT_SYSTEMS",
]

import astropy.units as u

from .base import AbstractUnitSystem
from .builtin import (
    CGSUnitSystem,
    DimensionlessUnitSystem,
    LTMAUnitSystem,
    SIUnitSystem,
)

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

# International System of Units default
si = SIUnitSystem(
    length=u.meter,
    time=u.second,
    mass=u.kilogram,
    electric_current=u.ampere,
    temperature=u.Kelvin,
    amount=u.mole,
    luminous_intensity=u.candela,
    angle=u.radian,
)

# Centimeter, gram, second
cgs = CGSUnitSystem(
    length=u.centimeter,
    time=u.second,
    mass=u.gram,
    angle=u.radian,
    force=u.dyne,
    energy=u.erg,
    pressure=u.barye,
    dynamic_viscosity=u.poise,
    kinematic_viscosity=u.stokes,
)


NAMED_UNIT_SYSTEMS: dict[str, AbstractUnitSystem] = {
    "galactic": galactic,
    "solarsystem": solarsystem,
    "dimensionless": dimensionless,
    "si": si,
    "cgs": cgs,
}
