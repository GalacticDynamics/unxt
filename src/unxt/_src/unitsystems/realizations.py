"""Realizations of unit systems."""
# pylint: disable=no-member

__all__ = (
    # unit system instance
    "galactic",
    "dimensionless",
    "solarsystem",
    "si",
    "cgs",
    # unit system alias
    "NAMED_UNIT_SYSTEMS",
)

from .base import AbstractUnitSystem
from .builtin import (
    CGSUnitSystem,
    DimensionlessUnitSystem,
    LTMAUnitSystem,
    SIUnitSystem,
)
from unxt.units import unit

# Dimensionless. This is a singleton.
dimensionless = DimensionlessUnitSystem()

# Galactic unit system
galactic = LTMAUnitSystem(
    length=unit("kpc"),
    time=unit("Myr"),
    mass=unit("Msun"),
    angle=unit("radian"),
)

# Solar system units
solarsystem = LTMAUnitSystem(
    length=unit("au"), time=unit("yr"), mass=unit("Msun"), angle=unit("radian")
)

#: International System of Units
si = SIUnitSystem(
    length=unit("meter"),
    time=unit("second"),
    mass=unit("kilogram"),
    electric_current=unit("ampere"),
    temperature=unit("Kelvin"),
    amount=unit("mole"),
    luminous_intensity=unit("candela"),
    angle=unit("radian"),
)

#: Centimeter, gram, second unit system
cgs = CGSUnitSystem(
    length=unit("centimeter"),
    time=unit("second"),
    mass=unit("gram"),
    angle=unit("radian"),
    force=unit("dyne"),
    energy=unit("erg"),
    pressure=unit("barye"),
    dynamic_viscosity=unit("poise"),
    kinematic_viscosity=unit("stokes"),
)


#: Named unit systems
NAMED_UNIT_SYSTEMS: dict[str, AbstractUnitSystem] = {
    "galactic": galactic,
    "solarsystem": solarsystem,
    "dimensionless": dimensionless,
    "si": si,
    "cgs": cgs,
}
