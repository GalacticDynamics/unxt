"""Realizations of unit systems."""
# pylint: disable=no-member

__all__ = [
    # unit system instance
    "galactic",
    "dimensionless",
    "solarsystem",
    "si",
    "cgs",
    # unit system alias
    "NAMED_UNIT_SYSTEMS",
]

from .base import AbstractUnitSystem
from .builtin import (
    CGSUnitSystem,
    DimensionlessUnitSystem,
    LTMAUnitSystem,
    SIUnitSystem,
)
from unxt._src.units.core import units

# Dimensionless. This is a singleton.
dimensionless = DimensionlessUnitSystem()

# Galactic unit system
galactic = LTMAUnitSystem(
    length=units("kpc"),
    time=units("Myr"),
    mass=units("Msun"),
    angle=units("radian"),
)

# Solar system units
solarsystem = LTMAUnitSystem(
    length=units("au"), time=units("yr"), mass=units("Msun"), angle=units("radian")
)

#: International System of Units
si = SIUnitSystem(
    length=units("meter"),
    time=units("second"),
    mass=units("kilogram"),
    electric_current=units("ampere"),
    temperature=units("Kelvin"),
    amount=units("mole"),
    luminous_intensity=units("candela"),
    angle=units("radian"),
)

#: Centimeter, gram, second unit system
cgs = CGSUnitSystem(
    length=units("centimeter"),
    time=units("second"),
    mass=units("gram"),
    angle=units("radian"),
    force=units("dyne"),
    energy=units("erg"),
    pressure=units("barye"),
    dynamic_viscosity=units("poise"),
    kinematic_viscosity=units("stokes"),
)


#: Named unit systems
NAMED_UNIT_SYSTEMS: dict[str, AbstractUnitSystem] = {
    "galactic": galactic,
    "solarsystem": solarsystem,
    "dimensionless": dimensionless,
    "si": si,
    "cgs": cgs,
}
