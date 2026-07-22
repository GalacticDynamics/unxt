"""Realizations of unit systems."""
# pylint: disable=no-member

__all__ = (
    # unit system instance
    "galactic",
    "dimensionless",
    "solarsystem",
    "si",
    "cgs",
    # natural unit systems
    "hep",
    "geometrized",
    "planck",
    "atomic",
    # unit system alias
    "NAMED_UNIT_SYSTEMS",
)

from .base import AbstractUnitSystem
from .builtin import (
    CGSUnitSystem,
    LTMAUnitSystem,
    SIUnitSystem,
    dimensionless,
)
from .core import unitsystem
from .flags import (
    AtomicUSysFlag,
    GeometrizedUSysFlag,
    HEPUSysFlag,
    PlanckUSysFlag,
)
from unxt.units import unit

# ``dimensionless`` is re-exported from ``builtin`` (it is defined alongside its
# class so ``core`` can use it without a circular import).

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


# ---------------------------------------------------------------
# Natural unit systems
#
# Each sets a set of fundamental constants to 1 (see the corresponding flag).
# The free-scale systems use their default scale (HEP: 1 GeV; geometrized: 1 m).

#: High-energy-physics units (hbar = c = 1), at the 1 GeV energy scale.
hep = unitsystem(HEPUSysFlag)

#: Geometrized units (c = G = 1), at the 1 meter length scale.
geometrized = unitsystem(GeometrizedUSysFlag)

#: Planck units (hbar = c = G = k_B = 1).
planck = unitsystem(PlanckUSysFlag)

#: Atomic (Hartree) units (m_e = hbar = e = 4*pi*eps0 = 1).
atomic = unitsystem(AtomicUSysFlag)


#: Named unit systems
NAMED_UNIT_SYSTEMS: dict[str, AbstractUnitSystem] = {
    "galactic": galactic,
    "solarsystem": solarsystem,
    "dimensionless": dimensionless,
    "si": si,
    "cgs": cgs,
    "hep": hep,
    "geometrized": geometrized,
    "planck": planck,
    "atomic": atomic,
}
