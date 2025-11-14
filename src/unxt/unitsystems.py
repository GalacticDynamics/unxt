"""Systems of units.

A unit system is a collection of units that are used together. In a unit system
there are base units and derived units. Base units are the fundamental units of
the system and derived units are constructed from the base units. For example,
in the SI system, the base units are the meter, kilogram, second, ampere,
kelvin, mole, and candela. Derived units are constructed from these base units,
for example, the newton is a derived unit of force.

`unxt` provides powerful tools for defining and working with unit systems. Unit
systems can be statically defined (useful for many tools and development
environments) or dynamically defined (useful for interactive environments and
Python's general dynamism). Unit systems can be extended, compared, and used for
decomposing units on quantities. There are many more features and tools for
working with unit systems in `unxt`.

"""
# pylint: disable=duplicate-code

__all__ = (
    "unitsystem",
    "unitsystem_of",
    # classes
    "AbstractUnitSystem",
    "UNITSYSTEMS_REGISTRY",
    "DimensionlessUnitSystem",
    "LTMAUnitSystem",
    # unit system instance
    "galactic",
    "dimensionless",
    "solarsystem",
    "si",
    "cgs",
    # unit system alias
    "NAMED_UNIT_SYSTEMS",
    # unit system flags
    "AbstractUSysFlag",
    "StandardUSysFlag",
    "DynamicalSimUSysFlag",
    # functions
    "equivalent",
)

from .setup_package import install_import_hook

with install_import_hook("unxt.unitsystems"):
    from ._src.unitsystems import (
        NAMED_UNIT_SYSTEMS,
        UNITSYSTEMS_REGISTRY,
        AbstractUnitSystem,
        AbstractUSysFlag,
        DimensionlessUnitSystem,
        DynamicalSimUSysFlag,
        LTMAUnitSystem,
        StandardUSysFlag,
        cgs,
        dimensionless,
        equivalent,
        galactic,
        si,
        solarsystem,
        unitsystem,
        unitsystem_of,
    )

# Clean up the namespace
del install_import_hook
