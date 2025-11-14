"""Tools for representing systems of units using ``astropy.units``."""
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

from .base import UNITSYSTEMS_REGISTRY, AbstractUnitSystem
from .builtin import DimensionlessUnitSystem, LTMAUnitSystem
from .compare import equivalent
from .core import unitsystem, unitsystem_of
from .flags import AbstractUSysFlag, DynamicalSimUSysFlag, StandardUSysFlag
from .realizations import (
    NAMED_UNIT_SYSTEMS,
    cgs,
    dimensionless,
    galactic,
    si,
    solarsystem,
)
