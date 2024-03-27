"""Tools for representing systems of units using ``astropy.units``."""

__all__ = ["UnitSystem", "DimensionlessUnitSystem"]

from typing import ClassVar, final

import astropy.units as u

from ._base import AbstractUnitSystem
from unxt._typing import Unit


@final
class UnitSystem(AbstractUnitSystem):
    """Represents a system of units.

    At minimum, this consists of a set of length, time, mass, and angle units, but may
    also contain preferred representations for composite units. For example, the base
    unit system could be ``{kpc, Myr, Msun, radian}``, but you can also specify a
    preferred velocity unit, such as ``km/s``.

    This class behaves like a dictionary with keys set by physical types (i.e. "length",
    "velocity", "energy", etc.). If a unit for a particular physical type is not
    specified on creation, a composite unit will be created with the base units. See the
    examples below for some demonstrations.

    Parameters
    ----------
    *units, **units
        The units that define the unit system. At minimum, this must contain length,
        time, mass, and angle units. If passing in keyword arguments, the keys must be
        valid :mod:`astropy.units` physical types.

    Examples
    --------
    If only base units are specified, any physical type specified as a key
    to this object will be composed out of the base units::

        >>> usys = UnitSystem(u.m, u.s, u.kg, u.radian)
        >>> usys["velocity"]
        Unit("m / s")

    However, preferred representations for composite units can also be specified::

        >>> usys = UnitSystem(u.m, u.s, u.kg, u.radian, u.erg)
        >>> usys["energy"]
        Unit("m2 kg / s2")
        >>> usys.preferred("energy")
        Unit("erg")

    This is useful for Galactic dynamics where lengths and times are usually given in
    terms of ``kpc`` and ``Myr``, but velocities are often specified in ``km/s``::

        >>> usys = UnitSystem(u.kpc, u.Myr, u.Msun, u.radian, u.km/u.s)
        >>> usys["velocity"]
        Unit("kpc / Myr")
        >>> usys.preferred("velocity")
        Unit("km / s")

    """

    _required_dimensions: ClassVar[list[u.PhysicalType]] = [
        u.get_physical_type("length"),
        u.get_physical_type("time"),
        u.get_physical_type("mass"),
        u.get_physical_type("angle"),
    ]


@final
class DimensionlessUnitSystem(AbstractUnitSystem):
    """A unit system with only dimensionless units."""

    _required_dimensions: ClassVar[list[u.PhysicalType]] = []

    def __init__(self) -> None:
        super().__init__(u.one)
        self._core_units = [u.one]

    def __getitem__(self, key: str | u.PhysicalType) -> Unit:
        return u.one

    def __str__(self) -> str:
        return "UnitSystem(dimensionless)"

    def __repr__(self) -> str:
        return "DimensionlessUnitSystem()"
