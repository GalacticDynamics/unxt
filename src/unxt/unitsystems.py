"""Tools for representing systems of units using ``astropy.units``."""

__all__ = [
    # unit system classes
    "UnitSystem",
    "DimensionlessUnitSystem",
    # general unit system constructor
    "unitsystem",
    # unit system instance
    "galactic",
    "dimensionless",
    "solarsystem",
    # unit system alias
    "NAMED_UNIT_SYSTEMS",
]

from collections.abc import Iterator
from typing import ClassVar, Union, cast, final

import astropy.units as u
from astropy.units.physical import _physical_unit_mapping
from plum import dispatch

from ._base import AbstractQuantity
from ._core import Quantity
from ._typing import Unit


class UnitSystem:
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

    _core_units: list[Unit]
    _registry: dict[u.PhysicalType, Unit]

    _required_dimensions: ClassVar[list[u.PhysicalType]] = [
        u.get_physical_type("length"),
        u.get_physical_type("time"),
        u.get_physical_type("mass"),
        u.get_physical_type("angle"),
    ]

    def __init__(
        self,
        units: Union[Unit, u.Quantity, AbstractQuantity, "UnitSystem"],
        *args: Unit | u.Quantity | AbstractQuantity,
    ) -> None:
        if isinstance(units, UnitSystem):
            if len(args) > 0:
                msg = "If passing in a UnitSystem, cannot pass in additional units."
                raise ValueError(msg)

            self._registry = units._registry.copy()  # noqa: SLF001
            self._core_units = units._core_units  # noqa: SLF001
            return

        units = (units, *args)

        self._registry = {}
        for unit in units:
            unit_ = (  # TODO: better detection of allowed unit base classes
                unit if isinstance(unit, u.UnitBase) else u.def_unit(f"{unit!s}", unit)
            )
            if unit_.physical_type in self._registry:
                msg = f"Multiple units passed in with type {unit_.physical_type!r}"
                raise ValueError(msg)
            self._registry[unit_.physical_type] = unit_

        self._core_units = []
        for phys_type in self._required_dimensions:
            if phys_type not in self._registry:
                msg = f"You must specify a unit for the physical type {phys_type!r}"
                raise ValueError(msg)
            self._core_units.append(self._registry[phys_type])

    def __getitem__(self, key: str | u.PhysicalType) -> u.UnitBase:
        key = u.get_physical_type(key)
        if key in self._required_dimensions:
            return self._registry[key]

        unit = None
        for k, v in _physical_unit_mapping.items():
            if v == key:
                unit = u.Unit(" ".join([f"{x}**{y}" for x, y in k]))
                break

        if unit is None:
            msg = f"Physical type '{key}' doesn't exist in unit registry."
            raise ValueError(msg)

        unit = unit.decompose(self._core_units)
        unit._scale = 1.0  # noqa: SLF001
        return unit

    def __len__(self) -> int:
        # Note: This is required for q.decompose(usys) to work, where q is a Quantity
        return len(self._core_units)

    def __iter__(self) -> Iterator[Unit]:
        yield from self._core_units

    def __repr__(self) -> str:
        return f"UnitSystem({', '.join(str(uu) for uu in self._core_units)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UnitSystem):
            return NotImplemented
        return bool(self._registry == other._registry)

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        """Hash the unit system."""
        return hash(tuple(self._core_units) + tuple(self._required_dimensions))

    def preferred(self, key: str | u.PhysicalType) -> Unit:
        """Return the preferred unit for a given physical type."""
        key = u.get_physical_type(key)
        if key in self._registry:
            return self._registry[key]
        return self[key]

    def as_preferred(self, quantity: AbstractQuantity | u.Quantity) -> Quantity:
        """Convert a quantity to the preferred unit for this unit system."""
        unit = self.preferred(quantity.unit.physical_type)
        return cast(AbstractQuantity, Quantity.constructor(quantity.to(unit), unit))


@final
class DimensionlessUnitSystem(UnitSystem):
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


# define galactic unit system
galactic = UnitSystem(u.kpc, u.Myr, u.Msun, u.radian, u.km / u.s)  # pylint: disable=no-member

# solar system units
solarsystem = UnitSystem(u.au, u.M_sun, u.yr, u.radian)  # pylint: disable=no-member

# dimensionless
dimensionless = DimensionlessUnitSystem()


NAMED_UNIT_SYSTEMS: dict[str, UnitSystem] = {
    "galactic": galactic,
    "solarsystem": solarsystem,
    "dimensionless": dimensionless,
}


# ===========================
# Unit-system constructor


@dispatch
def unitsystem(units: UnitSystem, /) -> UnitSystem:
    """Convert a UnitSystem or tuple of arguments to a UnitSystem.

    Examples
    --------
    >>> import astropy.units as u
    >>> from unxt.unitsystems import UnitSystem, unitsystem
    >>> usys = UnitSystem(u.kpc, u.Myr, u.Msun, u.radian, u.km/u.s)
    >>> usys
    UnitSystem(kpc, Myr, solMass, rad)

    >>> unitsystem(usys)
    UnitSystem(kpc, Myr, solMass, rad)

    """
    return units


@dispatch  # type: ignore[no-redef]
def unitsystem(
    units: (
        tuple[Unit | u.Quantity | AbstractQuantity, ...]
        | list[Unit | u.Quantity | AbstractQuantity]
    ),
    /,
) -> UnitSystem:
    """Convert a UnitSystem or tuple of arguments to a UnitSystem.

    Examples
    --------
    >>> import astropy.units as u
    >>> from unxt.unitsystems import UnitSystem, unitsystem

    >>> unitsystem((u.kpc, u.Myr, u.Msun, u.radian, u.km/u.s))
    UnitSystem(kpc, Myr, solMass, rad)

    >>> unitsystem([u.kpc, u.Myr, u.Msun, u.radian, u.km/u.s])
    UnitSystem(kpc, Myr, solMass, rad)

    """
    return UnitSystem(*units) if len(units) > 0 else dimensionless


@dispatch  # type: ignore[no-redef]
def unitsystem(_: None, /) -> UnitSystem:
    """Dimensionless unit system from None.

    Examples
    --------
    >>> from unxt.unitsystems import unitsystem
    >>> unitsystem(None)
    DimensionlessUnitSystem()

    """
    return dimensionless


@dispatch  # type: ignore[no-redef]
def unitsystem(unit0: Unit, /, *units: Unit) -> UnitSystem:
    """Convert a set of arguments to a UnitSystem.

    Examples
    --------
    >>> import astropy.units as u
    >>> from unxt.unitsystems import UnitSystem, unitsystem

    >>> unitsystem(u.kpc, u.Myr, u.Msun, u.radian)
    UnitSystem(kpc, Myr, solMass, rad)

    """
    return UnitSystem(unit0, *units)


@dispatch  # type: ignore[no-redef]
def unitsystem(name: str, /) -> UnitSystem:
    """Return unit system from name.

    Examples
    --------
    >>> from unxt.unitsystems import unitsystem
    >>> unitsystem("galactic")
    UnitSystem(kpc, Myr, solMass, rad)

    >>> unitsystem("solarsystem")
    UnitSystem(AU, yr, solMass, rad)

    >>> unitsystem("dimensionless")
    DimensionlessUnitSystem()

    """
    return NAMED_UNIT_SYSTEMS[name]
