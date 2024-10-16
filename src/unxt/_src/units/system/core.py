"""Tools for representing systems of units using ``astropy.units``."""

__all__ = ["unitsystem", "unitsystem_of"]


from collections.abc import Sequence
from dataclasses import field, make_dataclass
from typing import Annotated, Any

import astropy.units as u
import equinox as eqx
from plum import dispatch

from .base import UNITSYSTEMS_REGISTRY, AbstractUnitSystem
from .builtin import DimensionlessUnitSystem
from .flags import AbstractUnitSystemFlag, StandardUnitSystemFlag
from .realizations import NAMED_UNIT_SYSTEMS, dimensionless
from .utils import get_dimension_name
from unxt._src.dimensions.core import dimensions_of
from unxt._src.units.core import units

# ===================================================================
# `unitsystem` function


@dispatch
def unitsystem(usys: AbstractUnitSystem, /) -> AbstractUnitSystem:
    """Convert a UnitSystem or tuple of arguments to a UnitSystem.

    Examples
    --------
    >>> import astropy.units as u
    >>> from unxt.unitsystems import unitsystem
    >>> usys = unitsystem(u.kpc, u.Myr, u.Msun, u.radian)
    >>> usys
    unitsystem(kpc, Myr, solMass, rad)

    >>> unitsystem(usys) is usys
    True

    """
    return usys


@dispatch  # type: ignore[no-redef]
def unitsystem(units_: Sequence[Any], /) -> AbstractUnitSystem:
    """Convert a UnitSystem or tuple of arguments to a UnitSystem.

    Examples
    --------
    >>> import astropy.units as u
    >>> from unxt.unitsystems import unitsystem

    >>> unitsystem(())
    DimensionlessUnitSystem()

    >>> unitsystem((u.kpc, u.Myr, u.Msun, u.radian))
    unitsystem(kpc, Myr, solMass, rad)

    >>> unitsystem([u.kpc, u.Myr, u.Msun, u.radian])
    unitsystem(kpc, Myr, solMass, rad)

    """
    return unitsystem(*units_) if len(units_) > 0 else dimensionless


@dispatch  # type: ignore[no-redef]
def unitsystem(_: None, /) -> DimensionlessUnitSystem:
    """Dimensionless unit system from None.

    Examples
    --------
    >>> from unxt.unitsystems import unitsystem
    >>> unitsystem(None)
    DimensionlessUnitSystem()

    """
    return dimensionless


@dispatch  # type: ignore[no-redef]
def unitsystem(*units_: Any) -> AbstractUnitSystem:
    """Convert a set of arguments to a UnitSystem.

    Examples
    --------
    >>> import astropy.units as u
    >>> from unxt.unitsystems import unitsystem

    >>> unitsystem(u.kpc, u.Myr, u.Msun, u.radian)
    unitsystem(kpc, Myr, solMass, rad)

    """
    # Convert everything to a unit
    units_ = tuple(map(units, units_))

    # Check that the units all have different dimensions
    dimensions = tuple(map(dimensions_of, units_))
    dimensions = eqx.error_if(
        dimensions,
        len(set(dimensions)) < len(dimensions),
        "some dimensions are repeated",
    )

    # Return if the unit system is already registered
    if dimensions in UNITSYSTEMS_REGISTRY:
        return UNITSYSTEMS_REGISTRY[dimensions](*units_)

    # Otherwise, create a new unit system
    # dimension names of all the units
    du = {get_dimension_name(x).replace(" ", "_"): dimensions_of(x) for x in units_}
    # name: physical types
    cls_name = "".join(k.title().replace("_", "") for k in du) + "UnitSystem"
    # fields: name, unit
    fields = [
        (
            k,
            Annotated[u.UnitBase, v],
            field(init=True, repr=True, hash=True, compare=True),  # pylint: disable=invalid-field-call
        )
        for k, v in du.items()
    ]

    def _reduce_(self: AbstractUnitSystem) -> tuple:
        return (_call_unitsystem, self.base_units, None, None, None, None)

    # Make and register the dataclass class
    unitsystem_cls: type[AbstractUnitSystem] = make_dataclass(
        cls_name,
        fields,
        bases=(AbstractUnitSystem,),
        namespace={"__reduce__": _reduce_},
        frozen=True,
        slots=True,
        eq=True,
        repr=True,
        init=True,
    )

    # Make the dataclass instance
    return unitsystem_cls(*units_)


@dispatch  # type: ignore[no-redef]
def unitsystem(name: str, /) -> AbstractUnitSystem:
    """Return unit system from name.

    Examples
    --------
    >>> from unxt.unitsystems import unitsystem
    >>> unitsystem("galactic")
    unitsystem(kpc, Myr, solMass, rad)

    >>> unitsystem("solarsystem")
    unitsystem(AU, yr, solMass, rad)

    >>> unitsystem("dimensionless")
    DimensionlessUnitSystem()

    """
    return NAMED_UNIT_SYSTEMS[name]


@dispatch  # type: ignore[no-redef]
def unitsystem(usys: AbstractUnitSystem, *units_: Any) -> AbstractUnitSystem:
    """Create a unit system from an existing unit system and additional units.

    Examples
    --------
    We can add a new unit definition to an existing unit system:

    >>> import astropy.units as u
    >>> from unxt.unitsystems import unitsystem
    >>> usys = unitsystem("galactic")
    >>> unitsystem(usys, u.km/u.s)
    LengthTimeMassAngleSpeedUnitSystem(length=Unit("kpc"), time=Unit("Myr"), mass=Unit("solMass"), angle=Unit("rad"), speed=Unit("km / s"))

    We can also override the base unit of an existing unit system:

    >>> new_usys = unitsystem(usys, u.pc)
    >>> new_usys
    TimeMassAngleLengthUnitSystem(time=Unit("Myr"), mass=Unit("solMass"), angle=Unit("rad"), length=Unit("pc"))

    """  # noqa: E501
    new_usys = unitsystem(*units_)
    current_units = [
        unit
        for unit in usys.base_units
        if unit.physical_type not in new_usys.base_dimensions
    ]
    return unitsystem(*current_units, *units_)


@dispatch  # type: ignore[no-redef]
def unitsystem(flag: type[AbstractUnitSystemFlag], *_: Any) -> AbstractUnitSystem:
    msg = "Do not use the AbstractUnitSystemFlag directly, only use subclasses."
    raise ValueError(msg)


@dispatch  # type: ignore[no-redef]
def unitsystem(flag: type[StandardUnitSystemFlag], *units_: Any) -> AbstractUnitSystem:
    return unitsystem(*units_)


# ----


def _call_unitsystem(*args: Any) -> AbstractUnitSystem:
    return unitsystem(*args)


# ===================================================================
# `unitsystem_of` function


@dispatch.abstract
def unitsystem_of(obj: Any, /) -> AbstractUnitSystem:
    """Return the unit system of the object."""


@dispatch
def unitsystem_of(obj: Any, /) -> DimensionlessUnitSystem:
    """Return the unit system of the object.

    Examples
    --------
    >>> from unxt.unitsystems import unitsystem_of

    >>> unitsystem_of(1)
    DimensionlessUnitSystem()

    """
    return dimensionless


@dispatch
def unitsystem_of(obj: AbstractUnitSystem, /) -> AbstractUnitSystem:
    """Return the unit system from the unit system.

    Examples
    --------
    >>> from unxt.unitsystems import galactic, unitsystem_of

    >>> unitsystem_of(galactic) is galactic
    True

    """
    return obj
