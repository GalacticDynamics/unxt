"""Tools for representing systems of units using ``astropy.units``."""

__all__ = ("unitsystem", "unitsystem_of")


from collections.abc import Sequence
from dataclasses import field, make_dataclass
from typing import Annotated, Any

import equinox as eqx
import numpy as np
from astropy.constants import G as const_G  # noqa: N811, pylint: disable=E0611
from astropy.units import UnitBase as AstropyUnitBase
from plum import dispatch

from . import builtin_dimensions as ud
from .base import UNITSYSTEMS_REGISTRY, AbstractUnitSystem
from .builtin import DimensionlessUnitSystem
from .flags import AbstractUSysFlag, DynamicalSimUSysFlag, StandardUSysFlag
from .realizations import NAMED_UNIT_SYSTEMS, dimensionless
from .utils import parse_dimlike_name
from unxt.dims import dimension_of
from unxt.units import unit

# ===================================================================
# `unitsystem` function


@dispatch
def unitsystem(usys: AbstractUnitSystem, /) -> AbstractUnitSystem:
    """Convert a UnitSystem to a UnitSystem.

    Examples
    --------
    >>> from unxt.unitsystems import unitsystem
    >>> usys = unitsystem("kpc", "Myr", "Msun", "radian")
    >>> usys
    unitsystem(kpc, Myr, solMass, rad)

    >>> unitsystem(usys) is usys
    True

    """
    return usys


@dispatch
def unitsystem(seq: Sequence[Any], /) -> AbstractUnitSystem:
    """Convert a UnitSystem or tuple of arguments to a UnitSystem.

    Examples
    --------
    >>> import unxt as u

    >>> u.unitsystem(())
    DimensionlessUnitSystem()

    >>> u.unitsystem(("kpc", "Myr", "Msun", "radian"))
    unitsystem(kpc, Myr, solMass, rad)

    >>> u.unitsystem(["kpc", "Myr", "Msun", "radian"])
    unitsystem(kpc, Myr, solMass, rad)

    """
    return unitsystem(*seq) if len(seq) > 0 else dimensionless


@dispatch
def unitsystem(_: None, /) -> DimensionlessUnitSystem:
    """Dimensionless unit system from None.

    Examples
    --------
    >>> from unxt.unitsystems import unitsystem
    >>> unitsystem(None)
    DimensionlessUnitSystem()

    """
    return dimensionless


@dispatch
def unitsystem(*args: Any) -> AbstractUnitSystem:
    """Convert a set of arguments to a UnitSystem.

    Examples
    --------
    >>> from unxt.unitsystems import unitsystem

    >>> unitsystem("kpc", "Myr", "Msun", "radian")
    unitsystem(kpc, Myr, solMass, rad)

    """
    # Convert everything to a unit
    args = tuple(map(unit, args))

    # Check that the units all have different dimensions
    dims = tuple(map(dimension_of, args))
    dims = eqx.error_if(
        dims,
        len(set(dims)) < len(dims),
        "some dimensions are repeated",
    )

    # Return if the unit system is already registered
    if dims in UNITSYSTEMS_REGISTRY:
        return UNITSYSTEMS_REGISTRY[dims](*args)

    # Otherwise, create a new unit system
    # dimension names of all the units
    du = {parse_dimlike_name(x).replace(" ", "_"): dimension_of(x) for x in args}
    # name: physical types
    cls_name = "".join(k.title().replace("_", "") for k in du) + "UnitSystem"  # pylint: disable=unreachable
    # fields: name, unit
    fields = [
        (
            k,
            Annotated[AstropyUnitBase, v],
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
    return unitsystem_cls(*args)


@dispatch
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


@dispatch
def unitsystem(usys: AbstractUnitSystem, *args: Any) -> AbstractUnitSystem:
    """Create a unit system from an existing unit system and additional units.

    Examples
    --------
    We can add a new unit definition to an existing unit system:

    >>> from unxt.unitsystems import unitsystem
    >>> usys = unitsystem("galactic")
    >>> unitsystem(usys, "km/s")
    LengthTimeMassAngleSpeedUnitSystem(length=Unit("kpc"), time=Unit("Myr"), mass=Unit("solMass"), angle=Unit("rad"), speed=Unit("km / s"))

    We can also override the base unit of an existing unit system:

    >>> new_usys = unitsystem(usys, "pc")
    >>> new_usys
    TimeMassAngleLengthUnitSystem(time=Unit("Myr"), mass=Unit("solMass"), angle=Unit("rad"), length=Unit("pc"))

    """  # noqa: E501
    # TODO: not need this hack for single-string inputs
    # TODO: process new units without making a whole unit system
    if len(args) == 1 and isinstance(args[0], str):
        new_usys = unitsystem(unit(args[0]))
    else:
        new_usys = unitsystem(*args)

    current_units = [
        u for u in usys.base_units if dimension_of(u) not in new_usys.base_dimensions
    ]
    return unitsystem(*current_units, *args)  # pylint: disable=unreachable


@dispatch
def unitsystem(flag: type[AbstractUSysFlag], *_: Any) -> AbstractUnitSystem:
    """Raise an exception since the flag is abstract."""
    msg = "Do not use the AbstractUSysFlag directly, only use subclasses."
    raise TypeError(msg)


@dispatch
def unitsystem(flag: type[StandardUSysFlag], *args: Any) -> AbstractUnitSystem:
    """Create a standard unit system using the inputted units.

    Examples
    --------
    >>> from unxt import unitsystem, unitsystems
    >>> unitsystem(unitsystems.StandardUSysFlag, "kpc", "Myr", "Msun")
    LengthTimeMassUnitSystem(length=Unit("kpc"), time=Unit("Myr"), mass=Unit("solMass"))

    """
    return unitsystem(*args)


@dispatch
def unitsystem(
    flag: type[DynamicalSimUSysFlag],
    *args: Any,
    G: float | int = 1.0,  # noqa: N803
) -> AbstractUnitSystem:
    """Make a dynamical unit system.

    Examples
    --------
    >>> from unxt.unitsystems import unitsystem, DynamicalSimUSysFlag

    >>> unitsystem(DynamicalSimUSysFlag, "m", "kg")
    LengthMassTimeUnitSystem(length=Unit("m"), mass=Unit("kg"), time=Unit("122404 s"))

    """
    tmp = unitsystem(*args)

    # Use G for computing the missing units below:
    G = G * const_G

    added = ()
    if ud.length in tmp.base_dimensions and ud.mass in tmp.base_dimensions:
        time = 1 / np.sqrt(G * tmp["mass"] / tmp["length"] ** 3)
        added = (time,)
    elif ud.length in tmp.base_dimensions and ud.time in tmp.base_dimensions:
        mass = 1 / G * tmp["length"] ** 3 / tmp["time"] ** 2
        added = (mass,)
    elif ud.length in tmp.base_dimensions and ud.speed in tmp.base_dimensions:
        time = tmp["length"] / tmp["velocity"]
        mass = tmp["velocity"] ** 2 / G * tmp["length"]
        added = (time, mass)
    elif ud.mass in tmp.base_dimensions and ud.time in tmp.base_dimensions:
        length = np.cbrt(G * tmp["mass"] * tmp["time"] ** 2)
        added = (length,)
    elif ud.mass in tmp.base_dimensions and ud.speed in tmp.base_dimensions:
        length = G * tmp["mass"] / tmp["velocity"] ** 2
        time = length / tmp["velocity"]
        added = (length, time)
    elif ud.time in tmp.base_dimensions and ud.speed in tmp.base_dimensions:
        mass = 1 / G * tmp["velocity"] ** 3 * tmp["time"]
        length = G * mass / tmp["velocity"] ** 2
        added = (mass, length)

    return unitsystem(*tmp, *added)


# ----


def _call_unitsystem(*args: Any) -> AbstractUnitSystem:
    return unitsystem(*args)


# ===================================================================
# `unitsystem_of` function


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
