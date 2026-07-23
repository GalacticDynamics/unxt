"""Tools for representing systems of units using ``astropy.units``."""

__all__ = ("unitsystem", "unitsystem_of")


from collections.abc import Sequence
from dataclasses import field, make_dataclass
from typing import Annotated, Any

import equinox as eqx
import jax.tree_util as jtu
import numpy as np
from astropy.constants import (  # pylint: disable=E0611
    G as const_G,  # noqa: N811
    Ryd as const_Ryd,
    a0 as const_a0,
    c as const_c,
    e as const_e,
    h as const_h,
    hbar as const_hbar,
    k_B as const_kB,
    m_e as const_me,
)
from astropy.units import UnitBase as AstropyUnitBase
from plum import dispatch

from . import base, builtin_dimensions as ud
from .base import AbstractUnitSystem
from .builtin import DimensionlessUnitSystem, dimensionless
from .flags import (
    AbstractUSysFlag,
    AtomicUSysFlag,
    DynamicalSimUSysFlag,
    GeometrizedUSysFlag,
    HEPUSysFlag,
    PlanckUSysFlag,
    StandardUSysFlag,
)
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
    """Build a unit system from a sequence of units.

    Examples
    --------
    >>> import unxt as u

    >>> u.unitsystem(())
    DimensionlessUnitSystem()

    >>> u.unitsystem(("kpc", "Myr", "Msun", "radian"))
    unitsystem(kpc, Myr, solMass, rad)

    >>> u.unitsystem(["kpc", "Myr", "Msun", "radian"])
    unitsystem(kpc, Myr, solMass, rad)

    A sequence's elements are always units, so even a single-element list builds
    a system (a bare string, by contrast, is looked up as a *named* system):

    >>> u.unitsystem(["km"])
    LengthUnitSystem(length=Unit("km"))

    """
    # Elements of an explicit sequence are units. Convert them up front so a
    # single-element list dispatches to the unit-building path below rather than
    # to the ``str`` named-system lookup (``unitsystem(["km"])`` must not become
    # ``unitsystem("km")``).
    return unitsystem(*map(unit, seq)) if len(seq) > 0 else dimensionless


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

    With no arguments it is the dimensionless system, agreeing with
    ``unitsystem(None)`` and ``unitsystem([])``:

    >>> unitsystem()
    DimensionlessUnitSystem()

    """
    # No units -> the dimensionless system. Building an empty ``UnitSystem()``
    # here would be a distinct, non-``dimensionless`` class that then answers
    # every derived-dimension lookup with a silent SI default. Mirror the empty
    # ``Sequence`` dispatch, which already returns the dimensionless singleton.
    if not args:
        return dimensionless

    # Convert everything to a unit
    args = tuple(map(unit, args))

    # Check that the units all have different dimensions
    dims = tuple(map(dimension_of, args))
    dims = eqx.error_if(
        dims,
        len(set(dims)) < len(dims),
        "some dimensions are repeated",
    )

    # A unit system is identified by its *set* of dimensions, not the argument
    # order. Look up a registered class (a built-in like ``LTMAUnitSystem`` or a
    # previously-created dynamic one) by that set -- an O(1) hit on the by-set
    # index -- and build it in that class's own field order. This keeps the
    # built-ins in their conventional order (galactic stays length, time, mass,
    # angle) while making construction order-independent.
    #
    # Read the index through the ``base`` module (not a ``from base import``
    # binding) so that registration -- which mutates ``base._UNITSYSTEMS_BY_DIMSET``
    # -- and this lookup always agree on the same object, including when a test
    # swaps the module attribute for an isolated one.
    unit_by_dim = dict(zip(dims, args, strict=True))
    reg_cls = base._UNITSYSTEMS_BY_DIMSET.get(frozenset(dims))  # noqa: SLF001
    if reg_cls is not None:
        return reg_cls(*(unit_by_dim[d] for d in reg_cls._base_dimensions))  # noqa: SLF001

    # Otherwise, create a new unit system. Sort the units by dimension name so
    # the field order -- and hence the class identity -- is deterministic and
    # independent of the argument order.
    args = tuple(sorted(args, key=parse_dimlike_name))
    # dimension names of all the units
    du = {parse_dimlike_name(x).replace(" ", "_"): dimension_of(x) for x in args}
    # name: physical types
    cls_name = "".join(k.title().replace("_", "") for k in du) + "UnitSystem"
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

    # Register the dynamically created unit system as a static PyTree node
    jtu.register_static(unitsystem_cls)

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

    >>> unitsystem("planck")
    LengthMassTimeTemperatureUnitSystem(length=Unit("...e-35 m"), mass=Unit("...e-08 kg"), time=Unit("...e-44 s"), temperature=Unit("...e+32 K"))

    A single string is looked up as a *system* name, not a unit. A string that
    is not a registered system -- e.g. a unit like ``"m"`` -- raises a clear
    error that names the registered systems and points to the unit path:

    >>> try:
    ...     unitsystem("m")
    ... except ValueError as e:
    ...     print(e)
    'm' is not a registered unit system (available: ...). If you meant a unit,
    convert it first, e.g. unitsystem(unxt.unit('m')).

    """  # noqa: E501
    # Imported lazily to avoid a circular import: `realizations` is built on top
    # of the `unitsystem` factory defined in this module.
    from .realizations import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
        NAMED_UNIT_SYSTEMS,
    )

    try:
        return NAMED_UNIT_SYSTEMS[name]
    except KeyError:
        available = ", ".join(sorted(NAMED_UNIT_SYSTEMS))
        msg = (
            f"{name!r} is not a registered unit system (available: {available}). "
            f"If you meant a unit, convert it first, e.g. "
            f"unitsystem(unxt.unit({name!r}))."
        )
        raise ValueError(msg) from None


@dispatch
def unitsystem(usys: AbstractUnitSystem, *args: Any) -> AbstractUnitSystem:
    """Create a unit system from an existing unit system and additional units.

    Examples
    --------
    We can add a new unit definition to an existing unit system:

    >>> from unxt.unitsystems import unitsystem
    >>> usys = unitsystem("galactic")
    >>> unitsystem(usys, "km/s")
    AngleLengthMassSpeedTimeUnitSystem(angle=Unit("rad"), length=Unit("kpc"), mass=Unit("solMass"), speed=Unit("km / s"), time=Unit("Myr"))

    We can also override the base unit of an existing unit system. Replacing the
    length still leaves a length/time/mass/angle system, so it is recognized as
    the built-in ``LTMAUnitSystem`` shape:

    >>> new_usys = unitsystem(usys, "pc")
    >>> new_usys
    unitsystem(pc, Myr, solMass, rad)

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
    return unitsystem(*current_units, *args)


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
    LengthMassTimeUnitSystem(length=Unit("kpc"), mass=Unit("solMass"), time=Unit("Myr"))

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
# Natural unit systems.
#
# Each of these generalizes the `DynamicalSimUSysFlag` construction: a set of
# fundamental constants is fixed to the dimensionless value 1 by choosing the
# base units accordingly, then the resulting units are handed to the standard
# `unitsystem(*units)` constructor. Constants come from `astropy.constants`;
# `.decompose()` reduces each computed quantity to (scaled) SI base units, and
# `unit(<Quantity>)` folds each into a scaled unit so the systems have clean,
# predictable reprs.


def _reject_extra_args(flag: type[AbstractUSysFlag], args: tuple[Any, ...]) -> None:
    """Fail fast if a natural-unit flag is given unexpected positional arguments.

    Natural unit systems are either fully determined or take their single free
    scale as a keyword (``energy`` / ``length``), so any positional argument
    beyond the flag is a caller mistake rather than something to silently ignore.
    """
    if args:
        msg = (
            f"`{flag.__name__}` does not take positional arguments beyond the "
            f"flag; got {args!r}. Pass the scale as a keyword if applicable "
            f"(e.g. `energy=...` or `length=...`)."
        )
        raise TypeError(msg)


@dispatch
def unitsystem(
    flag: type[HEPUSysFlag],
    /,
    *args: Any,
    energy: Any = "GeV",
) -> AbstractUnitSystem:
    """Make a high-energy-physics unit system: ``hbar = c = 1``.

    One free scale remains, the keyword-only ``energy`` (default 1 GeV).

    Examples
    --------
    >>> from unxt.unitsystems import unitsystem, HEPUSysFlag

    >>> unitsystem(HEPUSysFlag)
    LengthMassTimeUnitSystem(length=Unit("...e-16 m"), mass=Unit("...e-27 kg"), time=Unit("...e-25 s"))

    >>> unitsystem(HEPUSysFlag, energy="TeV")["time"]
    Unit("...e-28 s")

    """  # noqa: E501
    _reject_extra_args(flag, args)
    e_scale = 1.0 * unit(energy)
    mass = (e_scale / const_c**2).decompose()
    length = (const_hbar * const_c / e_scale).decompose()
    time = (const_hbar / e_scale).decompose()
    return unitsystem(length, mass, time)


@dispatch
def unitsystem(
    flag: type[GeometrizedUSysFlag],
    /,
    *args: Any,
    length: Any = "m",
) -> AbstractUnitSystem:
    """Make a geometrized (general-relativity) unit system: ``c = G = 1``.

    One free scale remains, the keyword-only ``length`` (default 1 meter).

    Examples
    --------
    >>> from unxt.unitsystems import unitsystem, GeometrizedUSysFlag

    >>> unitsystem(GeometrizedUSysFlag)
    LengthMassTimeUnitSystem(length=Unit("m"), mass=Unit("...e+27 kg"), time=Unit("...e-09 s"))

    >>> unitsystem(GeometrizedUSysFlag, length="km")["length"]
    Unit("km")

    """  # noqa: E501
    _reject_extra_args(flag, args)
    length_scale = 1.0 * unit(length)
    time = (length_scale / const_c).decompose()
    mass = (const_c**2 * length_scale / const_G).decompose()
    return unitsystem(length_scale, mass, time)


@dispatch
def unitsystem(flag: type[PlanckUSysFlag], /, *args: Any) -> AbstractUnitSystem:
    """Make a Planck unit system: ``hbar = c = G = k_B = 1``.

    Fully determined -- there are no free scales. The base units are the Planck
    length, mass, time, and temperature.

    Examples
    --------
    >>> from unxt.unitsystems import unitsystem, PlanckUSysFlag

    >>> unitsystem(PlanckUSysFlag)
    LengthMassTimeTemperatureUnitSystem(length=Unit("...e-35 m"), mass=Unit("...e-08 kg"), time=Unit("...e-44 s"), temperature=Unit("...e+32 K"))

    """  # noqa: E501
    _reject_extra_args(flag, args)
    length = np.sqrt(const_hbar * const_G / const_c**3).decompose()
    mass = np.sqrt(const_hbar * const_c / const_G).decompose()
    time = np.sqrt(const_hbar * const_G / const_c**5).decompose()
    temperature = (np.sqrt(const_hbar * const_c**5 / const_G) / const_kB).decompose()
    return unitsystem(
        length,
        mass,
        time,
        temperature,
    )


@dispatch
def unitsystem(flag: type[AtomicUSysFlag], /, *args: Any) -> AbstractUnitSystem:
    """Make an atomic (Hartree) unit system: ``m_e = hbar = e = 4*pi*eps0 = 1``.

    Fully determined -- there are no free scales. The base units are the Bohr
    radius (length), electron mass, the atomic unit of time (``hbar / E_h`` with
    ``E_h`` the Hartree energy), and the elementary charge.

    Examples
    --------
    >>> from unxt.unitsystems import unitsystem, AtomicUSysFlag

    >>> unitsystem(AtomicUSysFlag)
    LengthMassTimeElectricalChargeUnitSystem(length=Unit("...e-11 m"), mass=Unit("...e-31 kg"), time=Unit("...e-17 s"), electrical_charge=Unit("...e-19 A s"))

    """  # noqa: E501
    _reject_extra_args(flag, args)
    # Hartree energy E_h = 2 * Rydberg energy (NOT one Rydberg).
    e_hartree = 2 * const_Ryd * const_h * const_c
    length = const_a0.decompose()
    mass = const_me.decompose()
    time = (const_hbar / e_hartree).decompose()
    charge = const_e.si.decompose()
    return unitsystem(
        length,
        mass,
        time,
        charge,
    )


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
