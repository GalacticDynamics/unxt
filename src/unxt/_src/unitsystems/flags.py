"""Flags for unit system definitions."""

__all__ = (
    "AbstractUSysFlag",
    "AtomicUSysFlag",
    "DynamicalSimUSysFlag",
    "GeometrizedUSysFlag",
    "HEPUSysFlag",
    "NaturalUSysFlag",
    "PlanckUSysFlag",
    "StandardUSysFlag",
)

from typing import Any


class AbstractUSysFlag:
    """Abstract class for unit system flags to provide dispatch control.

    Unit system flags are used to indicate the type of unit system being
    defined. They are not intended to be instantiated and are used to provide
    dispatch control for defining unit systems.

    Raises
    ------
    ValueError
        If an attempt is made to instantiate a unit system flag class.

    See Also
    --------
    `unxt.unitsystems.unitsystem` :
        Function to define a unit system. The `AbstractUSysFlag` can be provided
        as the first argument to indicate the type of unit system being defined.
        This is useful for multiple-dispatching to the correct constructor.

    Examples
    --------
    For this example we will use the `unxt.unitsystems.StandardUSysFlag`, which
    indicates the standard construction of a unit system directly from the set
    of base units. Flags like `unxt.unitsystems.DynamicalSimUSysFlag` can be
    used to create a unit system where the gravitational constant ``G`` is
    unit-less and units are defined accordingly.

    >>> import unxt

    Define a unit system with the standard flag:

    >>> unxt.unitsystem(unxt.unitsystems.StandardUSysFlag, "m", "kg", "s")
    LengthMassTimeUnitSystem(length=Unit("m"), mass=Unit("kg"), time=Unit("s"))

    """

    def __new__(cls, *_: Any, **__: Any) -> None:  # type: ignore[misc]
        msg = "unit system flag classes cannot be instantiated."
        raise ValueError(msg)


class StandardUSysFlag(AbstractUSysFlag):
    """Flag to indicate a standard unit system with no additional arguments.

    Examples
    --------
    >>> from unxt.unitsystems import unitsystem, StandardUSysFlag

    Define a unit system with the standard flag:

    >>> unitsystem(StandardUSysFlag, "m", "kg", "s")
    LengthMassTimeUnitSystem(length=Unit("m"), mass=Unit("kg"), time=Unit("s"))

    Further examples may be found in the ``unitsystem`` docs.

    """


class DynamicalSimUSysFlag(AbstractUSysFlag):
    """Flag to indicate a unit system with optional definition of G.

    Examples
    --------
    >>> from unxt.unitsystems import unitsystem, DynamicalSimUSysFlag

    Define a unit system with the dynamical simulation flag:

    >>> unitsystem(DynamicalSimUSysFlag, "m", "kg")
    LengthMassTimeUnitSystem(length=Unit("m"), mass=Unit("kg"), time=Unit("122404 s"))

    Further examples may be found in the ``unitsystem`` docs.

    """


class NaturalUSysFlag(AbstractUSysFlag):
    """Base flag for `natural unit systems <https://en.wikipedia.org/wiki/Natural_units>`_.

    Natural unit systems set a chosen set of fundamental physical constants to
    the dimensionless value 1. In `unxt` this is realized numerically: the base
    units are chosen so that the named constants evaluate to ``1.0`` in the
    resulting system, while the full dimensional structure is preserved (mass,
    length, time, ... remain distinct dimensions). This mirrors
    `DynamicalSimUSysFlag`, which sets Newton's ``G`` to 1.

    This class is abstract; use one of its subclasses:

    - `HEPUSysFlag` : high-energy-physics units (``hbar = c = 1``)
    - `GeometrizedUSysFlag` : geometrized units (``c = G = 1``)
    - `PlanckUSysFlag` : Planck units (``hbar = c = G = k_B = 1``)
    - `AtomicUSysFlag` : atomic (Hartree) units (``m_e = hbar = e = 4*pi*eps0 = 1``)

    """


class HEPUSysFlag(NaturalUSysFlag):
    """Flag for high-energy-physics (particle) natural units: ``hbar = c = 1``.

    One free scale remains -- an energy, defaulting to 1 GeV. The base units are
    ``mass = E / c**2``, ``length = hbar * c / E``, ``time = hbar / E``.

    Examples
    --------
    >>> from unxt.unitsystems import unitsystem, HEPUSysFlag

    >>> usys = unitsystem(HEPUSysFlag)
    >>> usys
    LengthMassTimeUnitSystem(length=Unit("...e-16 m"), mass=Unit("...e-27 kg"), time=Unit("...e-25 s"))

    >>> [str(d) for d in usys.base_dimensions]
    ['length', 'mass', 'time']

    A different energy scale can be chosen with the ``energy`` keyword (a larger
    energy gives a smaller length/time and a larger mass, since ``hbar = c = 1``):

    >>> usys_tev = unitsystem(HEPUSysFlag, energy="TeV")
    >>> usys_tev["time"] == usys["time"] / 1000
    True

    """  # noqa: E501


class GeometrizedUSysFlag(NaturalUSysFlag):
    """Flag for geometrized (general-relativity) natural units: ``c = G = 1``.

    One free scale remains -- a length, defaulting to 1 meter. The base units are
    ``time = length / c``, ``mass = c**2 * length / G``.

    Examples
    --------
    >>> from unxt.unitsystems import unitsystem, GeometrizedUSysFlag

    >>> usys = unitsystem(GeometrizedUSysFlag)
    >>> usys
    LengthMassTimeUnitSystem(length=Unit("m"), mass=Unit("...e+27 kg"), time=Unit("...e-09 s"))

    >>> [str(d) for d in usys.base_dimensions]
    ['length', 'mass', 'time']

    A different length scale can be chosen with the ``length`` keyword:

    >>> unitsystem(GeometrizedUSysFlag, length="km")["length"]
    Unit("km")

    """  # noqa: E501


class PlanckUSysFlag(NaturalUSysFlag):
    """Flag for Planck natural units: ``hbar = c = G = k_B = 1``.

    Fully determined -- there are no free scales. The base units are the Planck
    length, mass, time, and temperature.

    Examples
    --------
    >>> from unxt.unitsystems import unitsystem, PlanckUSysFlag

    >>> usys = unitsystem(PlanckUSysFlag)
    >>> usys
    LengthMassTimeTemperatureUnitSystem(length=Unit("...e-35 m"), mass=Unit("...e-08 kg"), time=Unit("...e-44 s"), temperature=Unit("...e+32 K"))

    >>> [str(d) for d in usys.base_dimensions]
    ['length', 'mass', 'time', 'temperature']

    """  # noqa: E501


class AtomicUSysFlag(NaturalUSysFlag):
    """Flag for atomic (Hartree) natural units: ``m_e = hbar = e = 4*pi*eps0 = 1``.

    Fully determined -- there are no free scales. The base units are the Bohr
    radius (length), electron mass, the atomic unit of time (``hbar / E_h`` with
    ``E_h`` the Hartree energy), and the elementary charge.

    Examples
    --------
    >>> from unxt.unitsystems import unitsystem, AtomicUSysFlag

    >>> usys = unitsystem(AtomicUSysFlag)
    >>> usys
    LengthMassTimeElectricalChargeUnitSystem(length=Unit("...e-11 m"), mass=Unit("...e-31 kg"), time=Unit("...e-17 s"), electrical_charge=Unit("...e-19 A s"))

    >>> [str(d) for d in usys.base_dimensions]
    ['length', 'mass', 'time', 'electrical charge']

    """  # noqa: E501
