"""Flags for unit system definitions."""

__all__ = ("AbstractUSysFlag", "DynamicalSimUSysFlag", "StandardUSysFlag")

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
