"""Tools for representing systems of units using ``astropy.units``."""

__all__ = ["AbstractUnitSystem", "UNITSYSTEMS_REGISTRY"]

from collections.abc import Iterator
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar, get_args, get_type_hints

import astropy.units as u
from astropy.units import PhysicalType as Dimension
from astropy.units.physical import _physical_unit_mapping

from is_annotated import isannotated

from .utils import get_dimension_name
from unxt._src.typing_ext import Unit as UnitT
from unxt._src.units.core import units

Unit = u.UnitBase

_UNITSYSTEMS_REGISTRY: dict[tuple[Dimension, ...], type["AbstractUnitSystem"]] = {}
UNITSYSTEMS_REGISTRY = MappingProxyType(_UNITSYSTEMS_REGISTRY)


@dataclass(frozen=True, slots=True, eq=True)
class AbstractUnitSystem:
    """Represents a system of units.

    This class behaves like a dictionary with keys set by physical types (i.e.
    "length", "velocity", "energy", etc.). If a unit for a particular physical
    type is not specified on creation, a composite unit will be created with the
    base units. See the examples below for some demonstrations.

    Examples
    --------
    If only base units are specified, any physical type specified as a key to
    this object will be composed out of the base units::

    >>> from unxt import unitsystem
    >>> import astropy.units as u
    >>> usys = unitsystem(u.m, u.s, u.kg, u.radian)
    >>> usys
    unitsystem(m, s, kg, rad)

    >>> usys["velocity"]
    Unit("m / s")

    This unit system defines energy::

    >>> usys = unitsystem(u.m, u.s, u.kg, u.radian, u.erg)
    >>> usys["energy"]
    Unit("erg")

    This is useful for Galactic dynamics where lengths and times are usually
    given in terms of ``kpc`` and ``Myr``, but velocities are often specified in
    ``km/s``::

    >>> usys = unitsystem(u.kpc, u.Myr, u.Msun, u.radian, u.km/u.s)
    >>> usys["velocity"]
    Unit("km / s")

    Unit systems can be hashed:

    >>> isinstance(hash(usys), int)
    True

    And iterated over:

    >>> [x for x in usys]
    [Unit("kpc"), Unit("Myr"), Unit("solMass"), Unit("rad"), Unit("km / s")]

    With length equal to the number of base units

    >>> len(usys)
    5

    """

    # ===============================================================
    # Class-level

    _base_field_names: ClassVar[tuple[str, ...]]
    _base_dimensions: ClassVar[tuple[Dimension, ...]]

    def __init_subclass__(cls) -> None:
        # Register class with a tuple of it's dimensions.
        # This requires processing the type hints, not the dataclass fields
        # since those are made after the original class is defined.
        field_names, dimensions = parse_field_names_and_dimensions(cls)

        # Check the unitsystem is not already registered
        # If `make_dataclass(slots=True)` then the class is made twice, the
        # second time adding the `__slots__` attribute
        if dimensions in _UNITSYSTEMS_REGISTRY and "__slots__" not in cls.__dict__:
            msg = f"Unit system with dimensions {dimensions} already exists."
            raise ValueError(msg)

        # Add attributes to the class
        cls._base_field_names = tuple(field_names)
        cls._base_dimensions = dimensions

        _UNITSYSTEMS_REGISTRY[dimensions] = cls

    # ===============================================================
    # Instance-level

    @property  # TODO: classproperty
    def base_dimensions(self) -> tuple[Dimension, ...]:
        """Dimensions required for the unit system."""
        return self._base_dimensions

    @property
    def base_units(self) -> tuple[UnitT, ...]:
        """List of core units."""
        return tuple(getattr(self, k) for k in self._base_field_names)

    def __getitem__(self, key: Dimension | str) -> UnitT:
        """Get the unit for a given physical type.

        Examples
        --------
        >>> from unxt import unitsystem
        >>> import astropy.units as u
        >>> usys = unitsystem(u.m, u.s, u.kg, u.radian)

        Something in the base dimensions:

        >>> usys["length"]
        Unit("m")

        Something not in the base dimensions:

        >>> usys["velocity"]
        Unit("m / s")

        """
        key = u.get_physical_type(key)
        if key in self.base_dimensions:
            return getattr(self, get_dimension_name(key))

        unit = None
        for k, v in _physical_unit_mapping.items():
            if v == key:
                unit = units(" ".join([f"{x}**{y}" for x, y in k]))
                break
        # IDK if it's possible to get here
        else:
            msg = f"Physical type {key!r} doesn't exist in unit registry."  # pragma: no cover  # noqa: E501
            raise ValueError(msg)  # pragma: no cover

        unit = unit.decompose(self.base_units)
        unit._scale = 1.0  # noqa: SLF001
        return unit

    def __len__(self) -> int:
        """Return the number of base units in the system.

        Examples
        --------
        >>> from unxt import unitsystem
        >>> import astropy.units as u
        >>> usys = unitsystem(u.m, u.s, u.kg, u.radian)
        >>> len(usys)
        4

        """
        # Note: This is required for q.decompose(usys) to work, where q is a Quantity
        return len(self.base_dimensions)

    # TODO: should this be changed to _base_field_names -> Iterator[str]?
    def __iter__(self) -> Iterator[UnitT]:
        """Iterate over the base units.

        This is different than a dictionary, which would iterate over the
        physical types (the keys).

        Examples
        --------
        >>> from unxt import unitsystem
        >>> import astropy.units as u
        >>> usys = unitsystem(u.m, u.s, u.kg, u.radian)
        >>> list(iter(usys))
        [Unit("m"), Unit("s"), Unit("kg"), Unit("rad")]

        """
        yield from self.base_units

    def __str__(self) -> str:
        """Return a string representation of the unit system.

        Examples
        --------
        >>> from unxt import unitsystem
        >>> import astropy.units as u
        >>> usys = unitsystem(u.m, u.s, u.kg, u.radian)
        >>> str(usys)
        'LTMAUnitSystem(length, time, mass, angle)'

        """
        fs = ", ".join(map(str, self._base_field_names))
        return f"{type(self).__name__}({fs})"


# ---------------------------------------------------------------


def parse_field_names_and_dimensions(
    cls: type,
) -> tuple[tuple[str, ...], tuple[Dimension, ...]]:
    # Register class with a tuple of it's dimensions.
    # This requires processing the type hints, not the dataclass fields
    # since those are made after the original class is defined.
    type_hints = get_type_hints(cls, include_extras=True)

    field_names = []
    dimensions = []
    for name, type_hint in type_hints.items():
        # Check it's Annotated
        if not isannotated(type_hint):
            continue

        # Get the arguments to Annotated
        origin, *f_args = get_args(type_hint)

        # Check that the first argument is a UnitBase
        if not issubclass(origin, Unit):
            continue

        # Need for one of the arguments to be a PhysicalType
        f_dims = [x for x in f_args if isinstance(x, Dimension)]
        if not f_dims:
            msg = f"Field {name!r} must be an Annotated with a dimension."
            raise TypeError(msg)
        if len(f_dims) > 1:
            msg = (
                f"Field {name!r} must be an Annotated with only one dimension; "
                f"got {f_dims}"
            )
            raise TypeError(msg)

        field_names.append(get_dimension_name(name))
        dimensions.append(f_dims[0])

    if len(set(dimensions)) < len(dimensions):
        msg = "Some dimensions are repeated."
        raise ValueError(msg)

    return tuple(field_names), tuple(dimensions)
