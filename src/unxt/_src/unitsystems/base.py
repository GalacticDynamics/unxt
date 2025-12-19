"""Tools for representing systems of units using ``astropy.units``."""

__all__ = ("UNITSYSTEMS_REGISTRY", "AbstractUnitSystem")

from collections.abc import Iterator
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar, get_args, get_type_hints

from astropy.units import PhysicalType, UnitBase as AstropyUnitBase
from astropy.units.physical import _physical_unit_mapping

from is_annotated import isannotated

from .utils import parse_dimlike_name
from unxt.dims import AbstractDimension, dimension
from unxt.units import AbstractUnit, unit

Unit = AstropyUnitBase

_UNITSYSTEMS_REGISTRY: dict[
    tuple[AbstractDimension, ...], type["AbstractUnitSystem"]
] = {}
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
    this object will be composed out of the base units:

    >>> from unxt import unitsystem
    >>> usys = unitsystem("m", "s", "kg", "radian")
    >>> usys
    unitsystem(m, s, kg, rad)

    >>> usys["velocity"]
    Unit("m / s")

    This unit system defines energy:

    >>> usys = unitsystem("m", "s", "kg", "radian", "erg")
    >>> usys["energy"]
    Unit("erg")

    This is useful for Galactic dynamics where lengths and times are usually
    given in terms of ``kpc`` and ``Myr``, but velocities are often specified in
    ``km/s``:

    >>> usys = unitsystem("kpc", "Myr", "Msun", "radian", "km / s")
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
    _base_dimensions: ClassVar[tuple[AbstractDimension, ...]]

    def __init_subclass__(cls) -> None:
        # Register class with a tuple of it's dimensions.
        # This requires processing the type hints, not the dataclass fields
        # since those are made after the original class is defined.
        field_names, dims = parse_field_names_and_dimensions(cls)

        # Check the unitsystem is not already registered
        # If `make_dataclass(slots=True)` then the class is made twice, the
        # second time adding the `__slots__` attribute
        if dims in _UNITSYSTEMS_REGISTRY and "__slots__" not in cls.__dict__:
            msg = f"Unit system with dimensions {dims} already exists."
            raise ValueError(msg)

        # Add attributes to the class
        cls._base_field_names = tuple(field_names)
        cls._base_dimensions = dims

        _UNITSYSTEMS_REGISTRY[dims] = cls

    # ===============================================================
    # USys API

    @property  # TODO: classproperty
    def base_dimensions(self) -> tuple[AbstractDimension, ...]:
        """Dimensions required for the unit system."""
        return self._base_dimensions

    @property
    def base_units(self) -> tuple[AbstractUnit, ...]:
        """List of core units."""
        return tuple(getattr(self, k) for k in self._base_field_names)

    # ===============================================================
    # Python stuff

    def __getitem__(self, key: AbstractDimension | str) -> AbstractUnit:
        """Get the unit for a given physical type.

        Examples
        --------
        >>> from unxt import unitsystem
        >>> usys = unitsystem("m", "s", "kg", "radian")

        Something in the base dimensions:

        >>> usys["length"]
        Unit("m")

        Something not in the base dimensions:

        >>> usys["velocity"]
        Unit("m / s")

        """
        key = dimension(key)
        if key in self.base_dimensions:
            return getattr(self, parse_dimlike_name(key))

        out: AbstractUnit
        for k, v in _physical_unit_mapping.items():
            if v == key:
                out = unit(" ".join([f"{x}**{y}" for x, y in k]))
                break
        # IDK if it's possible to get here
        else:
            msg = f"Physical type {key!r} doesn't exist in unit registry."  # pragma: no cover  # noqa: E501
            raise ValueError(msg)  # pragma: no cover

        out = out.decompose(self.base_units)
        out._scale = 1.0  # noqa: SLF001
        return out

    def __len__(self) -> int:
        """Return the number of base units in the system.

        Examples
        --------
        >>> from unxt import unitsystem
        >>> usys = unitsystem("m", "s", "kg", "radian")
        >>> len(usys)
        4

        """
        # Note: This is required for q.decompose(usys) to work, where q is a Quantity
        return len(self.base_dimensions)

    # TODO: should this be changed to _base_field_names -> Iterator[str]?
    def __iter__(self) -> Iterator[AbstractUnit]:
        """Iterate over the base units.

        This is different than a dictionary, which would iterate over the
        physical types (the keys).

        Examples
        --------
        >>> from unxt import unitsystem
        >>> usys = unitsystem("m", "s", "kg", "radian")
        >>> list(iter(usys))
        [Unit("m"), Unit("s"), Unit("kg"), Unit("rad")]

        """
        yield from self.base_units

    def __str__(self) -> str:
        """Return a string representation of the unit system.

        Examples
        --------
        >>> from unxt import unitsystem
        >>> usys = unitsystem("m", "s", "kg", "radian")
        >>> str(usys)
        'LTMAUnitSystem(length, time, mass, angle)'

        """
        fs = ", ".join(map(str, self._base_field_names))
        return f"{type(self).__name__}({fs})"

    # ===============================================================
    # Plum stuff

    #: This tells `plum` that this type can be efficiently cached.
    __faithful__: ClassVar[bool] = True


# ---------------------------------------------------------------


def parse_field_names_and_dimensions(
    cls: type,
) -> tuple[tuple[str, ...], tuple[AbstractDimension, ...]]:
    # Register class with a tuple of it's dimensions.
    # This requires processing the type hints, not the dataclass fields
    # since those are made after the original class is defined.
    type_hints = get_type_hints(cls, include_extras=True)

    field_names = []
    dims = []
    for name, type_hint in type_hints.items():
        # Check it's Annotated
        if not isannotated(type_hint):
            continue

        # Get the arguments to Annotated
        origin, *f_args = get_args(type_hint)

        # Check that the first argument is a UnitBase
        if not issubclass(origin, Unit):
            continue

        # Need for one of the arguments to be a Dimension
        f_dim = [x for x in f_args if isinstance(x, PhysicalType)]
        if not f_dim:
            msg = f"Field {name!r} must be an Annotated with a dimension."
            raise TypeError(msg)
        if len(f_dim) > 1:
            msg = (
                f"Field {name!r} must be an Annotated with only one dimension; "
                f"got {f_dim}"
            )
            raise TypeError(msg)

        field_names.append(parse_dimlike_name(name))
        dims.append(f_dim[0])

    if len(set(dims)) < len(dims):
        msg = "Some dimensions are repeated."
        raise ValueError(msg)

    return tuple(field_names), tuple(dims)
