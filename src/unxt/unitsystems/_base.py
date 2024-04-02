"""Tools for representing systems of units using ``astropy.units``."""

__all__ = ["AbstractUnitSystem"]

from collections.abc import Iterator
from typing import ClassVar, Union, cast

import astropy.units as u
from astropy.units.physical import _physical_unit_mapping

from unxt._quantity.base import AbstractQuantity
from unxt._quantity.core import Quantity
from unxt._typing import Unit


class AbstractUnitSystem:
    """Represents a system of units.

    This class behaves like a dictionary with keys set by physical types (i.e. "length",
    "velocity", "energy", etc.). If a unit for a particular physical type is not
    specified on creation, a composite unit will be created with the base units. See the
    examples below for some demonstrations.

    See Also
    --------
    :class:`unxt.unitsystems.UnitSystem`

    """

    _core_units: list[Unit]
    _registry: dict[u.PhysicalType, Unit]

    _required_dimensions: ClassVar[list[u.PhysicalType]]  # do in subclass

    def __init__(
        self,
        units: Union[Unit, u.Quantity, AbstractQuantity, "AbstractUnitSystem"],
        *args: Unit | u.Quantity | AbstractQuantity,
    ) -> None:
        if isinstance(units, AbstractUnitSystem):
            if len(args) > 0:
                msg = (
                    "If passing in a AbstractUnitSystem, "
                    "cannot pass in additional units."
                )
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
        return f"{type(self).__name__}({', '.join(map(str, self._core_units))})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AbstractUnitSystem):
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
        # Note that it's necessary to
        return cast(AbstractQuantity, Quantity.constructor(quantity.to(unit), unit))
