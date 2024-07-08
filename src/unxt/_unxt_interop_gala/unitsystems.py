"""Unitsystem compatibility."""
# TODO: move to a compatibility module

__all__: list[str] = []

from gala.units import (  # pylint: disable=import-error
    DimensionlessUnitSystem as GalaDimensionlessUnitSystem,
    UnitSystem as GalaUnitSystem,
)
from plum import dispatch

from unxt.unitsystems import DimensionlessUnitSystem, UnitSystem, dimensionless


@dispatch
def unitsystem(value: GalaUnitSystem, /) -> UnitSystem:
    usys = UnitSystem(*value._core_units)  # noqa: SLF001
    usys._registry = value._registry  # noqa: SLF001
    return usys


@dispatch  # type: ignore[no-redef]
def unitsystem(_: GalaDimensionlessUnitSystem, /) -> DimensionlessUnitSystem:
    return dimensionless
