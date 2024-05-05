"""Unitsystem compatibility."""
# TODO: move to a compatibility module

__all__: list[str] = []

try:  # TODO: less hacky way of supporting optional dependencies
    import pytest
except ImportError:  # pragma: no cover
    pass
else:
    _ = pytest.importorskip("gala")

from gala.units import (
    DimensionlessUnitSystem as GalaDimensionlessUnitSystem,
    UnitSystem as GalaUnitSystem,
)
from plum import dispatch

from ._core import DimensionlessUnitSystem, UnitSystem
from ._realizations import dimensionless


@dispatch
def unitsystem(value: GalaUnitSystem, /) -> UnitSystem:
    usys = UnitSystem(*value._core_units)  # noqa: SLF001
    usys._registry = value._registry  # noqa: SLF001
    return usys


@dispatch  # type: ignore[no-redef]
def unitsystem(_: GalaDimensionlessUnitSystem, /) -> DimensionlessUnitSystem:
    return dimensionless
