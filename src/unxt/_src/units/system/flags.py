__all__ = ["AbstractUnitSystemFlag", "StandardUnitSystemFlag"]

from typing import Any


class AbstractUnitSystemFlag:
    """Abstract class for unit system flags to provide dispatch control."""

    def __new__(cls, *_: Any, **__: Any) -> None:  # type: ignore[misc]
        msg = "unit system flag classes cannot be instantiated."
        raise ValueError(msg)


class StandardUnitSystemFlag(AbstractUnitSystemFlag):
    """Flag to indicate a standard unit system with no additional arguments."""


class SimulationUnitSystemFlag(AbstractUnitSystemFlag):
    """Flag to indicate a unit system with optional definition of G."""
