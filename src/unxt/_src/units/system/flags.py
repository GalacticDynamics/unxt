__all__ = [
    "AbstractUSysFlag",
    "StandardUSysFlag",
    "DynamicalSimUSysFlag",
]

from typing import Any


class AbstractUSysFlag:
    """Abstract class for unit system flags to provide dispatch control."""

    def __new__(cls, *_: Any, **__: Any) -> None:  # type: ignore[misc]
        msg = "unit system flag classes cannot be instantiated."
        raise ValueError(msg)


class StandardUSysFlag(AbstractUSysFlag):
    """Flag to indicate a standard unit system with no additional arguments."""


class DynamicalSimUSysFlag(AbstractUSysFlag):
    """Flag to indicate a unit system with optional definition of G."""
