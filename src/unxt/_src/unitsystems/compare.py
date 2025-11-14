"""Comparison functions."""

__all__ = ("equivalent",)

from plum import dispatch

from .base import AbstractUnitSystem


@dispatch
def equivalent(a: AbstractUnitSystem, b: AbstractUnitSystem, /) -> bool:
    """Check if two unit systems are equivalent."""
    return set(a.base_dimensions) == set(b.base_dimensions)
