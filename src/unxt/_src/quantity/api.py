"""Functional approach to Quantities."""

__all__ = ("uconvert", "ustrip", "is_unit_convertible", "wrap_to")

from typing import Any

from plum import dispatch

from unxt_api import is_unit_convertible, uconvert, ustrip, wrap_to


@dispatch
def wrap_to(x: Any, /, *, min: Any, max: Any) -> Any:
    """Wrap to the range [min, max)."""
    return wrap_to(x, min, max)  # redirect to the standard method
