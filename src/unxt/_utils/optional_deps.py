"""Optional dependencies."""

__all__ = ["HAS_GALA"]

from importlib.util import find_spec

HAS_GALA: bool = find_spec("gala") is not None
