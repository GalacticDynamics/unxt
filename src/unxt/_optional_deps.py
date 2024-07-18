"""Optional dependencies."""

__all__ = ["HAS_ASTROPY", "HAS_GALA"]

from importlib.util import find_spec

HAS_ASTROPY: bool = find_spec("astropy") is not None
HAS_GALA: bool = find_spec("gala") is not None
