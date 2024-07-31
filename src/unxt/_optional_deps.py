"""Optional dependencies. Internal use only."""

__all__ = ["HAS_ASTROPY", "HAS_GALA", "HAS_MATPLOTLIB"]

from importlib.util import find_spec

HAS_ASTROPY: bool = find_spec("astropy") is not None
HAS_GALA: bool = find_spec("gala") is not None
HAS_MATPLOTLIB: bool = (find_spec("matplotlib") is not None) and (
    find_spec("zeroth") is not None
)
