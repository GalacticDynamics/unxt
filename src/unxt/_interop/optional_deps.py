"""Optional dependencies. Internal use only."""

__all__ = ("OptDeps", "is_installed")

import importlib.util

from optional_dependencies import OptionalDependencyEnum, auto


class OptDeps(OptionalDependencyEnum):  # type: ignore[misc]  # pylint: disable=invalid-enum-extension
    """Optional external backends for ``unxt``.

    Only genuine third-party backends belong here: :class:`OptDeps` keys each
    member on its installed version, so members that share a version silently
    become enum aliases. unxt's own ``unxts.interop.*`` sub-packages always
    share unxt's release version, so they are detected with :func:`is_installed`
    instead of being enum members.
    """

    ASTROPY = auto()
    GALA = auto()


def is_installed(module: str) -> bool:
    """Return whether ``module`` can be imported.

    Used to detect unxt's optional interop sub-packages (``unxts.interop.*``),
    which are namespace packages that share unxt's version and therefore cannot
    be told apart by :class:`OptDeps` (whose members alias when they share a
    version).
    """
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False
