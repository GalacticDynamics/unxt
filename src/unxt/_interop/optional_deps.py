"""Optional dependencies. Internal use only."""

__all__ = ("OptDeps", "is_installed")

import importlib.util

from optional_dependencies import OptionalDependencyEnum, auto


class OptDeps(OptionalDependencyEnum):  # type: ignore[misc]  # pylint: disable=invalid-enum-extension
    """Optional external backends for ``unxt``.

    Only genuine third-party backends belong here. :class:`OptDeps` keys each
    member on its installed *version*, so any two members that share a version
    silently collapse into a single enum alias. Version is therefore an unsafe
    key for distinguishing packages: unxt's own ``unxts.*`` sub-packages are
    released together and so usually share a version (independent bug-fix
    releases can make them diverge), so their presence is detected by import
    with :func:`is_installed`, which resolves each module by name independent of
    version.
    """

    ASTROPY = auto()
    GALA = auto()


def is_installed(module: str) -> bool:
    """Return whether ``module`` can be imported.

    Used to detect unxt's optional interop sub-packages (``unxts.interop.*``),
    which cannot be reliably told apart by :class:`OptDeps` -- its members alias
    whenever they share a version, which the co-released ``unxts.*`` packages
    usually do (though independent bug-fix releases can make them diverge).
    """
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False
