"""Optional dependencies. Internal use only."""

__all__ = ("OptDeps",)

from optional_dependencies import OptionalDependencyEnum, auto
from optional_dependencies.utils import chain_checks, get_version, is_installed


class OptDeps(OptionalDependencyEnum):  # type: ignore[misc]  # pylint: disable=invalid-enum-extension
    """Optional backends and interop sub-packages for ``unxt``.

    ``optional-dependencies`` >= 0.4.1 keys members on an identity-based
    wrapper, so members resolving to the same version (or both uninstalled) no
    longer collapse into an alias. unxt's own ``unxts.*`` sub-packages are
    released together and so usually share a version; before that fix they had
    to be detected separately.
    """

    ASTROPY = auto()
    UNXTS_INTEROP_MATPLOTLIB = auto()
    UNXTS_INTEROP_XARRAY = auto()

    #: The gala interop extra is only usable with an importable gala backend:
    #: the extra can be installed while gala itself is not, since gala is
    #: skipped where it cannot build (e.g. Windows).
    UNXTS_INTEROP_GALA = chain_checks(
        get_version("unxts.interop.gala"), is_installed("gala")
    )
