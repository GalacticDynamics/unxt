"""`unxt` interoperability with other libraries."""
# ruff:noqa: F401

__all__: tuple[str, ...] = ()

from .optional_deps import OptDeps

# Register interoperability
if OptDeps.ASTROPY.installed:
    from . import unxt_interop_astropy

if OptDeps.GALA.installed:
    from . import unxt_interop_gala

if OptDeps.MATPLOTLIB.installed:
    from . import unxt_interop_mpl as interop_mpl

    interop_mpl.setup_matplotlib_support_for_unxt(enable=True)
