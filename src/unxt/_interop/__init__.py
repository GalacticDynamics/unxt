"""`unxt` interoperability with other libraries."""
# ruff:noqa: F401

__all__: tuple[str, ...] = ()

from .optional_deps import OptDeps, is_installed

# Register interoperability
if OptDeps.ASTROPY.installed:
    from . import unxt_interop_astropy

# gala itself must also be importable: unxts.interop.gala is installed on every
# platform, but its gala dependency is skipped where gala cannot build (Windows).
if is_installed("unxts.interop.gala") and OptDeps.GALA.installed:
    import unxts.interop.gala  # registers gala <-> unxt unitsystem conversions

if is_installed("unxts.interop.matplotlib"):
    import unxts.interop.matplotlib  # registers the matplotlib unit converter

if is_installed("unxts.interop.xarray"):
    import unxts.interop.xarray  # registers the `.unxt` xarray accessor
