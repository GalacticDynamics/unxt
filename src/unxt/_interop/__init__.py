"""`unxt` interoperability with other libraries."""
# ruff:noqa: F401

__all__: tuple[str, ...] = ()

from .optional_deps import OptDeps

# Register interoperability
if OptDeps.ASTROPY.installed:
    from . import unxt_interop_astropy

if OptDeps.UNXTS_INTEROP_GALA.installed:
    import unxts.interop.gala  # registers gala <-> unxt unitsystem conversions

if OptDeps.UNXTS_INTEROP_MATPLOTLIB.installed:
    import unxts.interop.matplotlib  # registers the matplotlib unit converter

if OptDeps.UNXTS_INTEROP_XARRAY.installed:
    import unxts.interop.xarray  # registers the `.unxt` xarray accessor
