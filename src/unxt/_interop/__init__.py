"""`unxt` interoperability with other libraries."""
# ruff:noqa: F401

__all__: tuple[str, ...] = ()

from .optional_deps import OptDeps, is_installed

# Register interoperability
if OptDeps.ASTROPY.installed:
    from . import unxt_interop_astropy

# Require both the interop package and the gala backend to be importable.
# `unxts.interop.gala` is an optional extra, and even when it is installed its
# `gala` dependency is skipped where gala cannot build (e.g. Windows), so the
# interop package can be present without an importable gala.
if is_installed("unxts.interop.gala") and OptDeps.GALA.installed:
    import unxts.interop.gala  # registers gala <-> unxt unitsystem conversions

if is_installed("unxts.interop.matplotlib"):
    import unxts.interop.matplotlib  # registers the matplotlib unit converter

if is_installed("unxts.interop.xarray"):
    import unxts.interop.xarray  # registers the `.unxt` xarray accessor
