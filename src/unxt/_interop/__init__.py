"""`unxt` interoperability with other libraries."""
# ruff:noqa: F401

__all__: tuple[str, ...] = ()

from .optional_deps import OptDeps

# Register interoperability.
#
# `no branch`: each gate is an import-time environment check, so within one
# process only the arm matching the installed extras can ever run -- the other
# is exercised by a CI job with a different env, in a separate coverage run.
# `OptDeps` itself is what decides these, and it is covered directly by
# `tests/unit/test_optional_deps.py`.
if OptDeps.ASTROPY.installed:  # pragma: no branch
    from . import unxt_interop_astropy

if OptDeps.UNXTS_INTEROP_GALA.installed:  # pragma: no branch  # implies gala
    import unxts.interop.gala  # registers gala <-> unxt unitsystem conversions

if OptDeps.UNXTS_INTEROP_MATPLOTLIB.installed:  # pragma: no branch
    import unxts.interop.matplotlib  # registers the matplotlib unit converter

if OptDeps.UNXTS_INTEROP_XARRAY.installed:  # pragma: no branch
    import unxts.interop.xarray  # registers the `.unxt` xarray accessor
