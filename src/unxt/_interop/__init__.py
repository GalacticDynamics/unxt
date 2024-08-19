"""`unxt` interoperability with other libraries."""
# ruff:noqa: F401

__all__: list[str] = []

from . import optional_deps

# Register interoperability
if optional_deps.HAS_ASTROPY:
    from . import unxt_interop_astropy

if optional_deps.HAS_GALA:
    from . import unxt_interop_gala

if optional_deps.HAS_MATPLOTLIB:
    from . import unxt_interop_mpl as interop_mpl

    interop_mpl.setup_matplotlib_support_for_unxt(enable=True)
