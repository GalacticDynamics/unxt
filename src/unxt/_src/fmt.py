"""String formatting: `wadler_lindig_parts` plus `unxt`'s layer over it.

The engine lives in `wadler_lindig_parts`, which knows nothing about `unxt`.
What `unxt` adds to its grammar is registered where it is *used*: the unit
spelling axis in `unxt._src.units`, and the array rendering in
`unxt._src.quantity.value`. What is left here is the part that belongs to no
single module -- the ``abbrev`` axis, which two unrelated types spell
differently, and the whole-spec aliases, which span axes from three of them.

Every module in `unxt` reaches the engine through here rather than importing
`wadler_lindig_parts` directly. Only the package's flat public surface is
re-exported: it has no private-but-documented tier, so nothing here reaches
for an underscore name across the distribution boundary.
"""

import wadler_lindig_parts as _wlp
from wadler_lindig_parts import *  # noqa: F403  # re-export: see the docstring
from wadler_lindig_parts import Axis, register_alias, register_axis

# The engine's own example types are its documentation, not `unxt`'s surface.
del examples

__all__ = [name for name in _wlp.__all__ if name != "examples"]


# ============================================================================
# The axes `unxt` puts into the grammar that belong to no one module

#: The abbreviated call form -- one idea spelled per type: a short class name
#: for a quantity (``use_short_name``), unquoted units for a unit system
#: (``quote_units``). It is registered here because it is registered for both:
#: in either module it would leave the other's rendering depending on an
#: unrelated import.
register_axis(
    Axis(
        name="abbrev",
        keywords={"abbrev": True},
        default=False,
        layouts={"call": lambda v: {"use_short_name": v, "quote_units": not v}},
    )
)

#: Whole-spec shorthands. Each spans axes owned by different modules, so like
#: ``abbrev`` they have nowhere else to live. An expansion is resolved when a
#: spec is parsed, not now, so naming ``dim`` -- a keyword of the ``unit`` axis
#: that `unxt._src.units` registers later -- is not an ordering constraint.
register_alias("compact", "call-abbrev")
register_alias("full", "call-array")
register_alias("dims", "call-dim")
