"""String formatting: `wadler_lindig_parts` plus `unxt`'s layer over it.

The engine lives in `wadler_lindig_parts`, which knows nothing about `unxt`.
What `unxt` adds to its grammar is registered where it is *used*: the unit
spelling axis in `unxt._src.units`, and the array rendering in
`unxt._src.quantity.value`. What is left here is the part that belongs to no
single module -- the ``abbrev`` axis, which two unrelated types spell
differently, and the whole-spec aliases, which span axes from three of them.

Modules use the engine as ``wadler_lindig_parts`` (``wlp``) directly; this
module exports nothing. A module whose rendering relies on these registrations
imports it for that side effect.
"""

__all__: tuple[str, ...] = ()

import wadler_lindig_parts as wlp

# ============================================================================
# The axes `unxt` puts into the grammar that belong to no one module

#: The abbreviated call form -- one idea spelled per type: a short class name
#: for a quantity (``use_short_name``), unquoted units for a unit system
#: (``quote_units``). It is registered here because it is registered for both:
#: in either module it would leave the other's rendering depending on an
#: unrelated import.
wlp.register_axis(
    wlp.Axis(
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
wlp.register_alias("compact", "call-abbrev")
wlp.register_alias("full", "call-array")
wlp.register_alias("dims", "call-dim")
