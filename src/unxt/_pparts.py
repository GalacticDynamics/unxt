r"""String formatting (private).

Not public API yet: the engine is settling, and the leading underscore keeps it
out of the documented surface until the shape has been exercised by a
downstream package.

**The name is the plan.** This module is spelled to become the standalone
package ``pparts``, so extraction is dropping the underscore: what is
``from unxt._pparts import pparts, PPart`` today becomes
``from pparts import pparts, PPart``, with no other edit at the call site. The
names re-exported here are exactly that package's intended surface -- see
`unxt._src.fmt.engine`, which is the code that moves.

The engine behind ``repr``, ``str``, ``format``, and the IPython
representations. It exists so those all agree on *what an object is made of*,
and so a new type can join in by registering one function.

This re-exports only what a downstream package needs -- a smaller set than the
engine defines, since widening later is easy and narrowing is not. Anything
unproven stays behind `unxt._src.fmt`. The contract, in three groups:

- **join in**: ``pparts`` (the extension point), ``PPart`` / ``PGroup``.
- **extend the grammar**: ``Axis``, ``register_axis``, ``register_alias``.
- **render**: ``pspec`` for ``__format__``; ``render`` + ``Spec`` for
  ``repr``/``str``.

The guide has the grammar, the axis table, and worked extension examples:
:doc:`/guides/formatting`.

Examples
--------
>>> import unxt as u
>>> from unxt._pparts import pspec
>>> q = u.Q([1.0, 2, 3], "m")

>>> pspec(q, "mul")
'[1., 2., 3.] * m'

>>> pspec(q, "latex")
'$[1.,~2.,~3.] \\, \\mathrm{m}$'

`unxt.quantity.AbstractQuantity.__format__` routes through `pspec`, so these
are also reachable as ``f"{q:mul}"`` and ``f"{q:latex}"``.

Register a type by registering `pparts`; it then gets every axis and every
markup, including ones it has never heard of -- note that ``**kw`` forwards
the axes this type does not itself act on:

>>> import dataclasses
>>> from unxt._pparts import PGroup, PPart, pparts

>>> @dataclasses.dataclass
... class Interval:
...     lo: u.Q
...     hi: u.Q

>>> @pparts.dispatch
... def _(obj: Interval, /, *, markup="text", **kw):
...     return (
...         PPart("open", "[", "sep"),
...         PGroup("lo", pparts(obj.lo, markup=markup, **kw)),
...         PPart("comma", ", ", "sep"),
...         PGroup("hi", pparts(obj.hi, markup=markup, **kw)),
...         PPart("close", ")", "sep"),
...     )

>>> iv = Interval(u.Q(0.0, "m"), u.Q(1.0, "m"))
>>> pspec(iv, "mul")
'[0. * m, 1. * m)'

>>> pspec(iv, "latex")
'$[0. \\, \\mathrm{m}, 1. \\, \\mathrm{m})$'

The unit axis reaches the nested quantities without `Interval` knowing it
exists:

>>> pspec(iv, "name")
'[0. meter, 1. meter)'

"""

# This module re-exports ``unxt._src.fmt``'s names verbatim, so the two
# ``__all__`` tuples trip duplicate-code; that overlap is the point of a shim.
# pylint: disable=duplicate-code

__all__ = (
    "Axis",
    "PGroup",
    "PPart",
    "Spec",
    "pparts",
    "pspec",
    "register_alias",
    "register_axis",
    "render",
)

from .setup_package import install_import_hook

with install_import_hook("unxt._pparts"):
    from ._src.fmt import (
        Axis,
        PGroup,
        PPart,
        Spec,
        pparts,
        pspec,
        register_alias,
        register_axis,
        render,
    )

# Clean up the namespace
del install_import_hook
