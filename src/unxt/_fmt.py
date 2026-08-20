r"""String formatting (private).

Not public API yet: the engine is settling, and `unxt._fmt` keeps it out of the
documented surface until the shape has been exercised by a downstream package.

The engine behind ``repr``, ``str``, ``format``, and the IPython
representations. It exists so those all agree on *what an object is made of*,
and so a new type can join in by registering one function.

**This re-exports only what a downstream package needs**, which is a smaller
set than the engine defines. Widening it later is easy and narrowing it is not,
so anything unproven stays behind `unxt._src.fmt` until something outside this
repo actually needs it. The full surface is documented there; what follows is
the contract:

*Join in* -- teach the engine your type:

- ``pparts``: decompose an object into roled fragments. The extension point.
- ``PPart`` / ``PGroup``: the fragments to decompose into.

*Extend the grammar* -- teach it a new axis:

- ``Axis``, ``register_axis``, ``register_alias``.

*Render* -- implement your dunders:

- ``pspec``: the shared ``__format__`` body (parse a spec, then render).
- ``render`` + ``Spec``: for ``repr``/``str``, which are the same rendering
  reached with a `Spec` built by hand rather than parsed.

Examples
--------
>>> import unxt as u
>>> from unxt._fmt import pspec
>>> q = u.Q([1.0, 2, 3], "m")

>>> pspec(q, "mul")
'[1., 2., 3.] * m'

>>> pspec(q, "latex")
'$[1.,~2.,~3.] \\mathrm{m}$'

`unxt.quantity.AbstractQuantity.__format__` routes through `pspec`, so these
are also reachable as ``f"{q:mul}"`` and ``f"{q:latex}"``.

Register a type by registering `pparts`; it then gets every axis and every
markup, including ones it has never heard of -- note that ``**kw`` forwards
the axes this type does not itself act on:

>>> import dataclasses
>>> from unxt._fmt import PGroup, PPart, pparts

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
'$[0. \\mathrm{m}, 1. \\mathrm{m})$'

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

with install_import_hook("unxt._fmt"):
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
