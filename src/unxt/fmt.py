r"""String formatting.

The engine behind ``repr``, ``str``, ``format``, and the IPython
representations. It exists so those all agree on *what an object is made of*,
and so a new type can join in by registering one function.

The main features are:

- ``unxt.fmt.pparts``: decompose an object into roled fragments. This is the
  extension point.
- ``unxt.fmt.parts_to_doc``: turn fragments into a `wadler_lindig` document, so
  plain-text rendering gets layout and composes inside a larger document.
- ``unxt.fmt.parts_to_markup``: turn fragments into an HTML or LaTeX string.
- ``unxt.fmt.pspec``: the shared ``__format__`` body, including
  ``unxt.fmt.FORMAT_PRESETS`` for f-string use like ``f"{q:compact}"``.

Examples
--------
>>> import unxt as u
>>> from unxt.fmt import pspec
>>> q = u.Q([1.0, 2, 3], "m")

>>> pspec(q, "mul")
'[1., 2., 3.] * m'

>>> pspec(q, "latex")
'$[1.,~2.,~3.] \\; \\mathrm{m}$'

`unxt.quantity.AbstractQuantity.__format__` routes through `pspec`, so these
are also reachable as ``f"{q:mul}"`` and ``f"{q:latex}"``.

Register a type by registering `pparts`; it then gets every preset and every
markup:

>>> import dataclasses
>>> from unxt.fmt import PGroup, PPart, pparts, parts_to_markup

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
>>> parts_to_markup(pparts(iv))
'[0. * m, 1. * m)'

>>> parts_to_markup(pparts(iv, markup="latex"), markup="latex")
'$[0. \\; \\mathrm{m}, 1. \\; \\mathrm{m})$'

"""

__all__ = (
    "FORMAT_PRESETS",
    "MARKUPS",
    "PGroup",
    "PPart",
    "parts_to_doc",
    "parts_to_markup",
    "pparts",
    "pspec",
    "pspec_fallback",
)

from .setup_package import install_import_hook

with install_import_hook("unxt.fmt"):
    from ._src.fmt import (
        FORMAT_PRESETS,
        MARKUPS,
        PGroup,
        PPart,
        parts_to_doc,
        parts_to_markup,
        pparts,
        pspec,
        pspec_fallback,
    )

# Clean up the namespace
del install_import_hook
