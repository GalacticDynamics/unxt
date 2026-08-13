r"""String formatting (private).

Not public API yet: the engine is settling, and `unxt._fmt` keeps it out of the
documented surface until the shape has been exercised by a downstream package.

The engine behind ``repr``, ``str``, ``format``, and the IPython
representations. It exists so those all agree on *what an object is made of*,
and so a new type can join in by registering one function.

The main features are:

- ``unxt._fmt.pparts``: decompose an object into roled fragments. This is the
  extension point.
- ``unxt._fmt.parts_to_doc``: turn fragments into a `wadler_lindig` document, so
  plain-text rendering gets layout and composes inside a larger document.
- ``unxt._fmt.parts_to_markup``: turn fragments into an HTML or LaTeX string.
- ``unxt._fmt.pspec``: the shared ``__format__`` body, including
  ``unxt._fmt.FORMAT_PRESETS`` for f-string use like ``f"{q:compact}"``.

Examples
--------
>>> import unxt as u
>>> from unxt._fmt import pspec
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
>>> from unxt._fmt import PGroup, PPart, pparts, parts_to_markup

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

# This module re-exports ``unxt._src.fmt``'s names verbatim, so the two
# ``__all__`` tuples trip duplicate-code; that overlap is the point of a shim.
# pylint: disable=duplicate-code

__all__ = (
    "FORMAT_PRESETS",
    "MARKUPS",
    "PGroup",
    "PPart",
    "bad_spec",
    "custom_pdoc_no_kind",
    "custom_pdoc_noarray",
    "doc_to_str",
    "parts_to_doc",
    "unwrap_math",
    "parts_to_markup",
    "pparts",
    "pspec",
    "pspec_fallback",
    "value_str",
)

from .setup_package import install_import_hook

with install_import_hook("unxt._fmt"):
    from ._src.fmt import (
        FORMAT_PRESETS,
        MARKUPS,
        PGroup,
        PPart,
        bad_spec,
        custom_pdoc_no_kind,
        custom_pdoc_noarray,
        doc_to_str,
        parts_to_doc,
        parts_to_markup,
        pparts,
        pspec,
        pspec_fallback,
        unwrap_math,
        value_str,
    )

# Clean up the namespace
del install_import_hook
