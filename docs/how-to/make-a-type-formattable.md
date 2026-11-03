# How to make your own type formattable

A type joins the formatting grammar by saying what it is **made of**, not how it renders. Register `pparts` and the type gets every markup, every value form and the per-element format spec — including axes added later by someone else.

The engine is its own distribution, [`wadler-lindig-parts`][wlp], and the guide to it lives there: [make a type formattable][guide] covers `pparts`, `ReprMixin`, and registering axes, aliases and markups of your own. Import from `wadler_lindig_parts`; `unxt._pparts` is a private re-export kept for the import path and adds nothing to it.

This page covers what is specific to `unxt`: what your type gets when it is made of quantities.

```{code-block} python

>>> import dataclasses
>>> import unxt as u
>>> q = u.Q([1.0, 2, 3], "m")

```

## Quantities inside your type honour `unxt`'s axes

Accept `**kw` in your `pparts` and forward it. The engine sends every axis your type does not itself handle, so the nested quantities answer specs your type has never heard of — `unit`, here, which is one of `unxt`'s own axes rather than the engine's:

```{code-block} python

>>> from wadler_lindig_parts import PGroup, PPart, pparts, pspec

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
>>> pspec(iv, "name")           # reaches the nested quantities
'[0. meter, 1. meter)'

```

`Interval` never mentions units, and would keep working if `unxt` added another axis tomorrow. That is the whole reason to report fragments instead of a string.

A type that registers no `pparts` still renders — it degrades to `str(obj)` — but asking it for a value format spec raises, since it has declared no elements to format.

## Registering `unxt`'s own axes is not special

`unxt` contributes the `unit` and `abbrev` axes and the `compact` / `full` / `dims` aliases, and it registers them through the same public functions you would use. There is no privileged set: an axis `coordinax`, `galax` or your own package adds is read out of a spec exactly like a built-in one. The constraints each registration enforces, and the `Axis` fields, are in {func}`wadler_lindig_parts.register_axis`.

## See also

- [make a type formattable][guide] — the engine's own guide: `pparts`, `ReprMixin`, and adding axes, aliases and markups.
- {doc}`../reference/formatting` — the specs your type will accept once registered, as they apply to `unxt`'s types.
- {doc}`format-quantities` — formatting a quantity.
- {doc}`../explanation/why-formatting-is-one-engine` — the design, and what it traded away.

[wlp]: https://galacticdynamics.github.io/wadler-lindig-parts/
[guide]: https://galacticdynamics.github.io/wadler-lindig-parts/how-to/make-a-type-formattable/
