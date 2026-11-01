# How to make your own type formattable

A type joins the formatting grammar by saying what it is **made of**, not how it renders. Register `pparts` and the type gets every markup, every value form and the per-element format spec — including axes added later by someone else.

This guide is for library authors. To format a quantity, see {doc}`format-quantities`; for the vocabulary itself, see {doc}`../reference/formatting`.

```{code-block} python

>>> import dataclasses
>>> import unxt as u
>>> q = u.Q([1.0, 2, 3], "m")

```

## Register `pparts`, inherit `ReprMixin`

Register `pparts` to say what the type is made of, and inherit `ReprMixin` to get `repr`, `str`, `format`, the Jupyter representations and `__pdoc__` from one spec string.

Use `@dataclass(repr=False)` — the default `repr=True` would overwrite the mixin's `__repr__`.

```{code-block} python

>>> import dataclasses
>>> from unxt._pparts import PPart, ReprMixin, pparts

>>> @dataclasses.dataclass(repr=False)
... class Version(ReprMixin):
...     major: int
...     minor: int
...     __repr_spec__ = "bare"

>>> @pparts.dispatch
... def _(obj: Version, /, *, markup="text", **kw):
...     return (
...         PPart("major", str(obj.major)),
...         PPart("dot", ".", "sep"),
...         PPart("minor", str(obj.minor)),
...     )

>>> Version(1, 2)
1.2
>>> f"{Version(1, 2):html}"
'<span>1</span>.<span>2</span>'

```

`__repr_spec__` is parsed when the class is created, so a bad spec fails at definition rather than on first print. Reassigning it later has no effect.

If your type's `repr` is a constructor expression, write `__pdoc__` yourself instead — see {doc}`../explanation/why-formatting-is-one-engine` for why that one route is not built from fragments.

## Pass the axes down to nested values

Accept `**kw` in your `pparts` and forward it. The engine sends every axis your type does not itself handle, so nested values honour specs your type has never heard of:

```{code-block} python

>>> from unxt._pparts import PGroup, pspec

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

A type that registers no `pparts` still renders — it degrades to `str(obj)` — but asking it for a value format spec raises, since it has declared no elements to format.

## Add a keyword of your own

To add an axis, register it with the layouts it applies to and how its value becomes a renderer argument. To add shorthand for a combination of existing keywords, register an alias.

Unregister both when you are done, or they stay in the process-wide registry:

```{code-block} python

>>> from unxt._pparts import (
...     Axis, register_axis, register_alias,
...     unregister_axis, unregister_alias,
... )

>>> _ = register_axis(Axis(
...     name="vector_form",
...     keywords={"vecform": True},
...     default=False,
...     layouts={"call": lambda v: {"vector_form": v}},
... ))
>>> register_alias("terse", "call-abbrev-type")

>>> f"{q:terse}"
"Q(f32[3], unit='m')"

>>> unregister_alias("terse")
>>> unregister_axis("vector_form")

```

To add a markup beside `text`, `html` and `latex`, register a row with `_content`, `wrap`, `vsep` and `escape`:

```{code-block} python

>>> from unxt._pparts import register_markup, unregister_markup

>>> register_markup(
...     "brackets",
...     {"_content": "[{}]", "wrap": "{}", "vsep": ", ", "escape": None},
... )
>>> f"{q:brackets}"
'[[1., 2., 3.]] * [m]'
>>> unregister_markup("brackets")

```

Registration raises on any collision rather than overwriting, so a spec cannot change meaning without someone editing it.

## See also

- {doc}`../reference/formatting` — `Axis` fields, registration constraints, markup rows.
- {doc}`format-quantities` — the specs your type will accept once registered.
- {doc}`../explanation/why-formatting-is-one-engine` — the design, and what it traded away.
