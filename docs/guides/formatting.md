# 🖋️ String Formatting

`repr`, `str`, and `__format__` are one rendering, reached three ways. They differ only in the **spec** they carry — a settled value for every axis — which a single renderer then executes. A format spec parses that spec out of a string; `repr` and `str` read it from [`unxt.config`](configuration.md).

Because `__format__` is reached through an f-string (`f"{obj:spec}"`), its vocabulary is something you carry from one `unxt` type to the next: a spec means the same thing everywhere.

```{code-block} python

>>> import unxt as u
>>> q = u.Q([1.0, 2, 3], "m")

>>> f"{q:mul}"
'[1., 2., 3.] * m'
>>> f"{q:latex}"
'$[1.,~2.,~3.] \\, \\mathrm{m}$'
>>> f"{q:.2f}"
'[1.00, 2.00, 3.00] m'

```

## The grammar

A spec is a `-`-joined run of **keywords**, optionally ending in a **Python format spec** applied to each element:

```
spec := keyword ("-" keyword)* ["-" <python format spec>]
```

The parse is total and strictly left-to-right: consume tokens while they are keywords, and **the first token that is not one ends keyword parsing — everything from there, including any further `-`, is the format spec.**

That single rule is what keeps the grammar unambiguous once an arbitrary format spec is in play. A format spec may contain `-` itself, as a sign flag (`-.2f`) or a fill character (`->10.2f`), and neither can be mistaken for a component boundary, because keywords are only recognised _before_ it begins.

Its one consequence worth remembering: **the format spec goes last.** `mul-name-.2f` works; `mul-.2f-name` swallows `name` into the format spec, which then fails as the malformed spec it is — and the error names the vocabulary it missed.

## The axes

Each keyword sets exactly one axis. Keywords are pairwise disjoint — no word names two axes — which is what makes the run order-independent (`html-bare` and `bare-html` are the same request) with no guessing anywhere.

| axis | keywords | default | notes |
| --- | --- | --- | --- |
| layout | `call`, `product` | `product` | `call` is `Quantity(…, unit='m')`; `product` is `1. m` |
| value | `array`, `values`, `type` | `values` | `Array([1.], dtype=float32)` / `[1.]` / `f32[1]` |
| markup | `text`, `html`, `latex` | `text` | product layout only |
| unit | `symbol`, `name`, `dim` | `symbol` | `m` / `meter` / `length` |
| separator | `mul`, `bare` | `bare` | product layout only |
| abbreviation | `abbrev` | off | call layout only |

```{code-block} python

>>> f"{q:type}"
'f32[3] m'
>>> f"{q:html-bare}"
'<span>[1., 2., 3.]</span> <span>m</span>'
>>> f"{q:name}"
'[1., 2., 3.] meter'
>>> f"{q:call}"
"Quantity([1., 2., 3.], unit='m')"

```

Setting one axis twice (`mul-bare`, `html-latex`) is an error, as is naming an axis the chosen layout has no concept of (`call-mul`). Both are reported as exactly that — silently ignoring either would hide the mistake.

## The format spec is the value axis

A trailing format spec is not a separate thing bolted onto the grammar: it is a **value of the `value` axis**, which accepts either one of its keywords or arbitrary text. That is why it composes with every other keyword, and why asking for both a keyword and a format spec is simply setting one axis twice:

```{code-block} python

>>> qq = u.Q([1.234, 2.345], "m")
>>> f"{qq:.2f}"
'[1.23, 2.35] m'
>>> f"{qq:mul-.2f}"
'[1.23, 2.35] * m'
>>> f"{qq:mul-name-.2f}"
'[1.23, 2.35] * meter'

>>> try:
...     f"{qq:type-.2f}"     # a shape/dtype summary has no elements to format
... except ValueError as e:
...     print(str(e).split(". Expected")[0])
invalid format spec 'type-.2f' for Quantity: 'value' is set twice

```

The default separator is `bare`, so a bare format spec keeps astropy's shape (`3.14 m`, not `3.14 * m`).

## Aliases

Sugar, never new meaning: each expands _textually_ into keywords before parsing, so an alias can never say something the grammar cannot, and combining one with a further keyword raises exactly the error its expansion would.

| alias     | expands to    |
| --------- | ------------- |
| `compact` | `call-abbrev` |
| `full`    | `call-array`  |
| `dims`    | `call-dim`    |

```{code-block} python

>>> f"{q:compact}"
"Q([1., 2., 3.], unit='m')"

>>> usys = u.unitsystem("kpc", "Myr", "Msun", "radian")
>>> f"{usys:dims}"
'LTMAUnitSystem(length, time, mass, angle)'

```

`abbrev` is one idea spelled per type: a short class name for a quantity, unquoted units for a unit system. `dims` is not a unit-system special case — it is just the `dim` value of the shared unit axis.

## Why `call` layout goes through `__pdoc__`

`call` layout renders via `wadler_lindig.pformat`, and so through the object's own `__pdoc__`. That is load-bearing rather than incidental: `__pdoc__` is where a type states how to _reconstruct_ itself, which is what keeps `eval(repr(usys)) == usys` true for every unit-system realization. `repr` is defined as call layout for that reason.

## Explicit axis assignment

Keywords share one flat namespace — that is what lets them be given in any order, since a bare word must identify its axis without help from position. The cost is that two packages can want the same word for unrelated things: `dim` is a unit spelling here, and could as reasonably be a manifold's dimensionality in `coordinax`.

Any keyword may therefore be written as an explicit assignment, `axis=word`:

```{code-block} python

>>> f"{q:unit=dim}"
'[1., 2., 3.] length'
>>> f"{q:markup=latex-sep=mul}" == f"{q:latex-mul}"
True

```

`=` says what a keyword actually does — a keyword _sets an axis to a value_ — so `unit=dim` is the operation spelled out, and bare `dim` is its shorthand. It also mirrors the keyword argument the axis translates to.

A bare word resolves while exactly one axis claims it — which is every word today, so the explicit form never intrudes on ordinary use. When two axes claim one, the bare form becomes ambiguous and says so, naming both ways out:

```
f"{q:dim}"  ->  ambiguous keyword 'dim'; claimed by 'manifold', 'unit'.
                Write it as one of: manifold=dim, unit=dim
```

It is per-token, so a collision costs one assignment on one word and leaves the rest of a spec alone — `latex-mul-unit=dim-.3f`, not a fully explicit rewrite. A downstream package can also write explicit forms from the start and be immune to a word it does not yet share.

`=` is also an _align_ character in a format spec, but align only ever appears at position 0 or 1, so a spec like `=>6` names no axis and falls through to the format spec like any other non-keyword:

```{code-block} python

>>> f"{u.Q(3, 'm'):=>6}"     # '=' fill, '>' alignment
'=====3 m'
>>> f"{u.Q(3, 'm'):>6}"      # '>' alignment, space fill
'     3 m'

```

## Extending

The engine is a mechanism, not a fixed vocabulary. Axes and aliases are registered, so a downstream package's axis is indistinguishable from a built-in one.

```{code-block} python

>>> from unxt._pparts import Axis, register_axis, register_alias

>>> _ = register_axis(Axis(
...     name="vector_form",
...     keywords={"vecform": True},   # spec words -> the value each sets
...     default=False,
...     layouts={"call": lambda v: {"vector_form": v}},  # membership IS applicability
... ))

>>> register_alias("terse", "call-abbrev-type")
>>> f"{q:terse}"
"Q(f32[3], unit='m')"

```

`layouts` does double duty: it says which layouts the axis applies to _and_ how its value becomes keyword arguments for that layout's renderer. The two layouts often want different things from one choice — `unit` becomes `show_units=` for `call` but `unit_style=` for `product` — which is why the translation is per-layout.

An axis may also declare `free_text=(...)`, naming the layouts in which its value may be arbitrary text instead of a keyword. At most one axis may do so, since a spec has only one trailing run to give; in `unxt` that axis is `value`.

Rules:

- Registration rejects a collision in either direction — a keyword that is already an alias, an alias that is already a keyword, a re-registered axis. Silently overwriting is how a spec changes meaning with nobody editing it.
- No keyword may itself be a legal Python format spec, or the scan rule would prefer the keyword reading and take a meaning users already had.
- Build a spec by hand with `Spec.of(**overrides)`, which fills every registered axis from the registry. Constructing the mapping directly leaves a hole that surfaces as a `KeyError` once a later axis is added.
- `""` is `str(obj)`, and `!r`/`!s` already cover `repr`/`str` — never add a `repr` or `str` keyword.

## Joining in

Register `pparts` and the whole grammar follows: every markup, every value form, the unit axis, and the per-element format spec. Accept `markup`, `short_arrays`, `value_spec` and `unit_style` as keyword arguments — plus `**kw`, since the product renderer forwards every axis it does not own, including ones registered by someone else.

```{code-block} python

>>> import dataclasses
>>> from unxt._pparts import PGroup, PPart, pparts, pspec

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

```

The unit axis reaches the nested quantities without `Interval` knowing it exists — `**kw` forwards it:

```{code-block} python

>>> pspec(iv, "name")
'[0. meter, 1. meter)'

```

A type that skips `pparts` still renders: it degrades to `str(obj)` as one opaque fragment, because a display path must not raise just because some field's type never registered. The one thing it cannot degrade on is a format spec — that formats _elements_, and a type that never said what its elements are has none — so asking for one is an error rather than a silently different rendering.

## Checking that a spec did something

An axis is offered to every type and quietly ignored by one that has no concept of it. That is deliberate — it is what lets a composite forward axes it has never heard of, so `f"{interval:name}"` reaches the nested quantities — but it means a spec can be accepted and do nothing:

- `f"{q:name}"` on `km / s` is inert: a composite unit has no long name, so the axis falls back to the symbol.
- `f"{q:dims}"` on a `Quantity` is inert: the alias expands to `call-dim`, and only unit systems honour it.

Whether an axis mattered cannot be declared per type — a composite genuinely does not honour `unit`, yet forwarding makes it work — so it is asked by experiment instead: render again with the axis reset to its default and see whether the output moves. Set `WARN_INERT_AXES` on the engine to have every spec checked and warn when one changed nothing.

It is off by default because each probe is a second render, several times the cost of a plain format call. The flag lives on the engine, not in `unxt.config`, so it survives extraction — `unxt`, `coordinax` and `galax` each set it for themselves.

## Layering

The code is split along the seam it will eventually be cut at:

- **the engine** — domain-agnostic and self-contained: fragments, the markup table, the wadler-lindig feed, the `pparts` dispatcher, the scan-rule parser, `Spec`, the layouts, and the axis registry. It imports nothing from `unxt`, `jax`, `numpy` or `astropy`; a test asserts that from its import list, so the claim cannot rot into prose. It is destined to become a standalone package named **`pparts`**, after the extension point it turns on.
- **`unxt`'s layer** — the axes `unxt` puts into the grammar, the aliases, and the array helpers those axes need.

`coordinax` and `galax` add their own axes the same way, as **peers** of `unxt`'s layer rather than clients of it. There is no privileged set of axes, and a downstream axis is read from a spec exactly like a built-in one.
