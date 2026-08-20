# 📜 Conventions

## Naming Conventions

`unxt` uses a few conventions to make the code more readable and to avoid verbosity. Many of these are also found in the [Glossary](glossary.md).

- `Abstract...`: a class that is not meant to be instantiated directly, but rather to be subclassed. Abstract classes are prefixed with 'Abstract'. Concrete (or 'final') classes are not so prefixed. As a further rule, no abstract class inherits from a concrete class and no concrete class inherits from any other concrete class.
- `USys`: a shorthand for "unit system", used in class names for concision.
- `Sim`: a shorthand for "simulation", used in class names for concision.

## Functional vs Object-Oriented APIs

As `JAX` is function-oriented, but Python is generally object-oriented, `unxt` provides both functional and object-oriented APIs. The functional APIs are the primary APIs, but the object-oriented APIs are easy to use and call the functional APIs, so lose none of the power.

As an example, consider the following code snippets:

```{code-block} python

>>> import unxt as u

>>> q = u.Q(1, 'm')
>>> q
Quantity(Array(1, dtype=int32...), unit='m')
```

First we'll show the object-oriented API:

```{code-block} python

>>> q.uconvert('cm')
Quantity(Array(100., dtype=float32, weak_type=True), unit='cm')
```

And now the function-oriented API:

```{code-block} python

>>> u.uconvert("cm", q)
Quantity(Array(100., dtype=float32, weak_type=True), unit='cm')
```

## Argument Order of Functional APIs

The functional APIs in `unxt` are inspired by the `Unitful.jl` library. The way to remember the order of arguments is to think of the function as constructing an operator that is then applied to the quantity.

For example, to convert a quantity `q` to centimeters, we use the `uconvert` function with the unit as the first argument and the quantity as the second:

```{code-block} python

>>> u.uconvert("cm", q)  # convert[to_unit](quantity)
Quantity(Array(100., dtype=float32, weak_type=True), unit='cm')
```

One of the reasons for this order is because it works very well with a multiple-dispatch system, where many variants of the same function can be defined based on the types of the arguments. The arguments for "operator" part of the function are the first arguments, and the arguments for the "operand" are the last arguments.

## Multiple Dispatch

`unxt` uses [multiple dispatch](https://beartype.github.io/plum/) to hook into `quax`'s flexible and extensible system to enable custom array-ish objects, like `Quantity`, in `JAX`. Also, `unxt` uses multiple dispatch to enable deep interoperability between `unxt` and other libraries, like `astropy`, `gala` (, and anything user-defined).

For example, `unxt` provides a `Quantity.from_` method that can convert an `astropy.Quantity` to a `unxt.Quantity`:

```{code-block} python

>>> import astropy.units as apyu
>>> import unxt as u

>>> aq = apyu.Quantity(1, 'm')  # Astropy Quantity
>>> aq
<Quantity 1. m>

>>> xq = u.Q.from_(aq)  # unxt Quantity
>>> xq
Quantity(Array(1., dtype=float32), unit='m')

```

This easy interoperability is enabled by multiple dispatch, which allows the `Quantity.from_` method to dispatch to the correct implementation based on the types of the arguments.

For more information on multiple dispatch, see the [plum documentation](https://beartype.github.io/plum/).

## Format Specs

`repr`, `str`, and `__format__` are one rendering, reached three ways. They differ only in the `unxt._fmt.Spec` they carry — a settled value for every axis — which `unxt._fmt.render` then executes. A format spec parses that `Spec` out of a string; `repr` and `str` read it from `unxt.config`. There is no second system and no separate preset table.

Because `__format__` is reached through an f-string (`f"{obj:spec}"`), its vocabulary is something a user carries from one `unxt` type to the next, so every type routing through `unxt._fmt.pspec` shares it and a spec means the same thing everywhere.

### The grammar

A spec is a `-`-joined run of **keywords**, optionally followed by a **Python format spec** applied per element:

```
spec := keyword ("-" keyword)* ["-" <python format spec>]
```

The parse is total and strictly left-to-right: consume tokens while they are keywords, and **the first token that is not one ends keyword parsing — everything from there, including any further `-`, is the value spec.**

That single rule is what keeps the grammar unambiguous once an arbitrary format spec is in play. A format spec may contain `-` itself, as a sign flag (`-.2f`) or a fill character (`->10.2f`), and neither can be mistaken for a component boundary, because keywords are only recognised _before_ the value spec begins.

Its one consequence worth remembering: **the value spec goes last.** `mul-name-.2f` works; `mul-.2f-name` swallows `name` into the value spec, which then fails as the malformed spec it is (and the error names the vocabulary it missed).

### The axes

Each keyword sets exactly one axis. Keywords are pairwise disjoint — no word names two axes — which is what makes the run order-independent (`html-bare` and `bare-html` are the same request) with no guessing anywhere. A test enforces that the vocabularies stay disjoint.

| axis | keywords | default | notes |
| --- | --- | --- | --- |
| layout | `call`, `product` | `product` | `call` is `Quantity(…, unit='m')`; `product` is `1. m` |
| value | `array`, `values`, `type` | `values` | `Array([1.], dtype=float32)` / `[1.]` / `f32[1]` |
| markup | `text`, `html`, `latex` | `text` | product layout only |
| unit | `symbol`, `name`, `dim` | `symbol` | `m` / `meter` / `length` |
| separator | `mul`, `bare` | `bare` | product layout only |
| abbreviation | `abbrev` | off | call layout only |

Setting one axis twice (`mul-bare`, `html-latex`) is an error, as is naming an axis the chosen layout has no concept of (`call-mul`). Both are reported as exactly that. Silently ignoring either would hide the mistake.

A value spec requires `values` (the default) — a `type` summary has no elements to format — and product layout.

```{code-block} python

>>> import unxt as u
>>> q = u.Q([1.0, 2, 3], "m")

>>> f"{q:mul}"
'[1., 2., 3.] * m'
>>> f"{q:type}"
'f32[3] m'
>>> f"{q:html-bare}"
'<span>[1., 2., 3.]</span> <span>m</span>'
>>> f"{q:name}"
'[1., 2., 3.] meter'
>>> f"{q:call}"
"Quantity([1., 2., 3.], unit='m')"

```

A value spec is applied to every element, and composes with the keywords:

```{code-block} python

>>> qq = u.Q([1.234, 2.345], "m")
>>> f"{qq:.2f}"
'[1.23, 2.35] m'
>>> f"{qq:mul-.2f}"
'[1.23, 2.35] * m'
>>> f"{qq:mul-name-.2f}"
'[1.23, 2.35] * meter'

```

The default separator is `bare`, so a bare value spec keeps astropy's shape (`3.14 m`, not `3.14 * m`).

### Aliases

Sugar, never new meaning: each expands _textually_ into core keywords before parsing, so an alias can never say something the grammar cannot, and combining one with a further keyword raises exactly the error its expansion would.

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

`abbrev` is one idea spelled per type: a short class name for a quantity, unquoted units for a unit system. `dims` stops being a unit-system special case — it is just the `dim` value of the shared unit axis.

### Why `call` layout routes through `__pdoc__`

`call` layout renders via `wadler_lindig.pformat`, and so through the object's own `__pdoc__`. That is load-bearing rather than incidental: `__pdoc__` is where a type states how to _reconstruct_ itself, which is what keeps `eval(repr(usys)) == usys` true for every unit-system realization. `repr` is defined as call layout for that reason, and a test pins the round-trip.

### Extending

- A new thing that varies orthogonally to the existing axes is a **new axis with its own keywords**, not a new alias — the point of the grammar is that components compose instead of every combination needing a hand-written name.
- A new alias must expand to a valid core spec, and a test checks that every one does.
- Keywords are lowercase, one word, no punctuation, and must not collide with an existing keyword _or_ alias — a colliding name would silently steal a meaning, which is how an earlier revision broke `f"{q:compact}"`.
- No keyword may itself be a legal Python format spec, or the scan rule would prefer the keyword reading and take a meaning users already had. A test checks this for every word.
- `""` is `str(obj)`, and `!r`/`!s` already cover `repr`/`str` — never add a `repr` or `str` keyword.

### For a type joining the engine

Register `pparts` and the whole grammar follows: every markup, every value form, the unit axis, and the per-element value spec. Accept `markup`, `short_arrays`, `value_spec` and `unit` as keyword arguments — plus `**kw`, so a future axis does not break the signature.

A type that skips `pparts` still renders: it degrades to `str(obj)` as one opaque fragment, because a display path must not raise just because some field's type never registered. The one thing it cannot degrade on is a value spec — that formats _elements_, and a type that never said what its elements are has none — so asking for one is an error rather than a silently different rendering.
