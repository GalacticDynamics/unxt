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

## Format Spec Presets

`repr`, `str`, and the IPython representations are each free to look however a type wants. `__format__` is different: it is reached through an f-string (`f"{obj:preset}"`), so its spec strings are part of the vocabulary a user carries from one `unxt` type to the next. Every type that routes its `__format__` through `unxt._fmt.pspec` shares the same vocabulary — a spec means the same thing everywhere it is honoured, and a type has to do nothing to inherit one added later.

Two structures exist:

- **Call-style** — `ClassName(value, unit=...)`, rendered by `wadler_lindig.pformat`. Named by a flat entry in `unxt._fmt.FORMAT_PRESETS`: `full` (full array repr), `compact` (short class name, compact array), `dims` (dimension names instead of units — unit-system only; see below).
- **Product-style** — `value * unit`, rendered by decomposing the object with `pparts` and running it through `parts_to_doc` / `parts_to_markup`. Named by a small DSL rather than one preset per combination.

### The product-style DSL

A product-style spec is up to four optional, independent components, canonically spelled `<markup>-<array>-<separator>-<unit>`:

| component | values | omitted means |
| --- | --- | --- |
| markup | `html`, `latex` | `text` |
| array | `short` (shape/dtype summary, e.g. `f32[3]`), `values` (the default form, nameable so it can pair with a value spec) | the default form, e.g. `[1., 2., 3.]` |
| separator | `mul` (show `*`), `bare` (space, no operator) | `mul` |
| unit | `long` (the unit's spelled-out name, e.g. `meter`) | the short/symbol form, e.g. `m` |

```{code-block} python

>>> import unxt as u

>>> q = u.Q([1.0, 2, 3], "m")
>>> f"{q:mul}"
'[1., 2., 3.] * m'
>>> f"{q:short}"
'f32[3] * m'
>>> f"{q:html-bare}"
'<span>[1., 2., 3.]</span> <span>m</span>'
>>> f"{q:html-short-mul}"
'<span>f32[3]</span> * <span>m</span>'
>>> f"{u.Q(1.0, 'm'):long}"
'1. * meter'
```

Omitting a component takes its default, which is what makes the short forms just an unremarkable case of the same grammar rather than a separate alias table: `mul` is short for `text-mul` (separator is the only component present), `html` is short for `html-mul` (markup present, separator defaults to `mul`), and `short` is short for `text-mul-short`.

The parser accepts the tokens in any order — `html-bare` and `bare-html` render identically — but `<markup>-<array>-<separator>-<unit>` is the one spelling used in docs and error messages, so pick it when writing a spec by hand. Two tokens from the same component (`mul-bare`, `html-latex`) is invalid and raises `ValueError`, same as an unknown spec.

**The array component alone may also carry a trailing Python format spec**, applied to every element (`np.array2string`'s `formatter=`), e.g. `mul-.3g` or just `.3g-mul`. `values` is spelled out so it has something to attach to on its own: `values-.3g` (equivalently, bare `.3g` combined with any other DSL token — `mul-.3g` reads just as well and is the more natural choice, since `mul` already exists). `short` and a value spec don't compose — a shape/dtype summary has no per-element values to format, and combining them raises. The spec is reassembled from whichever pieces are left over once every recognised keyword is pulled out, in their original order, so a value spec with its own embedded `-` (a sign flag, or a custom fill character) survives being combined with another component: `mul-->10.2f` is `mul` plus the value spec `->10.2f` (fill `-`, align `>`), not three components.

**A value spec needs a real DSL token to activate.** A _bare_ value spec with no recognised keyword anywhere in it — `.3g`, or `:>10` (`:` as a fill character) — is untouched by this parser and keeps going through the unrelated, pre-existing `pspec_fallback` mechanism unchanged: scalar-only, space-joined, unit unaffected (see below). Only once it's paired with a keyword (`mul-.3g`, or `values-.3g` on its own) does the DSL claim it and the array-formatter path apply.

Product-style specs only decompose meaningfully for a type that registers `pparts`; a type that skips this still accepts them without raising, rendering `str(obj)` as one opaque fragment. That is a deliberate degrade, not a bug: a display path must not raise just because one field's type forgot to register. The unit component works the same way: `long` falls back to the short form rather than raising when a unit has no long name (a composite like `km / s`, or dimensionless).

`short` and `compact` are easy to mix up — both mean "terse" in English, but they name different things entirely: `short` is this DSL's array component (shape/dtype summary, product-style); `compact` is a call-style `FORMAT_PRESETS` entry (short class name, kept full array values). That near-miss is also why the array component's own default-form token is spelled `values`, not `compact` — a first pass tried `compact` there too, and a bare `f"{q:compact}"` silently stopped meaning the call-style preset once `_parse_product_spec` claimed it first. `pspec` checking `FORMAT_PRESETS` before the DSL parse (see below) is the general fix; picking non-colliding names is what avoids needing it in the first place.

`dims` is the one call-style preset that is inherently type-specific: only a unit system has "dimension names" to show instead of units.

```{code-block} python

>>> usys = u.unitsystem("kpc", "Myr", "Msun", "radian")
>>> f"{usys:dims}"
'LTMAUnitSystem(length, time, mass, angle)'
```

It is the precedent for how a type extends the shared vocabulary with a call-style preset of its own, rather than forcing every concept to be universal. Do this when a concept has no equivalent for other types; reuse an existing name — or the product-style DSL — when it does.

Rules for extending the vocabulary:

- A genuinely new _component_ of product-style rendering (a thing that can vary orthogonally to markup/array/separator/unit) is a new DSL token, not a new flat preset — the whole point of the DSL is that components compose instead of each combination needing its own hand-written entry.
- A concept with no product-style equivalent, and no fit in the DSL, is a new `FORMAT_PRESETS` entry (as `dims` is). Reuse an existing name if its meaning already fits.
- Lowercase, one word, no punctuation, and not a DSL token already spoken for (`html`, `latex`, `short`, `values`, `mul`, `bare`, `long`).
- Check any new name against _both_ tables, not just the one you're adding to — a DSL token named `compact` collided with the pre-existing `FORMAT_PRESETS` entry of that name during development, which is why the array component is called `values` instead.
- `""` (empty spec) always means `str(obj)`, and `!r`/`!s` already cover `repr`/`str` — never add a `"repr"` or `"str"` preset, and never let `""` become a dict key.

Rules for a type joining the engine:

- Register `pspec_fallback` if a _non-preset, non-DSL_ spec should mean something (as `.2f` does for a quantity's value). Skip it and an unrecognised spec raises `ValueError`.
- Register `pparts` to make the product-style DSL decompose meaningfully instead of falling back to a single opaque `str(obj)` fragment. Accept `value_spec` (a per-element Python format spec, or `None`) and `long_unit` (a bool) as keyword arguments — even if unused — so the engine can pass them through without every type needing to opt in explicitly; `**kw` alone is enough to stay compatible with a future component this DSL doesn't have yet.
