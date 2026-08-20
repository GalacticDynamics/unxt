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

A product-style spec is up to three optional, independent components, canonically spelled `<markup>-<array>-<separator>`:

| component | values | omitted means |
| --- | --- | --- |
| markup | `html`, `latex` | `text` |
| array | `short` (shape/dtype summary, e.g. `f32[3]`) | the compact form, e.g. `[1., 2., 3.]` |
| separator | `mul` (show `*`), `bare` (space, no operator) | `mul` |

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
```

Omitting a component takes its default, which is what makes the short forms just an unremarkable case of the same grammar rather than a separate alias table: `mul` is short for `text-mul` (equivalently `mul` — separator is the only component present), `html` is short for `html-mul` (markup present, separator defaults to `mul`), and `short` is short for `text-mul-short`.

The parser accepts the three tokens in any order — `html-bare` and `bare-html` render identically — but `<markup>-<array>-<separator>` is the one spelling used in docs and error messages, so pick it when writing a spec by hand. Two tokens from the same component (`mul-bare`, `html-latex`) is invalid and raises `ValueError`, same as an unknown spec.

Product-style specs only decompose meaningfully for a type that registers `pparts`; a type that skips this still accepts them without raising, rendering `str(obj)` as one opaque fragment. That is a deliberate degrade, not a bug: a display path must not raise just because one field's type forgot to register.

`short` and `compact` are easy to mix up — both mean "terse" in English, but they cut different axes. `short` summarizes the _array_ (shape and dtype, product-style); `compact` shortens the _class name_ while keeping the array's compact values (call-style). Read the grammar, not the names, when in doubt.

`dims` is the one call-style preset that is inherently type-specific: only a unit system has "dimension names" to show instead of units.

```{code-block} python

>>> usys = u.unitsystem("kpc", "Myr", "Msun", "radian")
>>> f"{usys:dims}"
'LTMAUnitSystem(length, time, mass, angle)'
```

It is the precedent for how a type extends the shared vocabulary with a call-style preset of its own, rather than forcing every concept to be universal. Do this when a concept has no equivalent for other types; reuse an existing name — or the product-style DSL — when it does.

Rules for extending the vocabulary:

- A genuinely new _component_ of product-style rendering (a fourth thing that can vary orthogonally to markup/array/separator) is a new DSL token, not a new flat preset — the whole point of the DSL is that components compose instead of each combination needing its own hand-written entry.
- A concept with no product-style equivalent, and no fit in the DSL, is a new `FORMAT_PRESETS` entry (as `dims` is). Reuse an existing name if its meaning already fits.
- Lowercase, one word, no punctuation, and not a DSL token already spoken for (`html`, `latex`, `short`, `mul`, `bare`).
- `""` (empty spec) always means `str(obj)`, and `!r`/`!s` already cover `repr`/`str` — never add a `"repr"` or `"str"` preset, and never let `""` become a dict key.

Rules for a type joining the engine:

- Register `pspec_fallback` if a _non-preset, non-DSL_ spec should mean something (as `.2f` does for a quantity's value). Skip it and an unrecognised spec raises `ValueError`.
- Register `pparts` to make the product-style DSL decompose meaningfully instead of falling back to a single opaque `str(obj)` fragment.
