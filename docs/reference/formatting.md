# Format specs

A format spec selects how a value renders. It is accepted by `__format__` — so by f-strings, `format()` and `str.format` — and by `unxt._pparts.pspec`.

For instructions, see {doc}`../how-to/format-quantities`. For the reasoning behind the design, see {doc}`../explanation/why-formatting-is-one-engine`.

```{code-block} python

>>> import unxt as u
>>> q = u.Q([1.0, 2, 3], "m")

```

## Grammar

```
spec := keyword ("-" keyword)* ["-" <python format spec>]
```

A spec is a `-`-joined run of keywords, optionally ending in a Python format spec that is applied to each element of the value.

The parse is left-to-right and total: tokens are consumed while they are keywords, and the first token that is not a keyword ends keyword parsing. Everything from that token onward, including any further `-`, is the Python format spec.

Consequence: **the format spec goes last.** `mul-name-.2f` is valid. `mul-.2f-name` sends `.2f-name` to the value axis, which rejects it as a malformed Python format spec.

An empty spec is `str(obj)`.

## Axes

Each keyword sets exactly one axis. Keywords are pairwise disjoint, so a run is order-independent: `html-bare` and `bare-html` are the same spec.

| Axis | Keywords | Default | Applies to | Renders as |
| --- | --- | --- | --- | --- |
| `layout` | `call`, `product` | `product` | both | `Quantity([1.], unit='m')` / `[1.] m` |
| `value` | `array`, `values`, `type`, or a Python format spec | `values` | both | `Array([1.], dtype=float32)` / `[1.]` / `f32[1]` |
| `markup` | `text`, `html`, `latex` | `text` | `product` | see [Markups](#markups) |
| `unit` | `symbol`, `name`, `dim` | `symbol` | both | `m` / `meter` / `length` |
| `sep` | `mul`, `bare` | `bare` | `product` | `1. * m` / `1. m` |
| `abbrev` | `abbrev` | off | `call` | `Q(...)` instead of `Quantity(...)` |
| `width` | `width=<n>` | 88 | both | line width the layout wraps at |
| `indent` | `indent=<n>` | 2 | both | indent of a wrapped layout |

Naming an axis the chosen layout does not have is an error, not a silent no-op: `call-mul` raises, because `call` layout has no separator.

```{code-block} python

>>> f"{q:type}"
'f32[3] m'
>>> f"{q:name}"
'[1., 2., 3.] meter'
>>> f"{q:call}"
"Quantity([1., 2., 3.], unit='m')"
>>> f"{q:mul}"
'[1., 2., 3.] * m'

```

`width` and `indent` take a non-negative decimal integer. ASCII digits only; `width=１０` is not a count and falls through to the value axis.

```{code-block} python

>>> f"{q:call-width=20}"
"Quantity(\n  [1., 2., 3.],\n  unit='m'\n)"

>>> f"{q:call-indent=4-width=20}"
"Quantity(\n    [1., 2., 3.],\n    unit='m'\n)"

```

`width` applies in every markup. A markup's wrapper text is emitted but costs no columns, so HTML breaks at the same points plain text does.

```{code-block} python

>>> f"{q:width=10}"
'[1., 2.,\n   3.]\n  m'

>>> f"{q:html-width=10}"
'<span>[1., 2.,\n   3.]</span>\n  <span>m</span>'

```

## Aliases

An alias expands textually into keywords before parsing. It introduces no meaning the grammar does not already have, and combining one with a further keyword raises whatever its expansion would raise.

| Alias     | Expands to    |
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

## Markups

| Markup  | Wraps output in | Element separator | Escaping                 |
| ------- | --------------- | ----------------- | ------------------------ |
| `text`  | nothing         | `, `              | none                     |
| `html`  | nothing         | `, `              | `html.escape`            |
| `latex` | `$…$`           | `,~`              | LaTeX special characters |

`latex` overrides the separator roles, because math mode discards literal whitespace: `bare` renders `\,` and `mul` renders `\;`. The row also overrides `gap` (`\ `) and `pm` (`\pm`); no `unxt` type emits those roles, and they are there for packages whose fragments do.

```{code-block} python

>>> f"{q:html}"
'<span>[1., 2., 3.]</span> <span>m</span>'
>>> f"{q:latex}"
'$[1.,~2.,~3.] \\, \\mathrm{m}$'

```

## Explicit axis assignment

Any keyword may be written as `axis=word`. This is the full form; a bare keyword is its shorthand.

```{code-block} python

>>> f"{q:unit=dim}"
'[1., 2., 3.] length'
>>> f"{q:markup=latex-sep=mul}" == f"{q:latex-mul}"
True

```

A bare word resolves while exactly one axis claims it. When two axes claim one word, the bare form raises and names both resolutions:

```
f"{q:dim}"  ->  ambiguous keyword 'dim'; claimed by 'manifold', 'unit'.
                Write it as one of: manifold=dim, unit=dim
```

Assignment is per token, so a collision costs one word: `latex-mul-unit=dim-.3f`.

`=` is also a fill character in a Python format spec, where it appears only at position 0 or 1. A spec such as `=>6` names no axis and falls through to the value axis.

```{code-block} python

>>> f"{u.Q(3, 'm'):=>6}"
'=====3 m'

```

## Registration

| Function | Adds | Removes with |
| --- | --- | --- |
| `register_axis(Axis(...), *, replace=False)` | an axis and its keywords | `unregister_axis(name)` |
| `register_alias(name, expansion)` | an alias | `unregister_alias(name)` |
| `register_markup(name, row)` | a markup row | `unregister_markup(name)` |

`Axis` fields:

| Field | Meaning |
| --- | --- |
| `name` | the axis, and its key in a `Spec` |
| `keywords` | spec word → the value it sets |
| `default` | the value when no keyword names this axis |
| `layouts` | layout → a function turning this axis's value into renderer keyword arguments. Membership is applicability: an axis applies to exactly the layouts it has an entry for |
| `free_text` | the layouts in which this axis's value may be arbitrary text instead of a keyword |

A `register_markup` row requires `_content`, `wrap`, `vsep` and `escape`. Any other key is a per-role override.

Constraints, each of which raises rather than overwriting:

- A keyword may not already be an alias, and an alias may not already be a keyword.
- An axis name may not be registered twice unless `replace=True`.
- At most one axis may declare `free_text`, since a spec has one trailing run. In `unxt` that axis is `value`.
- No keyword may itself be a legal Python format spec, which would make the scan rule prefer the keyword reading over a meaning users already had.

`Spec.of(**overrides)` builds a spec by hand, filling every registered axis from the registry.

`inert_axes(obj, spec, rendered)` returns the axes a spec set explicitly that changed nothing. It re-renders once per axis set, so it is a debugging call rather than something to run on every format.

## Errors

All of these raise `ValueError` from `__format__`.

| Condition | Example |
| --- | --- |
| unknown keyword, and not a valid Python format spec | `f"{q:nope}"` |
| one axis set twice | `f"{q:mul-bare}"`, `f"{q:type-.2f}"` |
| axis not applicable to the layout | `f"{q:call-mul}"` |
| ambiguous bare keyword | two axes claiming one word |
| unknown markup | `register_markup` not called for it |

A type that registers no `pparts` renders as `str(obj)` in one fragment, but raises `TypeError` for a value format spec, since it has declared no elements to format.
