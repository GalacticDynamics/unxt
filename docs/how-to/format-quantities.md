# How to format a quantity as a string

Every `unxt` type accepts a **format spec** in an f-string. The spec is the same vocabulary on every type, so what you learn here carries over to `coordinax` and `galax`.

This guide covers the common jobs. For the full list of axes, keywords and defaults, see {doc}`../reference/formatting`. To change what `repr` and `str` do by default — for a call, a block or a whole project — see {doc}`control-display`. To make a type of your own accept these specs, see {doc}`make-a-type-formattable`.

```{code-block} python

>>> import unxt as u
>>> q = u.Q([1.0, 2, 3], "m")

```

## Pick a rendering

Name what you want. Keywords can be combined in any order with `-`:

```{code-block} python

>>> f"{q:mul}"                  # show the multiplication
'[1., 2., 3.] * m'
>>> f"{q:name}"                 # spell the unit out
'[1., 2., 3.] meter'
>>> f"{q:type}"                 # shape and dtype instead of values
'f32[3] m'
>>> f"{q:latex}"                # for a paper or a notebook
'$[1.,~2.,~3.] \\, \\mathrm{m}$'
>>> f"{q:html-mul}"             # combine them
'<span>[1., 2., 3.]</span> * <span>m</span>'

```

## Format the numbers

Put an ordinary Python format spec at the end. It applies to each element:

```{code-block} python

>>> qq = u.Q([1.234, 2.345], "m")
>>> f"{qq:.2f}"
'[1.23, 2.35] m'
>>> f"{qq:mul-name-.2f}"
'[1.23, 2.35] * meter'

```

The format spec has to go last. `mul-.2f-name` fails, because everything after `.2f` is read as part of the format spec.

## Control the line width

If a value is too wide for its destination, set `width=`:

```{code-block} python

>>> f"{q:width=10}"
'[1., 2.,\n   3.]\n  m'

>>> f"{q:call-indent=4-width=20}"
"Quantity(\n    [1., 2., 3.],\n    unit='m'\n)"

```

This works in every markup, and breaks at the same points plain text does, because a markup's tags cost no columns:

```{code-block} python

>>> f"{q:html-width=10}"
'<span>[1., 2.,\n   3.]</span>\n  <span>m</span>'

```

## Get output you can paste back into code

Ask for `call` layout — it renders the constructor expression:

```{code-block} python

>>> from unxt import unitsystem
>>> usys = unitsystem("kpc", "Myr", "Msun", "radian")
>>> f"{usys:call}"
"unitsystem(['kpc', 'Myr', 'solMass', 'rad'])"
>>> eval(f"{usys:call}") == usys
True

```

`repr` uses this layout, so the same holds for `repr`. The names in the output must be in scope where you evaluate it.

## Check whether a spec did anything

A spec can be accepted and change nothing: `f"{q:name}"` on a composite unit falls back to the symbol, because there is no single long name.

```{code-block} python

>>> f"{u.Q(1.0, 'km/s'):name}"
'1. km / s'

```

To find out which part of a spec did nothing, ask the engine. It re-renders once per axis, so call it while debugging rather than on every format:

```{code-block} python

>>> from unxt._src.fmt import inert_axes, parse_spec, render

>>> q = u.Q(1.0, "km/s")
>>> spec = parse_spec("name")
>>> inert_axes(q, spec, render(q, spec))
['unit']

```

## See also

- {doc}`../reference/formatting` — every axis, keyword and default.
- {doc}`control-display` — change the defaults rather than passing a spec.
- {doc}`make-a-type-formattable` — give a type of your own the same specs.
- {doc}`../explanation/why-formatting-is-one-engine` — why it works this way.
