# Why formatting is one engine

`unxt` has four ways to turn a value into a string: `repr`, `str`, `__format__`, and the representations Jupyter asks for. They share one spec vocabulary and one entry point, and most of them share the renderer as well.

The renderer is a separate distribution, [`wadler-lindig-parts`][wlp], and the argument for the design is documented with it: [why formatting is one engine][wlp-why] covers one renderer rather than four, the left-to-right spec scan, the shared keyword namespace, the inert-axis rule, and why the line breaking belongs to `wadler_lindig`. The two sections below are the parts that belong to `unxt`.

For the spec vocabulary itself, see {doc}`../reference/formatting`. To get something formatted, see {doc}`../how-to/format-quantities`; to make a type of your own formattable, see {doc}`../how-to/make-a-type-formattable`.

## Why the engine is a separate distribution

The engine knows about fragments, markups, the parser, the layouts and the axis registry, and nothing about quantities. `unxt` depends on it and contributes the axes it needs — `unit` and `abbrev`, and the `compact` / `full` / `dims` aliases.

The split exists because `coordinax` and `galax` need the same machinery. The alternative was for each to depend on `unxt` for its string formatting, which would make `unxt` a dependency of packages that have nothing else to do with units. The three register into the engine as peers: there is no privileged set of axes, and an axis added downstream is read out of a spec exactly like a built-in one. A separate distribution also cannot import `unxt`, so "the engine knows nothing about quantities" holds by packaging rather than by anyone remembering the rule.

What `unxt` gives up is control of the vocabulary. An axis `coordinax` registers is visible in a spec written against a `Quantity`, and a word two packages both want resolves only in its `axis=word` form. That is the price of the registry being shared, and {doc}`../reference/formatting` documents how a collision reads.

## Why `unitsystem`'s `repr` is not built from fragments

Every rendering but one is assembled from the fragments a type reports through `pparts`. The `call` layout is the exception: it hands off to `wadler_lindig.pformat`, which uses the type's own `__pdoc__`. So `eval(repr(x)) == x` is a promise each `unxt` type keeps for itself rather than one the renderer makes on its behalf.

Unit systems are where that matters. Their `repr` passes the units as a list:

```{code-block} python

>>> from unxt import unitsystem
>>> repr(unitsystem("kpc", "Myr", "Msun", "radian"))
"unitsystem(['kpc', 'Myr', 'solMass', 'rad'])"

```

The list is what distinguishes "these units" from "that system". `unitsystem` reads a single bare string as the name of a registered system, so `unitsystem('m')` raises instead of building a metre system:

```{code-block} python

>>> try:
...     unitsystem("m")
... except ValueError as e:
...     print(str(e).split(".")[0])
'm' is not a registered unit system (available: atomic, cgs, dimensionless, galactic, geometrized, hep, planck, si, solarsystem)

```

A renderer working from display fragments has no way to know that, and the type does. `product` layout shows what a unit system **is**; `call` layout shows how to **rebuild** it, and only the type knows the second:

```{code-block} python

>>> usys = unitsystem("kpc", "Myr", "Msun", "radian")
>>> f"{usys:product}"
'unitsystem(kpc, Myr, solMass, rad)'
>>> eval(repr(usys)) == usys
True

```

The names in the output have to be in scope for `eval` to resolve them, which is why that example imports `unitsystem` directly.

[wlp]: https://galacticdynamics.github.io/wadler-lindig-parts/
[wlp-why]: https://galacticdynamics.github.io/wadler-lindig-parts/explanation/why-formatting-is-one-engine/
