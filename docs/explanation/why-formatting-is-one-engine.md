# Why formatting is one engine

`unxt` has four ways to turn a value into a string: `repr`, `str`, `__format__`, and the representations Jupyter asks for. They share one spec vocabulary and one entry point, and most of them share the renderer as well. The constructor form `repr` produces is the exception, for reasons below. This page is about why that is worth doing and what it cost.

For the spec vocabulary itself, see {doc}`../reference/formatting`. To get something formatted, see {doc}`../how-to/format-quantities`; to make a type of your own formattable, see {doc}`../how-to/make-a-type-formattable`.

## Why one renderer and not four

Four renderers means four places a change has to land, and nothing to stop them drifting. That is not hypothetical here. Before this engine existed, `_repr_latex_` wrapped a unit's `repr` in `$…$` by slicing the first and last character off it. For a plain unit that worked. For a unit whose `repr` was not `$…$` — a `UnitsMatrix`, say — it quietly ate two real characters. The fault was structural: the LaTeX path had no way to know what the other three knew, so nothing kept it in step with them.

So a type no longer says how it renders. It says what it is _made of_: a tree of labelled fragments, returned from `pparts`. Each output route lays that same tree out differently. Add a markup and every type gets it. Register a type and it gets every markup, including ones added later by somebody else.

The cost is indirection. Reading `_repr_latex_` used to show you the whole story; now it shows you a call into a renderer, and the story is spread across the fragment tree, the markup table and the layout. That is a real loss, paid once, against four paths that can no longer disagree.

## Why the spec grammar scans left to right

A spec is a run of keywords that may end in a Python format spec: `mul-name-.2f`. The obvious way to parse that is to split on `-` and sort the pieces into keywords and leftovers. It does not work, because a Python format spec can contain `-` itself — as a sign flag in `-.2f`, or as a fill character in `->10.2f`. Split on `-` and `->10.2f` becomes two tokens, neither of which means anything.

So the scan is strictly left to right and stops looking for keywords at the first token that is not one. Everything after that point is the format spec, `-` and all. No token needs to be examined for what it might be; its position settles it.

The price is one rule the reader has to carry: the format spec goes last. `mul-.2f-name` does not fail with "unexpected keyword after format spec" — it cannot, because by then there are no keywords, only the string `.2f-name`, which is simply not a valid format spec. The alternative designs are worse: a delimiter other than `-` would make the common cases uglier, and sniffing each token for format-spec-ness would make adding a keyword able to change what an existing spec means.

## Why keywords share one namespace

`latex` names a markup. `name` names a unit spelling. Neither says which axis it belongs to, so a run can be written in any order: `html-bare` and `bare-html` are the same request.

The cost arrives when two packages want one word. `dim` is a unit spelling in `unxt`. In `coordinax` it could as reasonably mean a manifold's dimensionality. Both readings are right, and a flat namespace cannot hold both.

The way out is that `axis=word` is always available, and is in fact the real form: a keyword _sets an axis to a value_, so `unit=dim` is the operation spelled out and bare `dim` is shorthand for it. A bare word resolves while exactly one axis claims it; when two do, it raises and names both ways to disambiguate. Because that is per token, a collision costs one word in one spec — `latex-mul-unit=dim-.3f` — rather than a rewrite. A package that expects to share vocabulary can write the explicit form from the start and never be affected.

The alternative, namespacing every keyword by its axis from the beginning, would have made every ordinary spec longer to pay for a collision that has not happened yet.

## Why the constructor form is not built from fragments

Every rendering but one is assembled from the fragments a type reports through `pparts`. The `call` layout is the exception: it hands off to `wadler_lindig.pformat`, which uses the type's own `__pdoc__`.

The two forms answer different questions. `product` layout shows what a value **is** — `1. m`. The `call` layout shows how to **rebuild** it — `Quantity(1., unit='m')`. Only the type knows the second, and knowing it takes more than knowing how the value looks.

Unit systems are the clearest case. Their `repr` passes the units as a list:

```{code-block} python

>>> from unxt import unitsystem
>>> repr(unitsystem("kpc", "Myr", "Msun", "radian"))
"unitsystem(['kpc', 'Myr', 'solMass', 'rad'])"

```

The list is not decoration. `unitsystem` reads a single bare string as the name of a registered system, so `unitsystem('m')` raises instead of building a metre system:

```{code-block} python

>>> try:
...     unitsystem("m")
... except ValueError as e:
...     print(str(e).split(".")[0])
'm' is not a registered unit system (available: atomic, cgs, dimensionless, galactic, geometrized, hep, planck, si, solarsystem)

```

The list is what distinguishes "these units" from "that system". A renderer working from display fragments has no way to know that; the type does. So `call` layout delegates, and `eval(repr(x)) == x` stays a promise each type keeps for itself:

```{code-block} python

>>> usys = unitsystem("kpc", "Myr", "Msun", "radian")
>>> eval(repr(usys)) == usys
True

```

The names in the output have to be in scope for `eval` to resolve them, which is why that example imports `unitsystem` directly.

## Why an inert axis is allowed to do nothing

A spec can be accepted and change nothing. Asking for `name` on a quantity in `km / s` still renders `km / s`, because a composite unit has no single long name and the axis falls back to the symbol.

This could have been an error, and there is a case for it: a spec that does nothing is usually a mistake, and silence is how mistakes survive. But erroring would break composite types, which depend on that same forwarding. An `Interval` of two quantities does not know the unit axis exists; it passes the axes it does not handle down to its parts, and `f"{interval:name}"` reaches the nested quantities. A type cannot declare in advance which axes it honours, because the honest answer depends on the value: this unit has a long name, that one does not.

So whether an axis mattered is settled by experiment rather than by declaration — render again with the axis reset to its default and see whether the output moves. `inert_axes` does exactly that, and is a call you make while debugging rather than a setting, because each check costs a second render.

## Why the engine is separate from `unxt`'s layer

The code is split in two, at the line where it will eventually be cut into separate packages. One half knows about fragments, markups, the parser, the layouts and the axis registry, and nothing about quantities: it imports nothing from `unxt`, `jax`, `numpy` or `astropy`. The other half is the axes `unxt` contributes to the grammar.

The split exists because `coordinax` and `galax` need the same machinery. The alternative was for each to depend on `unxt` for its string formatting, which would make `unxt` a dependency of packages that have nothing else to do with units. The domain-agnostic half is intended to become a package of its own, named `pparts` after the extension point everything turns on.

A claim like "imports nothing from `unxt`" decays the moment someone adds a convenient import, so it is enforced rather than written down: a test reads the engine's import list and fails if anything domain-specific appears.

Once extracted, `unxt`, `coordinax` and `galax` register into the engine as peers. There is no privileged set of axes, and an axis added downstream is read out of a spec exactly like a built-in one.
