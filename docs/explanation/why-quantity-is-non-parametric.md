# Why `Quantity` is not parametric

In v1, `unxt.Quantity` encoded the physical dimension in its _type_: `Quantity["length"]` and `Quantity["time"]` were distinct Python classes. In v2 the default `Quantity` is a single class for every dimension, and the parametric behaviour moved out to the separate [`unxts.parametric`](../packages/unxts.parametric/index) package. This page explains why, because the reason is not the one most people guess.

## The reason people expect: `jit` cache misses

The intuitive story is that a class-per-dimension multiplies `jax.jit` compilations, and collapsing to one class collapses the cache. That story is wrong, and it is worth dismantling before the real one.

A quantity's `unit` is a **static** field. It lives in the pytree aux data — the treedef — and the treedef is part of the `jit` cache key. So a jitted function already specializes per distinct unit, with _either_ class: a function compiled for a length in metres is not reused for a time in seconds, because their units differ. And since a unit already implies its dimension, `ParametricQuantity`'s per-dimension class is redundant with the per-unit key. It adds no compilations. Both classes produce exactly the same number of `jit` compilations for the same inputs.

That part is inherent to putting units in the type system, and no choice of class structure changes it.

## The real reason: type proliferation

What the parametric class actually costs is the proliferation itself. Every physical dimension your program touches:

- creates a new Python class the first time it is used, via `plum`'s parametric machinery,
- registers a new JAX pytree node type and grows `plum`'s dispatch tables, all of which have to be tracked and searched, and
- pays a per-construction cost for dimension inference and `__check_init__` validation.

A single-class `Quantity` avoids all of it: one class, one registered pytree type, no on-the-fly class creation, and a shorter construction and dispatch path. The win is a smaller, simpler type surface and cheaper per-operation overhead — not fewer compilations.

There is a design argument alongside the performance one. A type system earns its keep when the distinctions it draws are ones you act on. Most `unxt` code never dispatches on dimension and never asks a type to enforce one; it does arithmetic and lets the unit algebra catch the mistakes. For that majority, a class per dimension is a cost with no corresponding benefit, and defaults should serve the majority.

## When parametric is still right

`ParametricQuantity` remains available, and is the better choice, when you genuinely need one of two things:

- **Runtime dimension checking.** `up.PQ["length"](1, "s")` raises immediately; `u.Q["length"](1, "s")` does not check at all.
- **Dimension-specific `plum` dispatch.** `up.PQ["length"]` is a distinct type you can write in an annotation and dispatch on; `u.Q["length"] is u.Quantity`.

Both are real needs — they are why the class still exists rather than having been deleted. Making it a separate, opt-in package rather than the default means the people who need those guarantees pay for them and nobody else does.

For everything else — arithmetic, unit conversion, JAX transforms, interop — the two classes behave identically.

## See also

- {doc}`../how-to/migrate-to-v2` — what changes in your code, and how to update it.
- {doc}`sharp-bits` — including what parametric types do to pytree registration.
- The [`unxts.parametric` package](../packages/unxts.parametric/index).
