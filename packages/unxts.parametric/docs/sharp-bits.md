# Sharp Bits

Gotchas specific to `ParametricQuantity`. For the core `unxt` sharp bits, see the [unxt Sharp Bits guide](../../guides/sharp-bits).

```{code-block} python
>>> import unxt as u
>>> import unxts.parametric as up
```

## Parametric types multiply pytree types (not jit compilations)

A common misconception is that feeding `ParametricQuantity` of different dimensions into a jitted function adds a recompilation _per dimension_. It does not. Recompilation is driven by the **unit**, a static field for _both_ classes, so a jitted function specializes per distinct unit either way. Because a unit already implies its dimension, `ParametricQuantity`'s per-dimension _type_ is redundant with the per-unit cache key and adds no extra compilations:

```{code-block} python
>>> import jax

>>> @jax.jit
... def square(x):
...     return x**2

>>> # Both classes recompile per *unit* ("m" and "s" are different units):
>>> _ = square(u.Q(5.0, "m"))
>>> _ = square(u.Q(5.0, "s"))    # 2 compilations
>>> _ = square(up.PQ(5.0, "m"))
>>> _ = square(up.PQ(5.0, "s"))  # also 2 compilations
```

What `ParametricQuantity` _does_ add is a new Python class — and a new registered JAX pytree node type — for every physical dimension, created on demand. That grows the pytree and `plum` dispatch type surface that must be tracked and searched, and adds per-construction dimension inference and a validation check. The default `Quantity` is a single type for all dimensions, so it avoids that proliferation and its overhead.

## Equality with `StaticValue`

A normal `ParametricQuantity` (backed by a JAX array) follows NumPy broadcasting: `==` returns an **element-wise boolean array**, not a scalar `bool`. That makes it unusable as a `jax.jit` `static_argnames` argument. Wrapping the value in a `StaticValue` makes `==` return a **scalar `bool`** (structural equality, like a tuple), so the whole quantity is hashable and safe as a static argument. This comparison is **unit-blind** — quantities are equal only when their unit labels match, so physically-equal but differently-labelled quantities stay distinct static-arg cache keys. Use {func}`unxt.equivalent` (or the `.is_equivalent` method) for a unit-aware "same physical quantity" check; the example below shows the unit-blind `==`:

```{code-block} python
>>> import numpy as np

>>> scale = up.PQ(u.quantity.StaticValue(np.array([2.0, 3.0])), "m")
>>> sv_km = u.quantity.StaticValue(np.array([0.002, 0.003]))
>>> bool(up.PQ(scale.value, "m") == up.PQ(sv_km, "km"))  # different unit labels
False
```
