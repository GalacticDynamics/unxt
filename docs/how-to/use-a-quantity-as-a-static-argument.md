# How to pass a quantity as a `jax.jit` static argument

A `jax.jit` static argument must be **hashable** and must compare with a scalar `bool`. An ordinary `Quantity` is neither — its value is a JAX array, so `==` returns an array — and passing one as `static_argnames` fails:

```{code-block} python
>>> import functools as ft
>>> import jax
>>> import jax.numpy as jnp
>>> import numpy as np
>>> import unxt as u

>>> @ft.partial(jax.jit, static_argnames=("scale",))
... def rescale(x, *, scale):
...     return x * jnp.asarray(scale.value)

>>> try:
...     rescale(jnp.ones(2), scale=u.Q(jnp.array([2.0, 3.0]), "m"))
... except ValueError:
...     print("not hashable")
not hashable

```

`unxt` gives you two ways to fix that. Which one you want depends on whether the _whole quantity_ is a constant, or only its value.

## If the whole quantity is constant, use `StaticQuantity`

`StaticQuantity` stores its value as a NumPy array, which is hashable:

```{code-block} python
>>> scale = u.StaticQuantity(np.array([2.0, 3.0]), "m")
>>> scale
StaticQuantity(array([2., 3.]), unit='m')

>>> rescale(jnp.ones(2), scale=scale)
Array([2., 3.], dtype=float32)

```

Call it again with the same value and JAX reuses the compiled code; call it with a different one and it recompiles, which is the behaviour you asked for by making it static.

A concrete JAX array is accepted and materialised back to NumPy, so you do not have to convert first:

```{code-block} python
>>> u.StaticQuantity(jnp.array([2.0, 3.0]), "m")
StaticQuantity(array([2., 3.], dtype=float32), unit='m')

```

Only a _traced_ value is rejected — a tracer cannot be a compile-time constant. Inside `jit`, `vmap` or `grad`, use a plain `Quantity`.

## If only the value is constant, wrap it in `StaticValue`

When the surrounding code expects a `Quantity` — a function signature, a dataclass field, a dispatch annotation — keep the type and make only the value static:

```{code-block} python
>>> scale = u.Q(u.quantity.StaticValue(np.array([2.0, 3.0])), "m")
>>> scale
Quantity(StaticValue(array([2., 3.])), unit='m')

>>> rescale(jnp.ones(2), scale=scale)
Array([2., 3.], dtype=float32)

```

It is still a `Quantity`, so everything that dispatches on `Quantity` still works.

## Know what `==` does to your cache key

Both static forms make `==` return a scalar `bool`, which is what lets them be cache keys:

```{code-block} python
>>> a = u.Q(u.quantity.StaticValue(np.array([2.0, 3.0])), "m")
>>> b = u.Q(u.quantity.StaticValue(np.array([2.0, 3.0])), "m")
>>> a == b
True

```

That comparison is **unit-blind** — it compares unit _labels_, not physical amounts. `2 m` and `0.002 km` are the same amount but different keys:

```{code-block} python
>>> c = u.Q(u.quantity.StaticValue(np.array([0.002, 0.003])), "km")
>>> a == c
False

```

This is deliberate. If `==` converted first, two quantities that trace to _different_ compiled code would collapse into one cache entry. When you want the physical question instead, use `equivalent` — see {doc}`compare-quantities`.

## Choosing

| Situation | Use |
| --- | --- |
| The whole quantity is a compile-time constant | `StaticQuantity` |
| Surrounding code needs a `Quantity`, but the value is constant | `Quantity(StaticValue(...), unit)` |
| The value is traced, or changes every call | plain `Quantity`, not static |

## See also

- {doc}`../reference/quantity` — `StaticQuantity`, `StaticValue`, `AllowValue`.
- {doc}`../explanation/equality-and-equivalence` — why static equality is unit-blind.
- {doc}`optimize-performance` — the other half of `jit` cost: pytree boundaries.
