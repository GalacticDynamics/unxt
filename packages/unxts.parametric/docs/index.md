# `unxts.parametric`

```{toctree}
:maxdepth: 1
:hidden:
```

`unxts.parametric` provides `ParametricQuantity` (alias `PQ`): a quantity that encodes its physical **dimension** in its _type_. It is the opt-in counterpart to the lightweight, non-parametric default `unxt.Quantity`.

`ParametricQuantity` used to be the default `Quantity` in `unxt` v1. As of v2 the non-parametric class is the default and the parametric class lives here, in its own package. See the [migration guide](../../migration.md) for the full mapping.

## Install

::::{tab-set}

:::{tab-item} uv

```bash
uv add unxts.parametric
```

:::

:::{tab-item} pip

```bash
pip install unxts.parametric
```

:::

::::

Throughout this guide we import `unxt` as `u` and `unxts.parametric` as `up` (so `ParametricQuantity` is `up.PQ`):

```{code-block} python
>>> import unxt as u
>>> import unxts.parametric as up
```

## Why the default `Quantity` is non-parametric

`Quantity` (`u.Q`) is the lightweight, non-parametric default: a single class — and a single JAX pytree type — for all physical dimensions. `ParametricQuantity` (`up.PQ`) instead encodes the dimension in its _type_ — `ParametricQuantity["length"]` and `ParametricQuantity["time"]` are distinct Python classes, created on demand, and each is registered as its own JAX pytree node type.

That per-dimension type proliferation carries real costs: a new class is created the first time each dimension is used (via `plum`'s parametric machinery), every one is a separately-registered pytree and dispatch type that JAX and `plum` must track, and construction runs dimension inference plus a validation check. The single-class `Quantity` avoids all of it — one class, one pytree type, lighter construction and dispatch.

A note on `jax.jit`: this is **not** about jit cache misses. The `unit` is a _static_ field, so it lives in the pytree aux data (the treedef), which is part of the jit cache key — a jitted function therefore specializes per distinct unit with **either** class (a call on `"m"` is not reused for `"s"`). Choosing the parametric class does not change that per-unit compilation; it only adds the redundant per-dimension _type_ (a unit already implies its dimension).

## When to reach for `ParametricQuantity`

Reach for `ParametricQuantity` only when you need one of its two extra features:

1. **Runtime dimension checking** — `up.PQ["length"](1, "m")` validates the unit against the dimension at construction; the default `u.Q["length"](1, "m")` accepts the subscript for compatibility but does not check it.
2. **Dispatch on specific dimensions** — `up.PQ["length"]` is a real type usable in `plum` dispatch annotations; `u.Q["length"]` is just `Quantity`.

Everything else — arithmetic, unit conversion, JAX transforms, interop — works identically with either class.

| Type | Use case | Dimension in type | Performance |
| --- | --- | --- | --- |
| `unxt.Quantity` | Default choice | ❌ None | Better |
| `ParametricQuantity` | Dimension dispatch / runtime checking | ✅ Yes | Good (per-dimension type) |
| `unxt.StaticQuantity` | Compile-time constants | ✅ Yes | Best (no tracer) |

## Construction and runtime dimension checking

When a `ParametricQuantity` is constructed it is parametrized by the unit's dimension. This can be specified explicitly:

```{code-block} python
>>> up.PQ["length"](1, "m")
ParametricQuantity(Array(1, dtype=int32...), unit='m')
```

or inferred from the unit:

```{code-block} python
>>> up.PQ(1, "m")
ParametricQuantity(Array(1, dtype=int32...), unit='m')
```

When given explicitly, `ParametricQuantity` checks the input dimensions. Here a length-parametrized class (correctly) refuses a unit of time:

```{code-block} python
>>> try:
...     up.PQ["length"](1, "s")
... except Exception as e:
...     print(e)
Physical type mismatch.
```

That should catch some bugs! By contrast, the default `Quantity` accepts the subscript as a no-op and does **not** check:

```{code-block} python
>>> u.Q["length"](1, "s")  # no error; subscript is informational only
Quantity(Array(1, dtype=int32...), unit='s')
```

Filling a `ParametricQuantity`'s parameter and constructing an instance may be separated:

```{code-block} python
>>> LengthQuantity = up.PQ["length"]
>>> LengthQuantity
<class 'unxt...ParametricQuantity[PhysicalType('length')]'>
```

## The dimension of a parametric class

Because a `ParametricQuantity` encodes its dimension in the _class itself_, `unxt.dimension_of` works on a parametrized class — not only on instances:

```{code-block} python
>>> u.dimension_of(up.PQ["length"])  # a parameterized class carries a dimension
PhysicalType('length')

>>> try: u.dimension_of(up.PQ)  # ... the unparameterized class does not
... except Exception as e: print(e)
can only get dimensions from parametrized ParametricQuantity -- ParametricQuantity[dim].
```

The default `Quantity` carries no dimension, so `dimension_of` on the _class_ raises — see the [Dimensions guide](../../guides/dimensions.md).

## Dimension annotations for type checking

Because the dimension lives in the type, you can annotate function signatures with dimensioned `ParametricQuantity` types, and `unxt`'s runtime type checking (via [`jaxtyping`](https://pypi.org/project/jaxtyping/)) will enforce them. This dimension-level checking is what `ParametricQuantity` adds over the default `Quantity` (whose type carries only dtype and shape); see the [Type Checking guide](../../guides/type-checking.md) for how to enable runtime checking.

```{code-block} python
>>> from jaxtyping import Shaped, jaxtyped
>>> from beartype import beartype as typechecker

>>> @jaxtyped(typechecker=typechecker)
... def velocity(
...     x: Shaped[up.PQ["length"], "N"],
...     t: Shaped[up.PQ["time"], "N"],
... ) -> Shaped[up.PQ["speed"], "N"]:
...     return x / t

>>> x = up.PQ([2.], "m")
>>> t = up.PQ([1.], "s")

>>> velocity(x, t)
ParametricQuantity(Array([2.], dtype=float32), unit='m / s')
```

The base class `AbstractParametricQuantity` and the concrete `unxt.Quantity` are _not_ parametric — `Quantity[<dimension>]` does nothing and is informational only. Explore [`plum`](https://beartype.github.io/plum/parametric.html) for more on parametric classes.

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

A normal `ParametricQuantity` (backed by a JAX array) follows NumPy broadcasting: `==` returns an **element-wise boolean array**, not a scalar `bool`. That makes it unusable as a `jax.jit` `static_argnames` argument. Wrapping the value in a `StaticValue` makes `==` return a **scalar `bool`** (structural equality, like a tuple), so the whole quantity is hashable and safe as a static argument. Unit conversion is applied before comparing, so equal physical values in compatible units compare equal:

```{code-block} python
>>> import numpy as np

>>> scale = up.PQ(u.quantity.StaticValue(np.array([2.0, 3.0])), "m")
>>> sv_km = u.quantity.StaticValue(np.array([0.002, 0.003]))
>>> bool(up.PQ(scale.value, "m") == up.PQ(sv_km, "km"))  # same physical value
True
```

## Public API

`unxts.parametric` exposes:

- `ParametricQuantity` — the dimension-parametrized quantity (alias `PQ`).
- `AbstractParametricQuantity` — its abstract base.

Importing `unxts.parametric` also registers, as import side effects, the promotion rules, `plum` conversions, and JAX primitive rules that let `ParametricQuantity` interoperate with the rest of `unxt`.

Install: `pip install unxts.parametric`
