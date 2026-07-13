# `unxts.parametric`

```{toctree}
:maxdepth: 1
:hidden:

quantity
dimensions
type-checking
configuration
sharp-bits
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

Throughout these guides we import `unxt` as `u` and `unxts.parametric` as `up` (so `ParametricQuantity` is `up.PQ`):

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
| `ParametricQuantity` | Dimension dispatch / runtime checking | ✅ Yes | Good (a distinct type per dimension) |
| `unxt.StaticQuantity` | Compile-time constants | ❌ None | Best (no tracer) |

## Guides

- [Parametric Quantities](quantity.md) — construction, runtime dimension checking, dimension-specific dispatch.
- [Dimensions](dimensions.md) — `dimension_of` on a parametric class.
- [Type Checking](type-checking.md) — dimension annotations enforced at runtime.
- [Configuration](configuration.md) — the `include_params` display option.
- [Sharp Bits](sharp-bits.md) — pytree-type proliferation and `StaticValue` equality.

## Public API

`unxts.parametric` exposes:

- `ParametricQuantity` — the dimension-parametrized quantity (alias `PQ`).
- `AbstractParametricQuantity` — its abstract base.
- `config` — the `unxts.parametric.config` singleton (see [Configuration](configuration.md)).

Importing `unxts.parametric` also registers, as import side effects, the promotion rules, `plum` conversions, and JAX primitive rules that let `ParametricQuantity` interoperate with the rest of `unxt`.

Install: `pip install unxts.parametric`
