# `unxts.parametric`

```{toctree}
:maxdepth: 1
:hidden:

tutorial-dimensions
quantity
type-checking
configuration
sharp-bits
```

`unxts.parametric` provides `ParametricQuantity` (alias `PQ`): a quantity that encodes its physical **dimension** in its _type_. It is the opt-in counterpart to the lightweight, non-parametric default `unxt.Quantity`.

`ParametricQuantity` used to be the default `Quantity` in `unxt` v1. As of v2 the non-parametric class is the default and the parametric class lives here, in its own package. See the [migration guide](../../how-to/migrate-to-v2) for the full mapping.

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

Throughout these pages we import `unxt` as `u` and `unxts.parametric` as `up` (so `ParametricQuantity` is `up.PQ`):

```{code-block} python
>>> import unxt as u
>>> import unxts.parametric as up
```

## At a glance

`ParametricQuantity` is used just like `Quantity`, but it encodes the physical dimension in its _type_ — and can check it at construction:

```{code-block} python

>>> up.PQ(1.0, "m")  # dimension inferred from the unit
ParametricQuantity(Array(1., dtype=float32, ...), unit='m')

>>> up.PQ["length"](1.0, "m")  # dimension checked against the unit
ParametricQuantity(Array(1., dtype=float32, ...), unit='m')

```

## Should you use it?

Reach for `ParametricQuantity` only when you need one of its two extra features:

1. **Runtime dimension checking** — `up.PQ["length"](1, "s")` raises; the default `u.Q["length"](1, "s")` accepts the subscript for compatibility but does not check it.
2. **Dispatch on specific dimensions** — `up.PQ["length"]` is a real type usable in `plum` dispatch annotations; `u.Q["length"]` is just `Quantity`.

Everything else — arithmetic, unit conversion, JAX transforms, interop — works identically with either class. The cost of the parametric class, and why the non-parametric one became the default, is set out in the core docs under [Why `Quantity` is not parametric](../../explanation/why-quantity-is-non-parametric); the comparison table against `StaticQuantity` is in [the sharp bits](../../explanation/sharp-bits).

## Pages

**Tutorial**

- [Let the type system catch a unit mistake](./tutorial-dimensions) — start here: watch a wrong unit get rejected at construction, then write a function that dispatches on physical dimension.

**Reference**

- [`ParametricQuantity`](./quantity) — construction, runtime dimension checking, dimension-specific dispatch, promotion with the default `Quantity`, and `dimension_of` on a parametrized class.
- [Configuration](./configuration) — the `include_params` display option.

**How-to**

- [How to check dimensions at runtime](./type-checking) — dimension annotations enforced by `jaxtyping`.

**Discussion**

- [The parametric sharp bits](./sharp-bits) — pytree-type proliferation and `StaticValue` equality.

## Public API

`unxts.parametric` exposes:

- `ParametricQuantity` — the dimension-parametrized quantity (alias `PQ`).
- `AbstractParametricQuantity` — its abstract base.
- `config` — the `unxts.parametric.config` singleton (see [Configuration](./configuration)).

Importing `unxts.parametric` also registers, as import side effects, the promotion rules, `plum` conversions, and JAX primitive rules that let `ParametricQuantity` interoperate with the rest of `unxt`.
