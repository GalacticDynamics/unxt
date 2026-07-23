# `unxts.linalg`

```{toctree}
:maxdepth: 1
:hidden:

quantity-matrix
units-matrix
linear-algebra
tutorial-metric
sharp-bits
```

`unxts.linalg` provides `QuantityMatrix` (alias `QM`): a quantity container whose elements may each carry a **different** unit. It is backed by a single JAX array plus a static `UnitsMatrix` describing the per-element units, and supports both 1-D (heterogeneous vector) and 2-D (heterogeneous matrix) structures.

It is useful for objects whose entries have mixed physical dimensions — Jacobians, metric tensors, and coordinate change-of-basis matrices — where a single scalar unit (as on `unxt.Quantity`) is not expressive enough.

## Install

The recommended install adds `unxts.linalg` alongside `unxt` via the `linalg` [extra](https://peps.python.org/pep-0508/#extras), so the two are resolved together as a compatible set:

::::{tab-set}

:::{tab-item} uv

```bash
uv add "unxt[linalg]"
```

:::

:::{tab-item} pip

```bash
pip install "unxt[linalg]"
```

:::

::::

Or install the package directly:

::::{tab-set}

:::{tab-item} uv

```bash
uv add unxts.linalg
```

:::

:::{tab-item} pip

```bash
pip install unxts.linalg
```

:::

::::

Throughout these guides we import `unxt` as `u` and `unxts.linalg` as `ul` (so `QuantityMatrix` is `ul.QM`):

```{code-block} python
>>> import jax.numpy as jnp
>>> import unxt as u
>>> import unxts.linalg as ul
```

## At a glance

A 1-D `QuantityMatrix` is a vector whose entries each have their own unit:

```{code-block} python
>>> qv = ul.QuantityMatrix(jnp.array([1.0, 2.0, 3.0]), unit=("m", "s", "kg"))
>>> qv.unit.to_string()
'(m, s, kg)'
>>> 2 * qv
QuantityMatrix(Array([2., 4., 6.], dtype=float32), unit='(m, s, kg)')
```

Indexing a single element yields an ordinary `unxt.Quantity`:

```{code-block} python
>>> qv[0]
Quantity(Array(1., dtype=float32), unit='m')
```

## Guides

- [Quantity matrices](quantity-matrix) — constructing `QuantityMatrix`, indexing, unit conversion, and arithmetic.
- [Units matrices](units-matrix) — the immutable, hashable `UnitsMatrix` unit structure.
- [Linear algebra](linear-algebra) — matmul, transpose, `diag`, `det`, and `inv` with per-element unit tracking.
- [Tutorial: a heterogeneous metric](tutorial-metric) — a worked end-to-end example.
- [Sharp bits](sharp-bits) — the 1-D/2-D restriction and the uniform-unit requirements under `jax.jit`.

## Public API

`unxts.linalg` exposes:

- `QuantityMatrix` — the heterogeneous-unit matrix/vector (alias `QM`).
- `UnitsMatrix` — the immutable, hashable per-element unit structure.
- `matmul`, `matvec`, `vecmat`, `vecdot` — the four Array-API matrix/vector products, each resolving the batched matrix-vs-vector ambiguity that `@` alone cannot.
- `det`, `inv` — unit-tracking determinant and inverse (with their JAX primitives `det_p`, `inv_p`).
- `cdict_units` — extract per-key units from a component dictionary.

Importing `unxts.linalg` also registers, as import side effects, the Quax primitive rules (add/sub, mul/div, dot-general/matmul, transpose, gather/diag, reduce-sum) and the `plum` conversions/dispatch that let `QuantityMatrix` interoperate with the rest of `unxt`.
