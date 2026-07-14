# `unxts.linalg`

`unxts.linalg` provides `QuantityMatrix` (alias `QM`): a quantity container whose elements may each carry a **different** unit. It is backed by a single JAX array plus a static `UnitsMatrix` describing the per-element units, and supports both 1-D (heterogeneous vector) and 2-D (heterogeneous matrix) structures.

It is useful for objects whose entries have mixed physical dimensions — Jacobians, metric tensors, and coordinate change-of-basis matrices.

## Install

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
>>> import unxt as u
>>> import unxts.linalg as ul
```

## Public API

`unxts.linalg` exposes:

- `QuantityMatrix` — the heterogeneous-unit matrix/vector (alias `QM`).
- `UnitsMatrix` — the immutable, hashable per-element unit structure.
- `det`, `inv` — unit-tracking determinant and inverse (with their JAX primitives `det_p`, `inv_p`).
- `cdict_units` — extract per-key units from a component dictionary.

Importing `unxts.linalg` also registers, as import side effects, the Quax primitive rules (add/sub/matmul/transpose/gather/reduce-sum) and the `plum` conversions/dispatch that let `QuantityMatrix` interoperate with the rest of `unxt`.
