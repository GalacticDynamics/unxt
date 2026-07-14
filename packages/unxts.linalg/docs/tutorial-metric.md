# Tutorial: a heterogeneous metric

This tutorial walks through a small end-to-end example: representing a diagonal _metric_ whose entries have mixed physical dimensions, then computing its determinant, inverse, and diagonal — all while `unxts.linalg` keeps the units straight. A single-unit `unxt.Quantity` could not represent such an object.

```{code-block} python
>>> import jax.numpy as jnp
>>> import quax
>>> import quaxed.numpy as qnp
>>> import unxt as u
>>> import unxts.linalg as ul
```

## The setup

Consider a 2-D configuration space with a radial coordinate (metres) and an angular coordinate (radians). A diagonal metric `g` that turns coordinate differences into a squared length has diagonal entries with _different_ units: the radial part is `m2` (length²) while the angular part carries `m2 / rad2` so that `g · dθ²` still has units of length².

We build `g` as a 2×2 `QuantityMatrix` with per-element units:

```{code-block} python
>>> g = ul.QuantityMatrix(
...     jnp.array([[1.0, 0.0], [0.0, 4.0]]),
...     unit=(("m2", "m2"), ("m2", "m2 / rad2")),
... )
>>> g.unit.to_string()
'((m2, m2), (m2, m2 / rad2))'
```

## Reading off the diagonal

`.diag()` extracts the diagonal as a 1-D `QuantityMatrix`, preserving each entry's (heterogeneous) unit — this is the part of the metric that actually scales each coordinate:

```{code-block} python
>>> d = g.diag()
>>> d.unit.to_string()
'(m2, m2 / rad2)'
>>> d.value
Array([1., 4.], dtype=float32)
```

## Determinant

The determinant of a diagonal metric is the product of its diagonal units — here `m2 · m2/rad2`:

```{code-block} python
>>> quax.quaxify(ul.det)(g).unit.to_string()
'm4 / rad2'
```

## Inverse

The inverse metric carries the reciprocal of each unit, ready to _raise_ an index again:

```{code-block} python
>>> ginv = quax.quaxify(ul.inv)(g)
>>> ginv.value
Array([[1.  , 0.  ],
       [0.  , 0.25]], dtype=float32)
>>> ginv.unit.to_string()
'((1 / m2, 1 / m2), (1 / m2, rad2 / m2))'
```

## Recap

- A `QuantityMatrix` let us attach a distinct unit to every entry of the metric — impossible with a single-unit `unxt.Quantity`.
- `.diag()`, `det`, and `inv` all propagated those per-element units automatically.
- Everything is plain JAX underneath, so the same objects flow through `jax.jit`, `jax.grad`, and `jax.vmap`.

See [Sharp bits](sharp-bits.md) for the current restrictions (1-D/2-D only, and the uniform-unit requirements of some operations under `jax.jit`).
