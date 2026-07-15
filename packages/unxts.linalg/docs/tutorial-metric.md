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

`inv` mixes the matrix entries, so it is only unit-correct when the units are **uniform** (the reciprocal then applies throughout). Our metric is heterogeneous, so `inv` refuses it rather than return a misleading per-element reciprocal:

```{code-block} python
>>> try:
...     quax.quaxify(ul.inv)(g)
... except ValueError as e:
...     print(str(e).split(";")[0])
inv on a QuantityMatrix requires uniform units (all entries equal)
```

For a uniform-unit metric the inverse carries the single reciprocal unit:

```{code-block} python
>>> h = ul.QuantityMatrix(jnp.array([[4.0, 0.0], [0.0, 1.0]]),
...                       unit=(("m2", "m2"), ("m2", "m2")))
>>> quax.quaxify(ul.inv)(h).unit.to_string()
'((1 / m2, 1 / m2), (1 / m2, 1 / m2))'
```

## Recap

- A `QuantityMatrix` let us attach a distinct unit to every entry of the metric — impossible with a single-unit `unxt.Quantity`.
- `.diag()` and `det` propagated those per-element units automatically; `inv` is defined for uniform units (see [Sharp bits](sharp-bits.md)).
- Everything is plain JAX underneath, so the same objects flow through `jax.jit`, `jax.grad`, and `jax.vmap`.

See [Sharp bits](sharp-bits.md) for the current restrictions (1-D/2-D only, and the uniform-unit requirements of `inv` and of `diag` under `jax.jit`).
