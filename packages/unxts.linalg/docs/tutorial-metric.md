# Build a heterogeneous metric

In this tutorial we will build a metric tensor whose entries have _different_ physical units, read its diagonal, take its determinant, and convert the whole thing to another unit system — all with the units tracked for us. A single-unit `unxt.Quantity` cannot represent such an object at all.

You need `unxt` and `unxts.linalg` installed and nothing else.

## Set up

```{code-block} python
>>> import jax.numpy as jnp
>>> import quax
>>> import unxt as u
>>> import unxts.linalg as ul

```

## Build the metric

Consider a flat 2-D space described in polar coordinates: a radial coordinate measured in metres and an angular coordinate measured in radians. A metric turns a pair of coordinate displacements into a squared length, so its entries cannot all carry the same unit — the radial part must be `m2`, while the angular part must be `m2 / rad2` for `g · dθ²` to still come out as an area.

Let's build it as a 2×2 `QuantityMatrix`, giving the units as a nested tuple with one entry per element:

```{code-block} python
>>> g = ul.QM(
...     jnp.array([[1.0, 0.0], [0.0, 4.0]]),
...     unit=(("m2", "m2"), ("m2", "m2 / rad2")),
... )
>>> g.unit.to_string()
'((m2, m2), (m2, m2 / rad2))'

```

Notice that the units printed back as a nested structure mirroring the array. That structure is stored separately from the numbers, which is what lets each element keep its own unit.

The object knows it is a 2-D matrix:

```{code-block} python
>>> g.shape, g.ndim
((2, 2), 2)

```

## Look at a single element

Indexing with a full `[i, j]` gives back an ordinary `unxt.Quantity` — the number together with that one element's unit:

```{code-block} python
>>> g[0, 0]
Quantity(Array(1., dtype=float32), unit='m2')

>>> g[1, 1]
Quantity(Array(4., dtype=float32), unit='m2 / rad2')

```

Two entries of the same matrix, two different units. That is the whole point of the type.

## Read off the diagonal

For a diagonal metric the interesting content is the diagonal itself — the factor scaling each coordinate. `.diag()` extracts it as a 1-D `QuantityMatrix`:

```{code-block} python
>>> d = g.diag()
>>> d.unit.to_string()
'(m2, m2 / rad2)'

>>> d.value
Array([1., 4.], dtype=float32)

```

Notice that the heterogeneous units came through unchanged. `.diag()` works on the stored unit structure rather than on the numbers, so each output element keeps the unit it had.

## Take the determinant

`unxts.linalg` provides `det` as a real JAX primitive. It is a `quax`-aware function, so we wrap the call with `quax.quaxify` to let it see a `QuantityMatrix`:

```{code-block} python
>>> det_g = quax.quaxify(ul.det)(g)
>>> det_g
Quantity(Array(4., dtype=float32), unit='m4 / rad2')

```

Look at what happened to the unit. The determinant of a diagonal matrix is the product of its diagonal entries, so the unit is the product of the diagonal units: `m2 × m2/rad2 = m4/rad2`. We never wrote that down — it came out of the per-element bookkeeping.

## Convert it to other units

Because every element's unit is tracked, the whole matrix converts at once. Let's express the same metric in kilometres by handing `uconvert` a matching unit structure:

```{code-block} python
>>> target = u.unit((("km2", "km2"), ("km2", "km2 / rad2")))
>>> g_km = g.uconvert(target)
>>> g_km.unit.to_string()
'((km2, km2), (km2, km2 / rad2))'

>>> g_km.value
Array([[1.e-06, 0.e+00],
       [0.e+00, 4.e-06]], dtype=float32)

```

Each number was scaled by the factor its own element required, and one square kilometre is a million square metres, so the entries dropped by `1e-6` as expected.

## What we built

You now have a metric tensor whose every entry carries its own physical unit, and you have taken its diagonal, its determinant and a unit conversion — with the units propagated for you at each step and never written down by hand after construction. Everything here is plain JAX underneath, so the same object flows through `jax.jit`, `jax.grad` and `jax.vmap`.

## Where to go next

- [`QuantityMatrix`](quantity-matrix) — construction, indexing, arithmetic.
- [Linear-algebra operations](linear-algebra) — the full set of unit-tracking operations.
- [The linalg sharp bits](sharp-bits) — the restrictions, including when `inv` refuses a heterogeneous matrix and why.
