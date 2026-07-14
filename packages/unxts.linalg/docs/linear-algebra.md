# Linear algebra with unit tracking

The whole point of `QuantityMatrix` is that linear-algebra operations carry the per-element units through the computation. `unxts.linalg` registers Quax rules for the underlying JAX primitives, so the [`quaxed`](https://github.com/GalacticDynamics/quaxed) drop-in `numpy` operators work directly on `QuantityMatrix` objects.

```{code-block} python
>>> import jax.numpy as jnp
>>> import quax
>>> import unxt as u
>>> import unxts.linalg as ul
```

## Matrix and vector products

`unxts.linalg` exposes the four NumPy / Array-API contraction functions, each of which contracts the shared axis and multiplies the corresponding per-element units:

| Function          | Operands        | Result            |
| ----------------- | --------------- | ----------------- |
| `ul.matmul(A, B)` | matrix × matrix | matrix            |
| `ul.matvec(A, v)` | matrix × vector | vector            |
| `ul.vecmat(v, A)` | vector × matrix | vector            |
| `ul.vecdot(a, b)` | vector · vector | scalar `Quantity` |

`matmul` contracts the shared axis and multiplies the corresponding units — here a dimensionless identity leaves a vector's units untouched:

```{code-block} python
>>> A = ul.QuantityMatrix(jnp.eye(3),
...                       unit=(("", "", ""), ("", "", ""), ("", "", "")))
>>> v = ul.QuantityMatrix(jnp.array([1.0, 2.0, 3.0]), unit=("m", "m", "m"))
>>> ul.matvec(A, v).value
Array([1., 2., 3.], dtype=float32)
```

Units on the contracted axis are combined and converted to a common reference, so mixed units are handled correctly:

```{code-block} python
>>> A2 = ul.QuantityMatrix(jnp.array([[1.0, 2.0], [3.0, 4.0]]),
...                        unit=(("m", "km"), ("m", "km")))
>>> v2 = ul.QuantityMatrix(jnp.array([1.0, 1.0]), unit=("s", "s"))
>>> ul.matvec(A2, v2).value
Array([2001., 4003.], dtype=float32)
```

A plain JAX array or an ordinary `unxt.Quantity` on one side is treated as dimensionless / uniform-unit respectively.

### Batches: prefer `matvec`/`vecmat` over `matmul`

A `QuantityMatrix` carries its logical 1-D/2-D units in the _trailing_ axes and treats _leading_ axes of the value array as batch dimensions. All four products broadcast those batch axes. But note a subtlety inherited from NumPy: a batched vector's value `(B, K)` is shape-indistinguishable from a matrix, so **`matmul` cannot express a batched matrix-vector product** — use `matvec` (and `vecmat`) for those.

```{code-block} python
>>> # A batch of 2 matrices applied to a batch of 2 vectors.
>>> Ab = ul.QuantityMatrix(
...     jnp.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
...     unit=(("m", "m"), ("m", "m")),
... )
>>> vb = ul.QuantityMatrix(jnp.ones((2, 2)), unit=("s", "s"))
>>> ul.matvec(Ab, vb).value
Array([[ 3.,  7.],
       [11., 15.]], dtype=float32)
```

## Transpose and diagonal

`.T` swaps both the values and the unit structure of a 2-D matrix:

```{code-block} python
>>> a = ul.QuantityMatrix(jnp.array([[1.0, 2.0], [3.0, 4.0]]),
...                       unit=(("m", "s"), ("kg", "rad")))
>>> a.T.unit.to_string()
'((m, kg), (s, rad))'
```

`.diag()` extracts the diagonal as a 1-D `QuantityMatrix`, operating directly on the static unit structure so it works under `jax.jit`:

```{code-block} python
>>> M = ul.QuantityMatrix(jnp.diag(jnp.array([1.0, 2.0, 3.0])),
...                       unit=(("m", "s", "kg"),
...                             ("m", "s", "kg"),
...                             ("m", "s", "kg")))
>>> d = M.diag()
>>> d.unit.to_string()
'(m, s, kg)'
>>> d.value
Array([1., 2., 3.], dtype=float32)
```

## Determinant and inverse

`unxts.linalg` provides custom `det` and `inv` primitives with full JAX transform support (JIT, autodiff, vmap). On plain arrays they behave like `jnp.linalg.det` / `jnp.linalg.inv`; on a `QuantityMatrix` (wrapped with `quax.quaxify`) they additionally track units.

The determinant multiplies the main-diagonal units:

```{code-block} python
>>> from unxts.linalg import det, inv
>>> G = ul.QuantityMatrix(jnp.array([[2.0, 0.0], [0.0, 3.0]]),
...                       unit=(("m2", "m2"), ("m2", "m2")))
>>> quax.quaxify(det)(G)
Quantity(Array(6., dtype=float32), unit='m4')
```

The inverse carries the reciprocal units:

```{code-block} python
>>> B = ul.QuantityMatrix(jnp.array([[4.0, 0.0], [0.0, 1.0]]),
...                       unit=(("m2", "m2"), ("m2", "m2")))
>>> r = quax.quaxify(inv)(B)
>>> r.unit.to_string()
'((1 / m2, 1 / m2), (1 / m2, 1 / m2))'
>>> r.value
Array([[0.25, 0.  ],
       [0.  , 1.  ]], dtype=float32)
```

Because `det` and `inv` are real JAX primitives, they compose with `jax.jit`, `jax.grad`, and `jax.vmap` on plain arrays:

```{code-block} python
>>> import jax
>>> jax.jit(det)(jnp.array([[2.0, 0.0], [0.0, 3.0]]))
Array(6., dtype=float32)
>>> jax.grad(det)(jnp.array([[2.0, 0.0], [0.0, 3.0]]))
Array([[3., 0.],
       [0., 2.]], dtype=float32)
```
