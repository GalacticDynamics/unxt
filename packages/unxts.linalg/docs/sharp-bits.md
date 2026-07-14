# Sharp bits

`QuantityMatrix` covers the common heterogeneous-unit vector/matrix cases, but a few restrictions follow from its design. Keep these in mind.

```{code-block} python
>>> import jax.numpy as jnp
>>> import quaxed.numpy as qnp
>>> import unxt as u
>>> import unxts.linalg as ul
```

## Only 1-D and 2-D structures

The unit structure is limited to 1-D (vector) and 2-D (matrix); there is no support for higher-rank _logical_ structures. (Leading **batch** dimensions on the value array are fine — only the trailing one or two axes are logical.)

```{code-block} python
>>> from unxts.linalg import UnitsMatrix
>>> try:
...     UnitsMatrix(jnp.zeros((2, 2, 2)))
... except (TypeError, ValueError) as e:
...     print("rejected")
rejected
```

## `det` and `inv` assume compatible units

`det` uses the product of the **main-diagonal** units, and `inv` assumes a **uniform** unit that it can reciprocate. These are exactly right for diagonal metrics and for matrices whose cofactor products share one physical dimension (the common case for coordinate metrics), but they are not general heterogeneous-unit determinants/inverses. Both require a 2-D matrix:

```{code-block} python
>>> import quax
>>> v = ul.QuantityMatrix(jnp.array([1.0, 2.0]), unit=("m", "s"))
>>> try:
...     quax.quaxify(ul.det)(v)
... except ValueError as e:
...     print("needs a 2-D matrix")
needs a 2-D matrix
```

## `diag` under `jax.jit` needs uniform units

The `.diag()` **method** operates on the static unit structure and works for heterogeneous units, even under `jit`:

```{code-block} python
>>> M = ul.QuantityMatrix(jnp.diag(jnp.array([1.0, 2.0, 3.0])),
...                       unit=(("m", "s", "kg"),
...                             ("m", "s", "kg"),
...                             ("m", "s", "kg")))
>>> M.diag().unit.to_string()
'(m, s, kg)'
```

By contrast `qnp.diag` lowers to a `gather`, whose indices are traced under `jit`; there the unit of each output element cannot be resolved individually, so **all units must be equal**. Prefer the `.diag()` method for heterogeneous-unit matrices.

## The `matmul` _function_ can't do a batched matrix-vector product

A batched vector's value `(B, K)` is shape-indistinguishable from a matrix. The **`@` operator is fine** — `QuantityMatrix.__matmul__` dispatches on the logical rank, so `A @ v` correctly does a (batched) matrix-vector product. But the raw **function** forms `jnp.matmul` / `quaxed.numpy.matmul` infer their contraction from the value shapes and so cannot; they silently succeed only when the sizes coincide and otherwise raise. `unxts.linalg.matmul` refuses a batched vector operand outright, pointing you at `matvec`. Prefer `@`, `ul.matvec`, or `ul.vecmat`. See [Linear algebra](linear-algebra.md#batches).

## It is a Quax type, not a materialisable array

`QuantityMatrix` is a `quax` array-ish value: it flows through `quax.quaxify`-ed functions but refuses to _materialise_ into a plain array (its elements have no single dtype-plus-unit), so use `.value` / `.unit` to inspect it, and `plum.convert(..., u.Q)` only when every unit is identical (see [Quantity matrices](quantity-matrix.md)).
