# Units matrices

`UnitsMatrix` is the immutable, hashable unit structure carried by a [`QuantityMatrix`](quantity-matrix.md). It wraps a 1-D or 2-D structure of `unxt.AbstractUnit` objects and provides tuple-style indexing, transposition, and serialization. Because it is hashable and static, it lives in a `QuantityMatrix`'s pytree _aux data_, so it is available concretely even under `jax.jit`.

```{code-block} python
>>> import unxt as u
>>> from unxts.linalg import UnitsMatrix
```

## Construction

1-D from a tuple of units (or unit strings), 2-D from a nested tuple:

```{code-block} python
>>> units_1d = UnitsMatrix(("m", "s", "kg"))
>>> units_1d.shape
(3,)
>>> units_2d = UnitsMatrix((("m", "s"), ("kg", "rad")))
>>> units_2d.shape
(2, 2)
```

`unxt.unit` is overloaded to build a `UnitsMatrix` from a (nested) tuple, so you rarely construct one directly — passing a tuple as the `unit=` of a `QuantityMatrix` is enough:

```{code-block} python
>>> u.unit(("m", "s", "kg"))
UnitsMatrix("(m, s, kg)")
```

## Indexing and iteration

Indexing a single element returns the underlying unit; indexing a row of a 2-D structure returns a `UnitsMatrix`:

```{code-block} python
>>> units_2d[0, 1]
Unit("s")
>>> units_2d[0]
UnitsMatrix("(m, s)")
```

## Serialization

`to_string` gives a compact human-readable form; `to_tuple` gives a nested tuple of units:

```{code-block} python
>>> units_2d.to_string()
'((m, s), (kg, rad))'
>>> UnitsMatrix(("m", "s")).to_tuple()
(Unit("m"), Unit("s"))
```

## Transpose and inverse

`.T` transposes the (2-D) unit structure, and `.inverse()` raises each unit to the power −1 — the operations underpinning `QuantityMatrix.T` and matrix inversion:

```{code-block} python
>>> UnitsMatrix((("m", "s"), ("kg", "rad"))).T
UnitsMatrix("((m, kg), (s, rad))")
>>> UnitsMatrix(("m2", "s2")).inverse()
UnitsMatrix("(1 / m2, 1 / s2)")
```

## Equality and hashing

Two `UnitsMatrix` objects compare equal when their shapes and units match; a `UnitsMatrix` also compares equal to an equivalent (nested) tuple. Hashing is by value, so a `UnitsMatrix` is a valid static field / dict key:

```{code-block} python
>>> UnitsMatrix(("m", "s")) == ("m", "s")
True
>>> hash(UnitsMatrix(("m", "s"))) == hash(UnitsMatrix(("m", "s")))
True
```
