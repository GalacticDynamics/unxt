# Quantity matrices

`QuantityMatrix` (alias `QM`) stores one numeric JAX array together with a static [`UnitsMatrix`](units-matrix.md) that gives the unit of **each** logical element. The shape of the unit structure decides whether the object behaves as a heterogeneous vector (1-D) or matrix (2-D).

```{code-block} python
>>> import jax.numpy as jnp
>>> import unxt as u
>>> import unxts.linalg as ul
```

## Construction

A 1-D `QuantityMatrix` takes a flat array and a tuple of units, one per element:

```{code-block} python
>>> qv = ul.QuantityMatrix(jnp.array([1.0, 2.0, 3.0]), unit=("m", "s", "kg"))
>>> qv.value
Array([1., 2., 3.], dtype=float32)
>>> qv.unit.to_string()
'(m, s, kg)'
>>> qv.ndim, qv.shape
(1, (3,))
```

A 2-D `QuantityMatrix` takes a 2-D array and a nested tuple of units:

```{code-block} python
>>> qm = ul.QuantityMatrix(jnp.ones((2, 2)), unit=(("m", "s"), ("kg", "rad")))
>>> qm.unit.to_string()
'((m, s), (kg, rad))'
>>> qm.ndim, qm.shape
(2, (2, 2))
```

Leading dimensions of the value array are treated as batch dimensions; only the trailing 1 or 2 axes are "logical" and must match the unit structure.

`QM` is a short alias for `QuantityMatrix` (mirroring `u.Q` for `Quantity`):

```{code-block} python
>>> ul.QM is ul.QuantityMatrix
True
```

### From a component dictionary

`QuantityMatrix.from_cdict` packs a dict of quantities into a 1-D matrix, preserving each entry's unit:

```{code-block} python
>>> v = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "s"), "z": u.Q(3.0, "kg")}
>>> ul.QuantityMatrix.from_cdict(v).unit.to_string()
'(m, s, kg)'
```

You may select and reorder a subset of keys:

```{code-block} python
>>> ul.QuantityMatrix.from_cdict(v, keys=("z", "x")).unit.to_string()
'(kg, m)'
```

## Indexing

Indexing a 1-D `QuantityMatrix` returns an ordinary `unxt.Quantity` — the element and its scalar unit:

```{code-block} python
>>> qv[0]
Quantity(Array(1., dtype=float32), unit='m')
>>> qv[2]
Quantity(Array(3., dtype=float32), unit='kg')
```

Indexing a row of a 2-D matrix returns a 1-D `QuantityMatrix`, while a full `[i, j]` index returns a scalar `Quantity`:

```{code-block} python
>>> qm2 = ul.QuantityMatrix(jnp.ones((2, 3)),
...                         unit=(("m", "s", "kg"), ("rad", "deg", "m")))
>>> qm2[0]
QuantityMatrix(Array([1., 1., 1.], dtype=float32), unit='(m, s, kg)')
>>> qm2[1, 2]
Quantity(Array(1., dtype=float32), unit='m')
```

## Arithmetic and unit conversion

Addition and subtraction convert each element of the right operand into the corresponding unit of the left operand before combining; the result adopts the left operand's units:

```{code-block} python
>>> import quaxed.numpy as qnp
>>> a = ul.QuantityMatrix(jnp.ones(3), unit=("m", "s", "kg"))
>>> b = ul.QuantityMatrix(jnp.ones(3), unit=("km", "ms", "g"))
>>> result = qnp.add(a, b)
>>> result.unit.to_string()
'(m, s, kg)'
>>> result.value
Array([1001.   ,    1.001,    1.001], dtype=float32)
```

Scalar multiplication scales the values and leaves the units unchanged:

```{code-block} python
>>> (2 * a).value
Array([2., 2., 2.], dtype=float32)
```

You can convert a whole `QuantityMatrix` to a compatible unit structure with `uconvert`:

```{code-block} python
>>> q = ul.QuantityMatrix(jnp.array([[1.0, 2.0], [3.0, 4.0]]),
...                       unit=(("m", "rad"), ("m", "rad")))
>>> target = u.unit((("km", "deg"), ("km", "deg")))
>>> q.uconvert(target).unit.to_string()
'((km, deg), (km, deg))'
```

When every element shares the same unit, a `QuantityMatrix` converts to a plain `unxt.Quantity`:

```{code-block} python
>>> import plum
>>> uniform = ul.QuantityMatrix(jnp.array([1.0, 2.0, 3.0]), unit=("m", "m", "m"))
>>> plum.convert(uniform, u.Q)
Quantity(Array([1., 2., 3.], dtype=float32), unit='m')
```

Mixed units make that conversion ambiguous, so it is rejected:

```{code-block} python
>>> try:
...     plum.convert(qv, u.Q)
... except ValueError as e:
...     print(e)
Cannot convert QuantityMatrix to Quantity unless all units are identical.
```
