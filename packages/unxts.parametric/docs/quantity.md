# Parametric Quantities

Constructing a `ParametricQuantity` and its runtime dimension checking. For the lightweight, non-parametric default, see the [unxt Quantity guide](../../guides/quantity).

```{code-block} python
>>> import unxt as u
>>> import unxts.parametric as up
```

## Construction and runtime dimension checking

When a `ParametricQuantity` is constructed it is parametrized by the unit's dimension. This can be specified explicitly:

```{code-block} python
>>> up.PQ["length"](1, "m")
ParametricQuantity(Array(1, dtype=int32...), unit='m')
```

or inferred from the unit:

```{code-block} python
>>> up.PQ(1, "m")
ParametricQuantity(Array(1, dtype=int32...), unit='m')
```

When given explicitly, `ParametricQuantity` checks the input dimensions. Here a length-parametrized class (correctly) refuses a unit of time:

```{code-block} python
>>> try:
...     up.PQ["length"](1, "s")
... except Exception as e:
...     print(e)
Physical type mismatch.
```

That should catch some bugs! By contrast, the default `Quantity` accepts the subscript as a no-op and does **not** check:

```{code-block} python
>>> u.Q["length"](1, "s")  # no error; subscript is informational only
Quantity(Array(1, dtype=int32...), unit='s')
```

## Dispatch on specific dimensions

Filling a `ParametricQuantity`'s parameter and constructing an instance may be separated — the parametrized class is a real type, usable in `plum` dispatch annotations:

```{code-block} python
>>> LengthQuantity = up.PQ["length"]
>>> LengthQuantity
<class 'unxt...ParametricQuantity[PhysicalType('length')]'>
```

The base class `AbstractParametricQuantity` and the concrete `unxt.Quantity` are _not_ parametric — `Quantity[<dimension>]` does nothing and is informational only. Explore [`plum`](https://beartype.github.io/plum/parametric.html) for more on parametric classes.

## Interoperating with the default `Quantity`

Importing `unxts.parametric` registers promotion rules, so `ParametricQuantity` and the default `Quantity` mix freely in arithmetic. Combining the two yields a `ParametricQuantity`, re-parametrized by the **result's** dimension:

```{code-block} python
>>> length = up.PQ["length"](2.0, "m")
>>> plain = u.Q(3.0, "m")

>>> length + plain
ParametricQuantity(Array(5., dtype=float32, ...), unit='m')
```

Because the dimension is tracked in the type, operations that change dimension re-parametrize the result — multiplying two lengths promotes the parameter to area:

```{code-block} python
>>> area = length * up.PQ["length"](5.0, "m")
>>> area
ParametricQuantity(Array(10., dtype=float32, ...), unit='m2')

>>> u.dimension_of(area)
PhysicalType('area')
```
