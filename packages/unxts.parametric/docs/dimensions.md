# Dimensions

How `unxt.dimension_of` interacts with parametric quantities. For dimensions in general, see the [unxt Dimensions guide](../../guides/dimensions).

```{code-block} python
>>> import unxt as u
>>> import unxts.parametric as up
```

## The dimension of a parametric class

Because a `ParametricQuantity` encodes its dimension in the _class itself_, `unxt.dimension_of` works on a parametrized class — not only on instances:

```{code-block} python
>>> u.dimension_of(up.PQ["length"])  # a parameterized class carries a dimension
PhysicalType('length')

>>> try: u.dimension_of(up.PQ)  # ... the unparameterized class does not
... except Exception as e: print(e)
can only get dimensions from parametrized ParametricQuantity -- ParametricQuantity[dim].
```

The default `Quantity` carries no dimension, so `dimension_of` on the _class_ raises (only instances have a unit, and hence a dimension) — see the [unxt Dimensions guide](../../guides/dimensions).
