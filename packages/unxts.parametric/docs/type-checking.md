# Type Checking

Annotating and checking the physical **dimension** of an argument. For enabling runtime type checking and for dtype/shape annotations on the default `Quantity`, see the [unxt Type Checking guide](../../guides/type-checking).

```{code-block} python
>>> import unxt as u
>>> import unxts.parametric as up
```

## Dimension annotations for type checking

Because the dimension lives in the type, you can annotate function signatures with dimensioned `ParametricQuantity` types, and `unxt`'s runtime type checking (via [`jaxtyping`](https://pypi.org/project/jaxtyping/)) will enforce them. This dimension-level checking is what `ParametricQuantity` adds over the default `Quantity` (whose type carries only dtype and shape):

```{code-block} python
>>> from jaxtyping import Shaped, jaxtyped
>>> from beartype import beartype as typechecker

>>> @jaxtyped(typechecker=typechecker)
... def velocity(
...     x: Shaped[up.PQ["length"], "N"],
...     t: Shaped[up.PQ["time"], "N"],
... ) -> Shaped[up.PQ["speed"], "N"]:
...     return x / t

>>> x = up.PQ([2.], "m")
>>> t = up.PQ([1.], "s")

>>> velocity(x, t)
ParametricQuantity(Array([2.], dtype=float32), unit='m / s')
```

Passing a quantity of the wrong dimension raises at call time (under runtime type checking). The base class `AbstractParametricQuantity` and the concrete `unxt.Quantity` are _not_ parametric — `Quantity[<dimension>]` does nothing and is informational only.
