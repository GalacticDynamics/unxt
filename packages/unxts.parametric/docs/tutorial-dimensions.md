# Let the type system catch a unit mistake

In this tutorial we will write a function that can only be called with the right kind of quantity, and watch `unxts.parametric` reject a wrong one before the function body ever runs. Along the way you will see a dimension get inferred, get checked, and get carried through arithmetic.

You need `unxt` and `unxts.parametric` installed and nothing else.

## Set up

```{code-block} python
>>> import unxt as u
>>> import unxts.parametric as up

```

## See what the default `Quantity` does not do

Start with the plain `unxt.Quantity`. Subscript it with a dimension and give it a unit that contradicts the subscript — seconds where we asked for a length:

```{code-block} python
>>> u.Q["length"](1.0, "s")
Quantity(Array(1., dtype=float32, weak_type=True), unit='s')

```

Nothing happened. No error, no warning; we got a quantity of seconds back. The subscript on the default `Quantity` is accepted for compatibility and then ignored — it carries no dimension in its type, so it has nothing to check against.

That is the gap `unxts.parametric` fills.

## Watch the same mistake get caught

Now make the same call with `up.PQ`. We expect this one to fail, so let's catch the error and print it:

```{code-block} python
>>> try:
...     up.PQ["length"](1.0, "s")
... except ValueError as e:
...     print(e)
Physical type mismatch.

```

There it is. `ParametricQuantity["length"]` is a genuine class that knows its dimension, so it compared the unit we passed against that dimension and refused. The wrong quantity never got built.

Give it a length and it is happy:

```{code-block} python
>>> up.PQ["length"](1.0, "m")
ParametricQuantity(Array(1., dtype=float32, weak_type=True), unit='m')

```

## Let the dimension be inferred

You do not have to write the subscript. Construct without one and the dimension is worked out from the unit:

```{code-block} python
>>> g = up.PQ(9.8, "m/s2")
>>> u.dimension_of(g)
PhysicalType('acceleration')

```

We never said "acceleration" anywhere. Look at what `g` actually is:

```{code-block} python
>>> type(g)
<class 'unxt...ParametricQuantity[PhysicalType('acceleration')]'>

```

The class itself is parametrized. That is the difference from the default `Quantity`, where every dimension shares one class — here, the dimension is part of the type, which is what made the check in the previous step possible.

A parametrized class carries its dimension even without an instance:

```{code-block} python
>>> u.dimension_of(up.PQ["length"])
PhysicalType('length')

```

## Multiply two lengths and watch the type change

Arithmetic re-parametrizes the result. Multiply two lengths:

```{code-block} python
>>> area = up.PQ(3.0, "m") * up.PQ(4.0, "m")
>>> area
ParametricQuantity(Array(12., dtype=float32, weak_type=True), unit='m2')

>>> u.dimension_of(area)
PhysicalType('area')

```

Notice that the result is not a length. Its type changed along with its unit:

```{code-block} python
>>> type(area) is up.PQ["area"]
True

```

The dimension was recomputed from the arithmetic, and the class followed.

## Write a function that dispatches on dimension

Now the payoff. Because `up.PQ["length"]` and `up.PQ["time"]` are real, distinct types, you can write two functions with the same name and let the argument's dimension pick between them:

```{code-block} python
>>> from plum import dispatch

>>> @dispatch
... def describe(q: up.PQ["length"]) -> str:
...     return "a length"

>>> @dispatch
... def describe(q: up.PQ["time"]) -> str:
...     return "a duration"

```

Call it with a length:

```{code-block} python
>>> describe(up.PQ(5.0, "m"))
'a length'

```

and with a time:

```{code-block} python
>>> describe(up.PQ(5.0, "s"))
'a duration'

```

Same call, different implementation, chosen by the physical dimension of the argument. We never wrote an `if`.

## Mix with plain quantities

Parametric and plain quantities combine freely — the result is parametric, carrying the dimension:

```{code-block} python
>>> up.PQ(2.0, "m") + u.Q(3.0, "m")
ParametricQuantity(Array(5., dtype=float32, weak_type=True), unit='m')

```

## What we built

You have a `describe` function that selects its implementation from the physical dimension of its argument, and you have seen a wrong unit rejected at construction rather than producing a plausible wrong answer later. Both come from the same thing: the dimension living in the type.

That capability is not free — a class and a JAX pytree type per dimension — which is why it is opt-in rather than the default.

## Where to go next

- [`ParametricQuantity`](./quantity) — the full construction and dispatch surface.
- [How to check dimensions at runtime](./type-checking) — enforcing dimensions on function signatures with `jaxtyping`.
- [Why `Quantity` is not parametric](../../explanation/why-quantity-is-non-parametric) — what the per-dimension type actually costs.
