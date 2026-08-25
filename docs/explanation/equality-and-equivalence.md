# Equality and equivalence

`unxt` gives you two comparisons, and they answer different questions. `==` asks whether two quantities are _the same object-shaped thing_; {func}`unxt.equivalent` asks whether they are _the same physical amount_. Most of the time these agree. Where they diverge is worth understanding, because the divergence is deliberate and it is load-bearing for `jax.jit`.

```{code-block} python
>>> import numpy as np
>>> import unxt as u
```

## `==` follows the value

For an ordinary array-backed `Quantity`, `==` behaves the way NumPy trained you to expect: element-wise, unit-aware, and returning an array of booleans wrapped in a dimensionless `Quantity` — wrapped, because the Array API asks that a result share a namespace with its inputs.

```{code-block} python
>>> u.Q([1.0, 2.0], "m") == u.Q([1.0, 9.0], "m")
Quantity(Array([ True, False], dtype=bool), unit='')
```

For a quantity whose value is a {class}`~unxt.quantity.StaticValue`, `==` returns a scalar `bool` instead:

```{code-block} python
>>> sv1 = u.quantity.StaticValue(np.array([1.0, 2.0]))
>>> sv2 = u.quantity.StaticValue(np.array([1.0, 2.0]))
>>> sv3 = u.quantity.StaticValue(np.array([9.0, 9.0]))

>>> u.Q(sv1, "m") == u.Q(sv2, "m")
True
>>> u.Q(sv1, "m") == u.Q(sv3, "m")
False
```

That difference is the whole point of `StaticValue`. A `jax.jit` `static_argnames` argument must be hashable and must compare with a scalar `bool`; an element-wise array cannot serve as a cache key. Making `==` scalar here is what lets a whole `Quantity` be passed as a compile-time constant.

## Why static equality is unit-blind

The scalar comparison goes further: it compares unit _labels_, not physical amounts. `1000 m` and `1 km` are **not** equal.

```{code-block} python
>>> sv_km = u.quantity.StaticValue(np.array([0.001, 0.002]))
>>> u.Q(sv1, "m") == u.Q(sv_km, "km")
False
```

This looks wrong until you follow it through to `jit`. If `==` converted units first, two physically equal but differently-labelled quantities would collapse into one `static_argnames` cache key — while still producing different traced code, because the unit is baked into the trace. It would also break the `__eq__`/`__hash__` contract, since two objects that compare equal must hash equal and these do not. Unit-blindness is what keeps the cache honest.

## `equivalent` answers the physical question

When you want "same physical amount, regardless of how it is labelled", that is {func}`unxt.equivalent`, or the `is_equivalent` method:

```{code-block} python
>>> u.equivalent(u.Q(sv1, "m"), u.Q(sv_km, "km"))
True
>>> u.Q(sv1, "m").is_equivalent(u.Q(sv_km, "km"))
True
```

`equivalent` mirrors `==`'s shape — scalar for `StaticValue`-backed operands, element-wise for array-backed ones:

```{code-block} python
>>> u.equivalent(u.Q([1.0, 2.0], "m"), u.Q([0.001, 0.009], "km"))
Quantity(Array([ True, False], dtype=bool), unit='')
```

and, unlike conversion, it returns `False` rather than raising when the dimensions cannot be reconciled at all — because "are these the same amount?" has a perfectly good answer for a length and a time:

```{code-block} python
>>> u.equivalent(u.Q(1.0, "m"), u.Q(1.0, "s"))
False
```

The same function also compares unit systems, reporting whether two of them span the same dimensions.

## Choosing

Use `==` for structural identity, and as a `jax.jit` `static_argnames` key. Use `equivalent` to ask whether two quantities represent the same physical amount. If you are upgrading from v1 and your `==` comparisons were unit-aware, `equivalent` is the drop-in replacement — see {doc}`../how-to/migrate-to-v2`.

## See also

- {doc}`../reference/quantity` — `StaticValue`, `StaticQuantity`, and the rest.
- {doc}`sharp-bits` — other places `unxt` behaves unlike NumPy.
