# How to compare quantities

Three different questions get asked of two quantities, and they have three different answers. Pick by what you actually want to know.

| Question                                  | Use                   |
| ----------------------------------------- | --------------------- |
| Could I convert one to the other's units? | `is_unit_convertible` |
| Are they the same physical amount?        | `equivalent`          |
| Are they the same object-shaped thing?    | `==`                  |

```{code-block} python
>>> import unxt as u

```

## Ask whether a conversion is possible

`is_unit_convertible(to_unit, from_)` answers without attempting the conversion, so you can branch instead of catching an exception:

```{code-block} python
>>> u.is_unit_convertible("km", "m")
True

>>> u.is_unit_convertible("s", "m")
False

```

The second argument can be anything carrying a unit, so you can ask about a quantity directly:

```{code-block} python
>>> u.is_unit_convertible("km", u.Q(1.0, "m"))
True

```

Reach for this when writing library code that must accept a caller's units and fail with your own error message rather than an `astropy` one.

## Ask whether two quantities are the same amount

`equivalent` converts before comparing, so differently-labelled quantities that describe the same amount come back true:

```{code-block} python
>>> u.equivalent(u.Q(1000.0, "m"), u.Q(1.0, "km"))
Quantity(Array(True, dtype=bool...), unit='')

```

There is a method form if you prefer it:

```{code-block} python
>>> u.Q(1000.0, "m").is_equivalent(u.Q(1.0, "km"))
Quantity(Array(True, dtype=bool...), unit='')

```

It works element-wise on arrays:

```{code-block} python
>>> u.equivalent(u.Q([1.0, 2.0], "m"), u.Q([0.001, 0.009], "km"))
Quantity(Array([ True, False], dtype=bool), unit='')

```

and — unlike conversion — it **returns `False` rather than raising** when the dimensions cannot be reconciled at all, because "are these the same amount?" has a perfectly good answer for a length and a time:

```{code-block} python
>>> u.equivalent(u.Q(1.0, "m"), u.Q(1.0, "s"))
False

```

That makes it safe to call on user input without guarding it first.

## Compare unit systems

The same function reports whether two unit systems span the same dimensions — regardless of which units they picked:

```{code-block} python
>>> u.equivalent(u.unitsystem("m", "kg", "s"), u.unitsystem("km", "g", "hr"))
True

```

Different dimensions, not just different units, make them inequivalent:

```{code-block} python
>>> u.equivalent(u.unitsystem("m", "kg", "s"), u.unitsystem("m", "kg", "s", "rad"))
False

```

## When `==` is the one you want

For ordinary array-backed quantities `==` also converts, and returns an element-wise result:

```{code-block} python
>>> u.Q(1000.0, "m") == u.Q(1.0, "km")
Quantity(Array(True, dtype=bool...), unit='')

```

The case where `==` and `equivalent` diverge is a quantity backed by `StaticValue`, where `==` is deliberately unit-blind so that it can serve as a `jax.jit` cache key. If you are comparing those, you almost certainly want `equivalent` — see {doc}`use-a-quantity-as-a-static-argument` and {doc}`../explanation/equality-and-equivalence`.

## See also

- {doc}`../explanation/equality-and-equivalence` — why the two differ, and what that has to do with `jax.jit`.
- {doc}`convert-units` — doing the conversion once you know it is possible.
- {doc}`../reference/quantity` — `is_unit_convertible` and the comparison operators.
