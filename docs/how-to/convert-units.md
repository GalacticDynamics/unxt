# How to convert between units

`unxt` offers three levels of conversion, and which one you want depends on what you are holding and what you want back.

| You have     | You want back               | Use              |
| ------------ | --------------------------- | ---------------- |
| a `Quantity` | a `Quantity` in new units   | `uconvert`       |
| a `Quantity` | the bare array in new units | `ustrip`         |
| a raw value  | a raw value in new units    | `uconvert_value` |

Each has both a function form and, where it applies, a method form. See {doc}`../explanation/api-conventions` for why both exist.

```{code-block} python
>>> import unxt as u
>>> q = u.Q(5, "m")
```

## Convert a quantity

To keep the units attached, use `uconvert`. If you prefer an object-oriented style, call the method:

```{code-block} python
>>> q.uconvert("cm")
Quantity(Array(500., dtype=float32, ...), unit='cm')
```

To write it functionally — operator first, operand last — call the function:

```{code-block} python
>>> u.uconvert("cm", q)
Quantity(Array(500., dtype=float32, ...), unit='cm')
```

If you are porting code from `astropy`, the `.to` method is available and does the same thing:

```{code-block} python
>>> q.to("cm")
Quantity(Array(500., dtype=float32, ...), unit='cm')
```

Converting between incompatible dimensions raises:

```{code-block} python
>>> try: q.uconvert("s")
... except Exception as e: print(e)
'm' (length) and 's' (time) are not convertible
```

## Strip the units off

To drop out of `unxt` and back into plain JAX — at an I/O boundary, or before handing an array to a library that does not understand quantities — use `ustrip`. It converts _and_ unwraps in one step, so the unit you name is the unit the numbers are in:

```{code-block} python
>>> u.ustrip("cm", q)
Array(500., dtype=float32, ...)
```

or as a method:

```{code-block} python
>>> q.ustrip("cm")
Array(500., dtype=float32, ...)
```

The `astropy` spelling `.to_value` also works:

```{code-block} python
>>> q.to_value("cm")
Array(500., dtype=float32, ...)
```

`ustrip` is also how you hand a quantity to code that does not understand units at all. A dimensionful `Quantity` refuses to become a bare array on its own — `np.asarray(q)` raises rather than guess which unit you meant — so name the unit and the refusal goes away:

```{code-block} python
>>> import numpy as np
>>> np.asarray(u.ustrip("cm", q))
array(500., dtype=float32)
```

See {doc}`../explanation/sharp-bits` for why it refuses.

If your input may be _either_ a quantity or a bare array — a common signature in library code — pass the `AllowValue` flag. A bare array is then taken to be already in the output units and passed through untouched, instead of raising:

```{code-block} python
>>> u.ustrip(u.quantity.AllowValue, "cm", 500)
500
```

## Convert raw values

When you never wanted the wrapper in the first place — inside a `jit`ted kernel, or converting a large batch — `uconvert_value` takes _from_ and _to_ units explicitly and works on raw numbers:

```{code-block} python
>>> u.uconvert_value("km", "m", 1000)
1.0

>>> import jax.numpy as jnp
>>> u.uconvert_value("km", "m", jnp.array([1000, 2000, 5000]))
Array([1., 2., 5.], dtype=float32, ...)
```

Unit objects work in place of strings:

```{code-block} python
>>> u.uconvert_value(u.unit("km"), u.unit("m"), 5000)
5.0
```

Reach for it when you are performing batch conversions, working inside `jit`, `vmap` or `grad`, or want to avoid constructing `Quantity` objects on a hot path. `uconvert` delegates to it internally for the numerical step.

For convenience the same function also accepts a `Quantity`, in which case it just calls `uconvert` — so you can migrate a call site to the lower-level function without first rewriting its inputs:

```{code-block} python
>>> u.uconvert_value("km", "m", q)
Quantity(Array(0.005, dtype=float32, ...), unit='km')
```

Inside a jitted function it composes as you would expect:

```{code-block} python
>>> import jax
>>> @jax.jit
... def batch_convert_to_km(values_in_m):
...     return u.uconvert_value("km", "m", values_in_m)

>>> batch_convert_to_km(jnp.array([1000., 5000., 10000.]))
Array([ 1.,  5., 10.], dtype=float32)
```

## See also

- {doc}`../reference/quantity` — the full `Quantity` surface.
- {doc}`interoperate-with-astropy` — converting with `astropy` units and quantities.
- {doc}`optimize-performance` — where conversion overhead actually costs you.
