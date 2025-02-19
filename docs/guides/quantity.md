# Quantity

## Creating Quantity Instances

`Quantity` objects are created by passing a value and a unit to the `Quantity`
constructor.

```{code-block} python
>>> import unxt as u
>>> u.Quantity(5, "m")
Quantity['length'](Array(5, dtype=int32, weak_type=True), unit='m')
```

The constructor will automatically
[convert](https://beartype.github.io/plum/conversion_promotion.html#conversion-with-convert)
the value to a `jax.Array` (if it is not already one) and convert the unit to a
`Unit` object.

The value and unit of a `Quantity` object can be accessed using the `value` and
`unit` attributes, respectively:

```{code-block} python
>>> q = u.Quantity([1, 2, 3, 5], "m")
>>> q.value
Array([1, 2, 3, 5], dtype=int32)

>>> q.unit
Unit("m")

```

If you want more flexible options to create a `Quantity`, you can use the
`Quantity.from_` class method. This uses multiple dispatch to determine the
appropriate constructor based on the input arguments.

```{code-block} python
>>> u.Quantity.from_(5, "m")  # same as Quantity(5, "m")
Quantity['length'](Array(5, dtype=int32, ...), unit='m')

>>> u.Quantity.from_({"value": [1, 2, 3], "unit": "m"})
Quantity['length'](Array([1, 2, 3], dtype=int32), unit='m')

>>> u.Quantity.from_(q)  # from another Quantity object
Quantity['length'](Array([1, 2, 3, 5], dtype=int32), unit='m')

>>> u.Quantity.from_(5, "m", dtype=float)  # specify the dtype
Quantity['length'](Array(5., dtype=float32), unit='m')

```

There are many more options available with `Quantity.from_`. For a complete list
of options run `Quantity.from_.methods` in an IDE.

<!-- skip: next -->

```{code-block} python
>>> u.Quantity.from_.methods
List of 9 method(s):
    [0] from_(cls: type, value: typing.Union[ArrayLike, ...], unit: typing.Any, *,
    dtype) -> unxt...quantity...AbstractQuantity
        <function AbstractQuantity.from_ at ...>
    ...
```

`Quantity.from_` assists with interfacing with other libraries, e.g. see
[Interop with Astropy](../interop/astropy.md).

## Converting to Different Units

`Quantity` objects can be converted to different units and values in those
units. If you prefer an object-oriented approach, use the `uconvert` method.

```{code-block} python
>>> q = u.Quantity(5, "m")
>>> q.uconvert("cm")
Quantity['length'](Array(500., dtype=float32, ...), unit='cm')
```

:::{note} :class: dropdown

The Astropy API `.to` is also available for `Quantity` objects.

```{code-block} python
>>> q.to("cm")
Quantity['length'](Array(500., dtype=float32, ...), unit='cm')
```

:::

If you prefer a more functional approach, use the `uconvert` function.

```{code-block} python
>>> u.uconvert("cm", q)
Quantity['length'](Array(500., dtype=float32, ...), unit='cm')
```

To convert to the value in the new units, use the `ustrip` function.

```{code-block} python
>>> u.ustrip("cm", q)
Array(500., dtype=float32, ...)
```

Alternatively the `ustrip` method can be used.

```{code-block} python
>>> q.ustrip("cm")
Array(500., dtype=float32, ...)
```

When working with either an array or a `Quantity` object, you can use the
`ustrip` function with the `unxt.quantity.AllowValue` flag to allow arrays
without units to be passed in, assuming them to be in the correct output units.

```{code-block} python
>>> import jax.numpy as jnp
>>> u.ustrip(u.quantity.AllowValue, "cm", 500)
500
```

:::{note} :class: dropdown

The Astropy API `.to_value` is also available for `Quantity` objects.

```{code-block} python
>>> q.to_value("cm")
Array(500., dtype=float32, ...)
```

:::

## With reference to `jax.Array`

`Quantity` objects are designed to mirror `jax.Array` and the
[Array API](https://data-apis.org/array-api/latest/).

:::{note}

If you find that a method or property is missing, please open an issue on the
[GitHub repository](https:://github.com/GalacticDynamics/unxt).

:::

This means you can operations on `Quantity` objects just like you would with
`jax.Array`.

### Arithmetic Operations

You can perform standard mathematical operations on `Quantity` objects:

```{code-block} python
>>> q1 = u.Quantity(5, "m")
>>> q2 = u.Quantity(10, "m")

>>> q1 + q2
Quantity['length'](Array(15, dtype=int32, ...), unit='m')

>>> q1 * 1.5
Quantity['length'](Array(7.5, dtype=float32, ...), unit='m')

>>> q1 / q2
Quantity['dimensionless'](Array(0.5, dtype=float32, ...), unit='')

>>> q1 ** 2
Quantity['area'](Array(25, dtype=int32, ...), unit='m2')

```

### Comparison Operations

```{code-block} python
>>> q1 = u.Quantity([1., 2, 3], "m")
>>> q2 = u.Quantity([100., 201, 300], "cm")

>>> q1 < q2
Array([False,  True, False], dtype=bool)

>>> q1 == q2
Array([ True, False,  True], dtype=bool)

```

### Indexing and Slicing

```{code-block} python
>>> q = u.Quantity([1, 2, 3, 4], "m")

>>> q[1]
Quantity['length'](Array(2, dtype=int32), unit='m')

>>> q[1:]
Quantity['length'](Array([2, 3, 4], dtype=int32), unit='m')

```

### Array Updates

`unxt` supports JAX-style array updates. See
[🔪 JAX - The Sharp Bits 🔪](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#array-updates-x-at-idx-set-y)
for more details.

```{code-block} python
>>> q = u.Quantity([1., 2, 3, 4], "m")

>>> q.at[2].set(u.Quantity(30.1, "cm"))
Quantity['length'](Array([1.   , 2.   , 0.301, 4.   ], dtype=float32), unit='m')

```

### JAX Functions

JAX function normally only support pure JAX arrays.

```{code-block} python

>>> import jax.numpy as jnp  # regular JAX
>>> x = u.Quantity([1, 2, 3], "m")

>>> try: jnp.square(x)
... except TypeError: print("not a pure JAX array")
not a pure JAX array

```

We use `quax` to enable Quantity support across most of the JAX ecosystem! See
the [quax docs](https://docs.kidger.site/quax/) for implementation details. The
short version is that you can use `Quantity` in JAX functions so long they pass
through a [`quax.quaxify`](https://docs.kidger.site/quax/api/quax/#quax.quaxify)
call. Here are a few examples:

This is the way to "quaxify" a JAX function. A powerful feature of `quaxify` is
that it enables `Quantity` support through _all_ the JAX functions inside the
top function. With `unxt` you can use normal JAX!

```{code-block} python
:emphasize-lines: 4

>>> import jax.numpy as jnp  # regular JAX
>>> from quax import quaxify

>>> @quaxify  # Now it works with Quantity... that's it!
... def func(x, y):
...     return jnp.square(x) + jnp.multiply(x, y)  # normal JAX

>>> y = u.Quantity([4, 5, 6], "m")
>>> func(x, y)
Quantity['area'](Array([ 5, 14, 27], dtype=int32), unit='m2')

```

[`quaxed`][quaxed] is a convenience library that pre-"quaxify"s JAX functions.
It's a drop-in replacement for much of JAX.

```{code-block} python
:emphasize-lines: 1

>>> import quaxed.numpy as jnp  # pre-quaxified JAX

>>> jnp.square(x) + jnp.multiply(x, y)
Quantity['area'](Array([ 5, 14, 27], dtype=int32), unit='m2')

```

`quaxed` is totally optional. You can
[`quax.quaxify`](https://docs.kidger.site/quax/api/quax/#quax.quaxify) manually,
to only decorate your top-level functions or to call 3rd party functions.

:::{attention}

`Quantity` should support **all** JAX functions. If you find a function that
doesn't work, please open an issue on the
[GitHub repository](https:://github.com/GalacticDynamics/unxt).

:::

:::{seealso}

[API Documentation for Quantities](../api/quantity.md)

:::

[quaxed]: https://quaxed.readthedocs.io/en/latest/
