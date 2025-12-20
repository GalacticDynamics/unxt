# Quantity

## Creating Quantity Instances

`Quantity` objects are created by passing a value and a unit to the `Quantity`
constructor (with `Q` as a shorthand).

```{code-block} python
>>> import unxt as u

>>> u.Quantity(5, "m")
Quantity(Array(5, dtype=int32, weak_type=True), unit='m')
```

The constructor will automatically
[convert](https://beartype.github.io/plum/conversion_promotion.html#conversion-with-convert)
the value to a `jax.Array` (if it is not already one) and convert the unit to a
`Unit` object.

The value and unit of a `Quantity` object can be accessed using the `value` and
`unit` attributes, respectively:

```{code-block} python
>>> q = u.Q([1, 2, 3, 5], "m")
>>> q.value
Array([1, 2, 3, 5], dtype=int32)

>>> q.unit
Unit("m")

```

If you want more flexible options to create a `Quantity`, you can use the
`Quantity.from_` class method. This uses multiple dispatch to determine the
appropriate constructor based on the input arguments.

```{code-block} python
>>> u.Q.from_(5, "m")  # same as Quantity(5, "m")
Quantity(Array(5, dtype=int32, ...), unit='m')

>>> u.Q.from_({"value": [1, 2, 3], "unit": "m"})
Quantity(Array([1, 2, 3], dtype=int32), unit='m')

>>> u.Q.from_(q)  # from another Quantity object
Quantity(Array([1, 2, 3, 5], dtype=int32), unit='m')

>>> u.Q.from_(5, "m", dtype=float)  # specify the dtype
Quantity(Array(5., dtype=float32), unit='m')

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
>>> q = u.Q(5, "m")
>>> q.uconvert("cm")
Quantity(Array(500., dtype=float32, ...), unit='cm')
```

:::{note}
:class: dropdown

The Astropy API `.to` is also available for `Quantity` objects.

```{code-block} python
>>> q.to("cm")
Quantity(Array(500., dtype=float32, ...), unit='cm')
```

:::

If you prefer a more functional approach, use the `uconvert` function.

```{code-block} python
>>> u.uconvert("cm", q)
Quantity(Array(500., dtype=float32, ...), unit='cm')
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

:::{note}
:class: dropdown

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
>>> q1 = u.Q(5, "m")
>>> q2 = u.Q(10, "m")

>>> q1 + q2
Quantity(Array(15, dtype=int32, ...), unit='m')

>>> q1 * 1.5
Quantity(Array(7.5, dtype=float32, ...), unit='m')

>>> q1 / q2
Quantity(Array(0.5, dtype=float32, ...), unit='')

>>> q1 ** 2
Quantity(Array(25, dtype=int32, ...), unit='m2')

```

### Comparison Operations

```{code-block} python
>>> q1 = u.Q([1., 2, 3], "m")
>>> q2 = u.Q([100., 201, 300], "cm")

>>> q1 < q2
Array([False,  True, False], dtype=bool)

>>> q1 == q2
Array([ True, False,  True], dtype=bool)

```

### Indexing and Slicing

```{code-block} python
>>> q = u.Q([1, 2, 3, 4], "m")

>>> q[1]
Quantity(Array(2, dtype=int32), unit='m')

>>> q[1:]
Quantity(Array([2, 3, 4], dtype=int32), unit='m')

```

### Array Updates

`unxt` supports JAX-style array updates. See
[🔪 JAX - The Sharp Bits 🔪](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#array-updates-x-at-idx-set-y)
for more details.

```{code-block} python
>>> q = u.Q([1., 2, 3, 4], "m")

>>> q.at[2].set(u.Q(30.1, "cm"))
Quantity(Array([1.   , 2.   , 0.301, 4.   ], dtype=float32), unit='m')

```

### JAX Functions

JAX function normally only support pure JAX arrays.

```{code-block} python

>>> import jax.numpy as jnp  # regular JAX
>>> x = u.Q([1, 2, 3], "m")

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

>>> y = u.Q([4, 5, 6], "m")
>>> func(x, y)
Quantity(Array([ 5, 14, 27], dtype=int32), unit='m2')

```

[`quaxed`][quaxed] is a convenience library that pre-"quaxify"s JAX functions.
It's a drop-in replacement for much of JAX.

```{code-block} python
:emphasize-lines: 1

>>> import quaxed.numpy as jnp  # pre-quaxified JAX

>>> jnp.square(x) + jnp.multiply(x, y)
Quantity(Array([ 5, 14, 27], dtype=int32), unit='m2')

```

`quaxed` is totally optional. You can
[`quax.quaxify`](https://docs.kidger.site/quax/api/quax/#quax.quaxify) manually,
to only decorate your top-level functions or to call 3rd party functions.

:::{attention}

`Quantity` should support **all** JAX functions. If you find a function that
doesn't work, please open an issue on the
[GitHub repository](https:://github.com/GalacticDynamics/unxt).

:::

## Pretty Printing

`Quantity` objects support the
[`wadler_lindig`](https://docs.kidger.site/wadler_lindig) library for pretty
printing.

```{code-block} python

>>> import wadler_lindig as wl

>>> q = u.Q([1, 2, 3], "m")

>>> wl.pprint(q)  # The default pretty printing
Quantity(i32[3], unit='m')

```

The type parameter can be included in the representation:

```{code-block} python

>>> wl.pprint(q, include_params=True)
Quantity['length'](i32[3], unit='m')

```

The `str` method uses this as well:

```{code-block} python

>>> print(q)
Quantity['length']([1, 2, 3], unit='m')

```

Arrays can be printed in full:

```{code-block} python

>>> wl.pprint(q, short_arrays=False)
Quantity(Array([1, 2, 3], dtype=int32), unit='m')

```

The `repr` method uses this setting:

```{code-block} python

>>> print(repr(q))
Quantity(Array([1, 2, 3], dtype=int32), unit='m')

```

The units can be turned from a named argument to a positional argument by
setting `named_unit=False`:

```{code-block} python

>>> wl.pprint(q, named_unit=False)
Quantity(i32[3], 'm')

```

Instead of printing the value as either a full Array or a short array, you can compactify the value to its compact Array form:

```{code-block} python

>>> wl.pprint(q, short_arrays="compact")
Quantity([1, 2, 3], unit='m')

```

For more compact output, the `Quantity` class has a short name `Q` that can be
used by setting `use_short_name=True`:

```{code-block} python

>>> wl.pprint(q, use_short_name=True)
Q(i32[3], unit='m')

```

The short name can be combined with other printing options:

```{code-block} python

>>> wl.pprint(q, use_short_name=True, include_params=True)
Q['length'](i32[3], unit='m')

>>> wl.pprint(q, use_short_name=True, short_arrays="compact")
Q([1, 2, 3], unit='m')

```

See the [`wadler_lindig` documentation](https://docs.kidger.site/wadler_lindig)
for more details on the pretty printing options.


# Specialized Quantity Objects

## Working with `Angle` Objects

The {class}`~unxt.quantity.Angle` class is a specialized quantity for
representing angular measurements, similar to {class}`~unxt.quantity.Quantity`
but with additional features and constraints tailored for angles.

### Creating Angles

You can create an {class}`~unxt.quantity.Angle` just like a
{class}`~unxt.quantity.Quantity`, by specifying a value and a unit with angular
dimensions:

```{code-block} python
>>> a = u.Angle(45, "deg")
>>> a
Angle(Array(45, dtype=int32, weak_type=True), unit='deg')
```

Just like {class}`~unxt.quantity.Quantity`, you can flexibly create
{class}`~unxt.quantity.Angle` objects using the
{meth}`~unxt.quantity.Angle.from_` constructor:

```{code-block} python
>>> u.Angle.from_(45, "deg")
Angle(Array(45, dtype=int32, weak_type=True), unit='deg')

>>> u.Angle.from_([45, 90], "deg")
Angle(Array([45, 90], dtype=int32), unit='deg')

>>> u.Angle.from_(jnp.array([10, 15, 20]), "deg")
Angle(Array([10, 15, 20], dtype=int32), unit='deg')

```

### Mathematical Operations

{class}`~unxt.quantity.Angle` objects support arithmetic operations,
broadcasting, and most mathematical functions, just like
{class}`~unxt.quantity.Quantity`:

```{code-block} python
>>> b = u.Angle(30, "deg")
>>> a + b
Angle(Array(75, dtype=int32, weak_type=True), unit='deg')
>>> 2 * a
Angle(Array(90, dtype=int32, weak_type=True), unit='deg')
>>> a.to("rad")
Angle(Array(0.7853982, dtype=float32, weak_type=True), unit='rad')
```

For more information on mathematical operations, see the unxt documentation.

### Enforced Dimensionality

Unlike a generic {class}`~unxt.quantity.Quantity`, the
{class}`~unxt.quantity.Angle` class enforces that the unit must be angular
(e.g., degrees, radians). Attempting to use a non-angular unit will raise an
error:

```{code-block} python
>>> try: u.Angle(1, "m")
... except ValueError as e: print(e)
Angle must have units with angular dimensions.
```

### Wrapping Angles

A key feature of {class}`~unxt.quantity.Angle` is the ability to wrap values
to a specified range, which is useful for keeping angles within a branch cut:

```{code-block} python
>>> a = u.Angle(370, "deg")
>>> a.wrap_to(u.Q(0, "deg"), u.Q(360, "deg"))
Angle(Array(10, dtype=int32, weak_type=True), unit='deg')
```

The {meth}`~unxt.quantity.Angle.wrap_to` method has a function counterpart

```{code-block} python
>>> u.quantity.wrap_to(a, u.Q(0, "deg"), u.Q(360, "deg"))
Angle(Array(10, dtype=int32, weak_type=True), unit='deg')
```

---

:::{seealso}

[API Documentation for Quantities](../api/quantity.md)

:::

[quaxed]: https://quaxed.readthedocs.io/en/latest/
