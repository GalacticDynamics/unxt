# Quantity

`unxt.quantity` provides the quantity classes: a value paired with a unit, registered as a JAX pytree so it flows through `jit`, `vmap` and `grad`.

| Class | Alias | Value type | Purpose |
| --- | --- | --- | --- |
| `AbstractQuantity` | — | — | Common base of every quantity class. |
| `Quantity` | `u.Q` | `jax.Array` | The default. Non-parametric: one class for every dimension. |
| `Angle` | — | `jax.Array` | Quantity constrained to angular units. |
| `StaticQuantity` | — | `numpy.ndarray` | Hashable quantity for `jax.jit` static arguments. |
| `StaticValue` | — | wraps `numpy.ndarray` | Makes a `Quantity`'s _value_ static. |

`ParametricQuantity` — which encodes the dimension in its type — lives in the separate [`unxts.parametric`](../packages/unxts.parametric/index) package. See {doc}`../explanation/why-quantity-is-non-parametric`.

```{code-block} python
>>> import unxt as u
```

## `Quantity`

### Construction

`Quantity(value, unit)`. The value is converted to a `jax.Array` if it is not already one; the unit is converted to a `Unit`.

```{code-block} python
>>> u.Q(5, "m")
Quantity(Array(5, dtype=int32...), unit='m')
```

`Q` is an alias for `Quantity`, and units may be given as strings, parsed by {func}`unxt.unit`.

### `Quantity.from_`

A multiple-dispatch constructor accepting a wider range of inputs than `__init__`.

```{code-block} python
>>> q = u.Q([1, 2, 3, 5], "m")

>>> u.Q.from_(5, "m")  # same as Quantity(5, "m")
Quantity(Array(5, dtype=int32...), unit='m')

>>> u.Q.from_({"value": [1, 2, 3], "unit": "m"})
Quantity(Array([1, 2, 3], dtype=int32), unit='m')

>>> u.Q.from_(q)  # from another Quantity object
Quantity(Array([1, 2, 3, 5], dtype=int32), unit='m')

>>> u.Q.from_(5, "m", dtype=float)  # specify the dtype
Quantity(Array(5., dtype=float32), unit='m')

```

The registered signatures can be listed at runtime:

<!-- skip: next -->

```{code-block} python
>>> u.Q.from_.methods
List of 9 method(s):
    [0] from_(cls: type, value: typing.Union[ArrayLike, ...], unit: typing.Any, *,
    dtype) -> unxt...quantity...AbstractQuantity
        <function AbstractQuantity.from_ at ...>
    ...
```

`from_` is also the conversion entry point for foreign quantity types — see {doc}`../how-to/interoperate-with-astropy`.

### Attributes

| Attribute | Type        | Description                      |
| --------- | ----------- | -------------------------------- |
| `value`   | `jax.Array` | The numerical value, in `unit`.  |
| `unit`    | `Unit`      | The unit. A static pytree field. |

```{code-block} python
>>> q.value
Array([1, 2, 3, 5], dtype=int32)

>>> q.unit
Unit("m")

```

### Subscripting

`Quantity[<dimension>]` returns `Quantity` unchanged. The subscript is informational: the default class carries dtype and shape in its type, but **not** dimension, so it performs no check and cannot be used for dimension-specific dispatch. Use `unxts.parametric.PQ[<dimension>]` if you need either.

```{code-block} python
>>> u.Q["length"] is u.Quantity
True
```

### Unit conversion

| Method | Function | Returns |
| --- | --- | --- |
| `q.uconvert(unit)` | `u.uconvert(unit, q)` | `Quantity` in `unit` |
| `q.ustrip(unit)` | `u.ustrip(unit, q)` | bare array in `unit` |
| `q.to(unit)` | — | `Quantity` — `astropy` spelling of `uconvert` |
| `q.to_value(unit)` | — | bare array — `astropy` spelling of `ustrip` |
| — | `u.uconvert_value(to, from_, value)` | bare value, no `Quantity` involved |

```{code-block} python
>>> u.Q(5, "m").uconvert("cm")
Quantity(Array(500., dtype=float32, ...), unit='cm')
```

See {doc}`../how-to/convert-units`.

### Arithmetic

Standard operators apply, propagating units through the unit algebra.

```{code-block} python
>>> q1 = u.Q(5, "m")
>>> q2 = u.Q(10, "m")

>>> q1 + q2
Quantity(Array(15, dtype=int32...), unit='m')

>>> q1 * 1.5
Quantity(Array(7.5, dtype=float32, ...), unit='m')

>>> q1 / q2
Quantity(Array(0.5, dtype=float32...), unit='')

>>> q1 ** 2
Quantity(Array(25, dtype=int32...), unit='m2')

```

Operations between incompatible dimensions raise:

```{code-block} python
>>> try: q1 + u.Q(5.0, "second")
... except Exception as e: print(e)
's' (time) and 'm' (length) are not convertible
```

### Comparison

Comparison operators convert before comparing and return a dimensionless `Quantity` of booleans.

```{code-block} python
>>> qa = u.Q([1., 2, 3], "m")
>>> qb = u.Q([100., 201, 300], "cm")

>>> qa < qb
Quantity(Array([False,  True, False], dtype=bool), unit='')

>>> qa == qb
Quantity(Array([ True, False,  True], dtype=bool), unit='')

```

`==` behaves differently for `StaticValue`-backed quantities, and {func}`unxt.equivalent` is the unit-aware alternative — see {doc}`../explanation/equality-and-equivalence`.

### Indexing and updates

Quantities mirror `jax.Array` and the [Array API](https://data-apis.org/array-api/latest/). Indexing returns a `Quantity`; updates are functional, through `.at[]`.

```{code-block} python
>>> qi = u.Q([1, 2, 3, 4], "m")

>>> qi[1]
Quantity(Array(2, dtype=int32), unit='m')

>>> qi[1:]
Quantity(Array([2, 3, 4], dtype=int32), unit='m')

```

```{code-block} python
>>> u.Q([1., 2, 3, 4], "m").at[2].set(u.Q(30.1, "cm"))
Quantity(Array([1.   , 2.   , 0.301, 4.   ], dtype=float32), unit='m')

```

:::{note}

If a `jax.Array` method or property you expect is missing, please open an issue on the [GitHub repository](https://github.com/GalacticDynamics/unxt).

:::

### Display

`repr()` and `str()` are produced by [`wadler_lindig`](https://docs.kidger.site/wadler_lindig) and are governed by {doc}`configuration`; see {doc}`../how-to/control-display`.

## `Angle`

{class}`~unxt.quantity.Angle` is a quantity constrained to angular units.

```{code-block} python
>>> a = u.Angle(45, "deg")
>>> a
Angle(Array(45, dtype=int32...), unit='deg')
```

It supports `from_` and the full arithmetic surface:

```{code-block} python
>>> u.Angle.from_([45, 90], "deg")
Angle(Array([45, 90], dtype=int32), unit='deg')

>>> a + u.Angle(30, "deg")
Angle(Array(75, dtype=int32...), unit='deg')

>>> a.to("rad")
Angle(Array(0.7853982, dtype=float32, weak_type=True), unit='rad')
```

**Enforced dimensionality.** Unlike `Quantity`, a non-angular unit raises at construction:

```{code-block} python
>>> try: u.Angle(1, "m")
... except ValueError as e: print(e)
Angle must have units with angular dimensions.
```

**Wrapping.** `wrap_to(lower, upper)` maps the value into a half-open range, keeping angles on a chosen branch cut. It has a function counterpart `unxt.quantity.wrap_to`.

```{code-block} python
>>> u.Angle(370, "deg").wrap_to(u.Q(0, "deg"), u.Q(360, "deg"))
Angle(Array(10, dtype=int32...), unit='deg')

>>> u.quantity.wrap_to(u.Angle(370, "deg"), u.Q(0, "deg"), u.Q(360, "deg"))
Angle(Array(10, dtype=int32...), unit='deg')
```

Trigonometric and product operations on an `Angle` return a plain `Quantity`.

## `StaticQuantity`

A non-parametric quantity whose value is stored as a static, hashable NumPy array — which is what lets it be a `jax.jit` static argument.

It accepts Python scalars and anything array-like that NumPy can materialise, **including a concrete JAX array**, which is converted back to NumPy. Only a _traced_ value is rejected, since a tracer cannot be static:

```{code-block} python
>>> import jax.numpy as jnp
>>> u.StaticQuantity(jnp.array([1.0, 2.0]), "m")
StaticQuantity(array([1., 2.], dtype=float32), unit='m')
```

```{code-block} python
>>> import numpy as np
>>> import jax
>>> import jax.numpy as jnp
>>> from functools import partial

>>> sq = u.quantity.StaticQuantity(np.array([1.0, 2.0]), "m")  # also u.StaticQuantity
>>> jq = u.Q(jnp.array([1.0, 1.0]), "m")

>>> @partial(jax.jit, static_argnames=("sq",))
... def add(jq, sq):
...     return jq + u.Q(jnp.asarray(sq.value), sq.unit)

>>> add(jq, sq)
Quantity(Array([2., 3.], dtype=float32), unit='m')
```

Prefer `StaticQuantity` when the entire quantity is static.

## `StaticValue`

Wraps a NumPy array so that it can be the _value_ of an ordinary `Quantity`, keeping the `Quantity` type while making the value static. Arithmetic behaves like the wrapped array, and `StaticValue + StaticValue` returns a `StaticValue`.

```{code-block} python
>>> sv = u.quantity.StaticValue(np.array([1.0, 2.0]))
>>> q_static = u.Q(sv, "m")

>>> q_static + u.Q(jnp.array([3.0, 4.0]), "m")
Quantity(Array([4., 6.], dtype=float32), unit='m')
```

Because `==` on such a quantity returns a scalar `bool` and `StaticValue` is hashable, the whole `Quantity` is hashable and can be a `jax.jit` compile-time constant:

```{code-block} python
from functools import partial
import jax
import jax.numpy as jnp

@partial(jax.jit, static_argnames=("scale",))
def rescale(x, *, scale):
    return x * jnp.asarray(scale.value)

scale = u.Q(u.quantity.StaticValue(np.array([2.0, 3.0])), "m")
rescale(jnp.ones(2), scale=scale)   # compiles once
rescale(jnp.ones(2), scale=scale)   # cache hit — no recompilation

new_scale = u.Q(u.quantity.StaticValue(np.array([5.0, 7.0])), "m")
rescale(jnp.ones(2), scale=new_scale)  # different value → recompiles
```

Use `Quantity(StaticValue, ...)` when you need the dynamic/static distinction at the _value_ level while keeping the `Quantity` type; use `StaticQuantity` when the whole quantity is static. The equality semantics of both are covered in {doc}`../explanation/equality-and-equivalence`.

## `is_unit_convertible`

`is_unit_convertible(to_unit, from_, /)` reports whether a conversion is possible, without attempting it. Use it to branch rather than to catch an exception.

```{code-block} python
>>> u.is_unit_convertible("km", "m")
True

>>> u.is_unit_convertible("s", "m")
False
```

The second argument may be anything with a unit, not just a unit — a `Quantity` works:

```{code-block} python
>>> u.is_unit_convertible("km", u.Q(1.0, "m"))
True
```

## `register_ufunc`

NumPy ufuncs reach quantities through `__array_ufunc__`. The **built-in** ufuncs already work — they delegate to the matching `quaxed.numpy` function, which propagates units:

```{code-block} python
>>> import numpy as np
>>> np.sqrt(u.Q(4.0, "m2"))
Quantity(Array(2., dtype=float32...), unit='m')
```

A **custom** ufunc — one you built with `numpy.frompyfunc`, `numba`, or a third-party library — carries no unit semantics, so `unxt` cannot guess one. Calling it on a quantity raises rather than silently dropping the unit:

```{code-block} python
>>> doubler = np.frompyfunc(lambda x: 2 * x, 1, 1)

>>> try:
...     doubler(u.Q(3.0, "m"))
... except TypeError:
...     print("no handler registered")
no handler registered
```

`register_ufunc(ufunc)` supplies the missing rule. The decorated handler is called as `handler(ufunc, method, *inputs, **kwargs)` and must return a unit-carrying result:

```{code-block} python
>>> @u.quantity.register_ufunc(doubler)
... def _(ufunc, method, x, /, **kw):
...     return u.Q(2 * x.value, x.unit)

>>> doubler(u.Q(3.0, "m"))
Quantity(Array(6., dtype=float32...), unit='m')
```

The registry is keyed on the ufunc _object_, not its name, so a custom ufunc that happens to share a name with a built-in still requires its own handler. Handlers may themselves be `plum`-dispatched on the input types.

Registration is global and permanent for the process.

## `AllowValue`

A flag accepted by `ustrip` that permits a bare, unitless array to pass through unchanged, taken to already be in the requested units.

```{code-block} python
>>> u.ustrip(u.quantity.AllowValue, "cm", 500)
500
```

## See also

- {doc}`api/quantity` — the generated API documentation.
- {doc}`../how-to/convert-units`, {doc}`../how-to/use-jax-functions`
- {doc}`../explanation/equality-and-equivalence`, {doc}`../explanation/why-quantity-is-non-parametric`
