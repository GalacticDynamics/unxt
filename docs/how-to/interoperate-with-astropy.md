# How to interoperate with astropy

[astropy-link]: https://www.astropy.org/
[astropy-units]: https://docs.astropy.org/en/stable/units/index.html

`unxt` uses [`astropy.units`][astropy-units] as its unit and dimension backend, so astropy's `Unit` and `PhysicalType` objects _are_ `unxt`'s. Quantities are the exception: `unxt` has its own class hierarchy, and mixing the two under `jax.jit` loses units silently. This guide covers both halves.

```{code-block} python
>>> import unxt as u
>>> import astropy.units as apyu
```

## Pass astropy units and dimensions straight through

No conversion step is needed. `unxt.unit`, `unxt.unit_of`, `unxt.dimension` and `unxt.dimension_of` all accept astropy objects and return astropy objects:

```{code-block} python
>>> dim = apyu.get_physical_type("length")
>>> u.dimension(dim)
PhysicalType('length')

>>> u.dimension_of(dim)
PhysicalType('length')

>>> meter = apyu.Unit("m")
>>> u.unit(meter)
Unit("m")

>>> u.unit_of(meter)
Unit("m")

```

## Convert an astropy quantity

To bring an `astropy.units.Quantity` into `unxt`, use `Quantity.from_`:

```{code-block} python
>>> aq = apyu.Quantity(1, 'm')
>>> xq = u.Q.from_(aq)
>>> xq
Quantity(Array(1., dtype=float32), unit='m')
```

To go either direction — including back out to astropy — use `plum.convert`:

```{code-block} python
>>> from plum import convert

>>> convert(aq, u.Q)
Quantity(Array(1., dtype=float32), unit='m')

>>> convert(xq, apyu.Quantity)
<Quantity 1. m>

```

## Convert at the boundary, always

:::{warning}

Arithmetic mixing an astropy quantity with an `unxt` quantity works **eagerly** but **fails silently under `jax.jit`**.

:::

Eagerly, astropy's `__array_ufunc__` handles the operation and both units survive:

```{code-block} python
>>> import jax

>>> apy = apyu.Quantity(2.0, "km")
>>> q = u.Q(3.0, "m")

>>> apy * q
<Quantity 6. km m>
```

Under `jit`, `km` is dropped and the result claims `m` — the magnitude survives but the answer is wrong by the conversion factor, here 1000×:

```{code-block} python
>>> jax.jit(lambda a, b: a * b)(apy, q)
Quantity(Array(6., dtype=float32), unit='m')
```

`+` and `-` at least fail loudly:

```{code-block} python
>>> try:
...     jax.jit(lambda a, b: a + b)(apy, q)
... except apyu.UnitConversionError as e:
...     print(type(e).__name__)
UnitConversionError
```

`unxt` cannot intercept this: `jax` converts the astropy `ndarray` subclass into a unitless tracer before any `unxt` code runs, so the unit is gone by then. Capturing the astropy quantity as a closure constant instead of passing it as an argument loses it too, by a different route.

The fix is to convert _before_ anything crosses into a jitted function:

```{code-block} python
>>> qa = u.Q.from_(apy)  # 2.0 km, now a unxt Quantity
>>> qa
Quantity(Array(2., dtype=float32), unit='km')

>>> jax.jit(lambda a, b: a * b)(qa, q)
Quantity(Array(6., dtype=float32), unit='km m')

>>> jax.jit(lambda a, b: a + b)(qa, q)
Quantity(Array(2.003, dtype=float32), unit='km')
```

As a rule: keep astropy quantities at the edges of your program and convert once on the way in. Mixed-library arithmetic working eagerly is not evidence that it will work under `jit`.

## Convert raw values with astropy units

`uconvert_value` accepts astropy `Unit` objects wherever it accepts strings, which is useful when the units are already astropy objects from somewhere else:

```{code-block} python
>>> import numpy as np

>>> u.uconvert_value(apyu.Unit("km"), apyu.Unit("m"), 1000)
1.0

>>> u.uconvert_value(apyu.Unit("km"), apyu.Unit("m"), np.array([1000, 2000, 5000]))
array([1., 2., 5.])

>>> u.uconvert_value(apyu.Unit("m/s"), apyu.Unit("km/s"), 1)
1000.0

```

When the two units are identical the value is returned unchanged, without going through astropy's conversion machinery:

```{code-block} python
>>> u.uconvert_value(apyu.Unit("m"), apyu.Unit("m"), 1000)
1000

```

To use astropy's equivalencies, enable them around the call as you normally would:

```{code-block} python
>>> with apyu.add_enabled_equivalencies(apyu.temperature()):
...     u.uconvert_value(apyu.Unit("deg_C"), apyu.Unit("K"), 273.15)
0.0

```

The function is JAX-compatible, so it composes with `jit`, `vmap` and `grad`:

```{code-block} python
>>> @jax.jit
... def convert_to_km(values_in_m):
...     return u.uconvert_value(apyu.Unit("km"), apyu.Unit("m"), values_in_m)

>>> convert_to_km(np.array([1000., 5000., 10000.]))
Array([ 1.,  5., 10.], dtype=float32)

```

## See also

- {doc}`convert-units` — conversion within `unxt`.
- {doc}`../explanation/sharp-bits` — other places JAX and units interact badly.
- [Astropy][astropy-link] and [`astropy.units`][astropy-units].
