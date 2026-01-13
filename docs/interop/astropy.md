# 🔭 Astropy

[Astropy][astropy-link] is a widely-used Python library for astronomy. One of
its many widely-used sub-packages is its [`astropy.units`][astropy-units]
library, which provide objects and methods for working with physical quantities,
units, and dimensions.

`unxt` uses Astropy's units and dimensions objects as its backend, while
providing a more function-oriented API and JAX support. Unsurprisingly, `unxt`
has deep support for Astropy objects.

## Dimensions

```{code-block} python

>>> import unxt as u
>>> import astropy.units as apyu

>>> dim = apyu.get_physical_type("length")  # Astropy
>>> dim
PhysicalType('length')

>>> u.dimension("length")  # unxt
PhysicalType('length')

>>> u.dimension(dim)  # unxt <-> Astropy
PhysicalType('length')

>>> u.dimension_of(dim)  # unxt <-> Astropy
PhysicalType('length')

```

## Units

```{code-block} python

>>> import unxt as u
>>> import astropy.units as apyu

>>> meter = apyu.Unit("m")  # Astropy
>>> meter
Unit("m")

>>> u.unit("m")  # unxt
Unit("m")

>>> u.unit(meter)  # unxt <-> Astropy
Unit("m")

>>> u.unit_of(meter)  # unxt <-> Astropy
Unit("m")

```

## Quantities

`unxt` uses Astropy's units and dimensions objects as its backend, but has it's
own `Quantity` class hierarchy.

Converting an [Astropy Quantity][astropy-Quantity] to a [unxt
Quantity][unxt-Quantity] is straightforward -- use `unxt.Quantity.from_`:

```{code-block} python

>>> import unxt as u
>>> import astropy.units as apyu

>>> aq = apyu.Quantity(1, 'm')  # Astropy Quantity
>>> aq
<Quantity 1. m>

>>> xq = u.Quantity.from_(aq)  # unxt Quantity
>>> xq
Quantity(Array(1., dtype=float32), unit='m')
```

Alternatively, the multiple-dispatch library on which `unxt` is built enables
2-way conversion.

```{code-block} python
>>> from plum import convert

>>> convert(aq, u.Quantity)
Quantity(Array(1., dtype=float32), unit='m')

>>> convert(xq, apyu.Quantity)
<Quantity 1. m>

```

## Unit Conversion with Astropy

`unxt` provides full support for unit conversions using Astropy's units. The
low-level `uconvert_value` function works seamlessly with Astropy unit objects,
enabling high-performance conversions suitable for JAX transformations.

### `uconvert_value` with Astropy Units

The `uconvert_value` function accepts Astropy unit objects and performs efficient
numerical conversions:

```{code-block} python

>>> import unxt as u
>>> import astropy.units as apyu
>>> import numpy as np

>>> # Scalar conversion
>>> u.uconvert_value(apyu.Unit("km"), apyu.Unit("m"), 1000)
1.0

>>> # Array conversion
>>> u.uconvert_value(apyu.Unit("km"), apyu.Unit("m"), np.array([1000, 2000, 5000]))
array([1., 2., 5.])

>>> # No conversion when units are identical (hot-path optimization)
>>> u.uconvert_value(apyu.Unit("m"), apyu.Unit("m"), 1000)
1000

```

### Complex Unit Conversions

Support for composite units and unit equivalencies:

```{code-block} python

>>> # Velocity units
>>> u.uconvert_value(apyu.Unit("m/s"), apyu.Unit("km/s"), 1)
1000.0

>>> # With equivalencies (e.g., temperature)
>>> import astropy.units as apyu
>>> with apyu.add_enabled_equivalencies(apyu.temperature()):
...     u.uconvert_value(apyu.Unit("deg_C"), apyu.Unit("K"), 273.15)
0.0

```

### Performance Considerations

`uconvert_value` with Astropy units is optimized for performance:
- **Hot-path**: When units are identical, the value is returned unchanged
- **JAX compatible**: Works with JIT compilation, vmap, and autodiff
- **Low overhead**: Direct Astropy unit conversion without Quantity wrapping

```{code-block} python

>>> import jax
>>> import astropy.units as apyu

>>> @jax.jit
... def convert_to_km(values_in_m):
...     return u.uconvert_value(apyu.Unit("km"), apyu.Unit("m"), values_in_m)

>>> convert_to_km(np.array([1000., 5000., 10000.]))
Array([ 1.,  5., 10.], dtype=float32)

```

<!-- Links -->

[astropy-link]: https://www.astropy.org/
[astropy-units]: https://docs.astropy.org/en/stable/units/index.html
[astropy-Quantity]:
  https://docs.astropy.org/en/stable/api/astropy.units.Quantity.html
[unxt-Quantity]:
  https://unxt.readthedocs.io/en/latest/api/quantities/#unxt.quantity.AbstractQuantity
