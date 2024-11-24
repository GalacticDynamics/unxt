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

>>> import astropy.units as u
>>> from unxt import dimension, dimension_of

>>> dim = u.get_physical_type("length")  # Astropy
>>> dim
PhysicalType('length')

>>> dimension("length")  # unxt
PhysicalType('length')

>>> dimension(dim)  # unxt <-> Astropy
PhysicalType('length')

>>> dimension_of(dim)  # unxt <-> Astropy
PhysicalType('length')

```

## Units

```{code-block} python

>>> import astropy.units as u
>>> import unxt as ux

>>> meter = u.Unit("m")  # Astropy
>>> meter
Unit("m")

>>> ux.unit("m")  # unxt
Unit("m")

>>> ux.unit(meter)  # unxt <-> Astropy
Unit("m")

>>> ux.unit_of(meter)  # unxt <-> Astropy
Unit("m")

```

## Quantities

`unxt` uses Astropy's units and dimensions objects as its backend, but has it's
own `Quantity` class hierarchy.

Converting an [Astropy Quantity][astropy-Quantity] to a [unxt
Quantity][unxt-Quantity] is straightforward -- use `unxt.Quantity.from_`:

```{code-block} python

>>> import astropy.units as u
>>> from unxt import Quantity

>>> aq = u.Quantity(1, 'm')  # Astropy Quantity
>>> aq
<Quantity 1. m>

>>> xq = Quantity.from_(aq)  # unxt Quantity
>>> xq
Quantity['length'](Array(1., dtype=float32), unit='m')
```

Alternatively, the multiple-dispatch library on which `unxt` is built enables
2-way conversion.

```{code-block} python
>>> from plum import convert

>>> convert(aq, Quantity)
Quantity['length'](Array(1., dtype=float32), unit='m')

>>> convert(xq, u.Quantity)
<Quantity 1. m>

```

<!-- Links -->

[astropy-link]: https://www.astropy.org/
[astropy-units]: https://docs.astropy.org/en/stable/units/index.html
[astropy-Quantity]:
  https://docs.astropy.org/en/stable/api/astropy.units.Quantity.html
[unxt-Quantity]:
  https://unxt.readthedocs.io/en/latest/api/quantities/#unxt.quantity.AbstractQuantity
