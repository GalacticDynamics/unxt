# 🔭 Astropy

[Astropy][astropy-link] is a widely-used Python library for astronomy. One of
its many widely-used sub-packages is its [`astropy.units`][astropy-units]
library, which provide objects and methods for working with physical quantities,
units, and dimensions.

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

<!-- Links -->

[astropy-link]: https://www.astropy.org/
[astropy-units]: https://docs.astropy.org/en/stable/units/index.html
[astropy-Quantity]:
  https://docs.astropy.org/en/stable/api/astropy.units.Quantity.html
[unxt-Quantity]:
  https://unxt.readthedocs.io/en/latest/api/quantities/#unxt.quantity.AbstractQuantity
