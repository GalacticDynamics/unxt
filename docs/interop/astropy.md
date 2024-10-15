# 🔭 Astropy

[Astropy][astropy-link] is a widely-used Python library for astronomy. One of
its many widely-used sub-packages is its units library, which provide objects
and methods for working with physical quantities, units, and dimensions.

Converting an Astropy Quantity to an unxt Quantity is straightforward -- use
`unxt.Quantity.from_`:

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
