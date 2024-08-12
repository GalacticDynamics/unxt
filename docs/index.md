# unxt

Unxt is unitful quantities and calculations in [JAX][jax], built on
[Equinox][equinox] and [Quax][quax].

Yes, it supports auto-differentiation (`grad`, `jacobian`, `hessian`) and
vectorization (`vmap`, etc).

## Installation

[![PyPI platforms][pypi-platforms]][pypi-link]
[![PyPI version][pypi-version]][pypi-link]

<!-- [![Conda-Forge][conda-badge]][conda-link] -->

```bash
pip install unxt
```

## Getting started

```python
from unxt import Quantity

x = Quantity(jnp.arange(1, 5, dtype=float), "kpc")
print(x)
# Quantity['length'](Array([1., 2., 3., 4.], dtype=float64), unit='kpc')

# Addition / Subtraction
print(x + x)
# Quantity['length'](Array([2., 4., 6., 8.], dtype=float64), unit='kpc')

# Multiplication / Division
print(2 * x)
# Quantity['length'](Array([2., 4., 6., 8.], dtype=float64), unit='kpc')

y = Quantity(jnp.arange(4, 8, dtype=float), "Gyr")

print(x / y)
# Quantity['speed'](Array([0.25      , 0.4       , 0.5       , 0.57142857], dtype=float64), unit='kpc / Gyr')

# Exponentiation
print(x**2)
# Quantity['area'](Array([0., 1., 4., 9.], dtype=float64), unit='kpc2')

# Unit Checking on operations
try:
    x + y
except Exception as e:
    print(e)
# 'Gyr' (time) and 'kpc' (length) are not convertible
```

`unxt` is built on [`quax`][quax], which enables custom array-ish objects in
JAX. For convenience we use the [`quaxed`][quaxed] library, which is just a
`quax.quaxify` wrapper around `jax` to avoid boilerplate code.

```python
from quaxed import grad, vmap
import quaxed.numpy as jnp

print(jnp.square(x))
# Quantity['area'](Array([ 1.,  4.,  9., 16.], dtype=float64), unit='kpc2')

print(qnp.power(x, 3))
# Quantity['volume'](Array([ 1.,  8., 27., 64.], dtype=float64), unit='kpc3')

print(vmap(grad(lambda x: x**3))(x))
# Quantity['area'](Array([ 3., 12., 27., 48.], dtype=float64), unit='kpc2')
```

Since `Quantity` is parametric, it can do runtime dimension checking!

```python
LengthQuantity = Quantity["length"]
print(LengthQuantity(2, "km"))
# Quantity['length'](Array(2, dtype=int64, weak_type=True), unit='km')

try:
    LengthQuantity(2, "s")
except ValueError as e:
    print(e)
# Physical type mismatch.
```

## Citation

[![DOI][zenodo-badge]][zenodo-link]

If you found this library to be useful and want to support the development and
maintenance of lower-level code libraries for the scientific community, please
consider citing this work.

## See also: other libraries in the Quax ecosystem

[Quax][quax]: the base library.

[coordinax][coordinax]: Vector representations (built on `unxt`).

[galax][galax]: Galactic dynamics in Jax (built on `unxt` and
[coordinax][coordinax]).

<!-- LINKS -->

[coordinax]: https://github.com/GalacticDynamics/coordinax
[equinox]: https://docs.kidger.site/equinox/
[galax]: https://github.com/GalacticDynamics/galax
[jax]: https://jax.readthedocs.io/en/latest/
[quax]: https://github.com/patrick-kidger/quax
[quaxed]: https://quaxed.readthedocs.io/en/latest/
[pypi-link]: https://pypi.org/project/unxt/
[pypi-platforms]: https://img.shields.io/pypi/pyversions/unxt
[pypi-version]: https://img.shields.io/pypi/v/unxt
[zenodo-badge]: https://zenodo.org/badge/734877295.svg
[zenodo-link]: https://zenodo.org/doi/10.5281/zenodo.10850455
