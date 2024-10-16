<h1 align='center'> unxt </h1>
<h3 align="center">Unitful Quantities in JAX</h3>

<br>

Unxt is unitful quantities and calculations in [JAX][jax], built on
[Equinox][equinox] and [Quax][quax].

Yes, it supports auto-differentiation (`grad`, `jacobian`, `hessian`) and
vectorization (`vmap`, etc).

## Installation

[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

```bash
pip install unxt
```

## [Documentation][rtd-link]

[![Documentation Status][rtd-badge]][rtd-link]

### Quick example

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

`unxt` is built on [`quax`](https://github.com/patrick-kidger/quax), which
enables custom array-ish objects in JAX. For convenience we use the
[`quaxed`](https://pypi.org/project/quaxed/) library, which is just a
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

## Development

[![codecov][codecov-badge]][codecov-link]
[![Actions Status][actions-badge]][actions-link]

We welcome contributions!

<!-- prettier-ignore-start -->
[equinox]: https://docs.kidger.site/equinox/
[jax]: https://jax.readthedocs.io/en/latest/
[quax]: https://github.com/patrick-kidger/quax

[actions-badge]:            https://github.com/GalacticDynamics/unxt/workflows/CI/badge.svg
[actions-link]:             https://github.com/GalacticDynamics/unxt/actions
[codecov-badge]:            https://codecov.io/gh/GalacticDynamics/unxt/graph/badge.svg
[codecov-link]:             https://codecov.io/gh/GalacticDynamics/unxt
[pypi-link]:                https://pypi.org/project/unxt/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/unxt
[pypi-version]:             https://img.shields.io/pypi/v/unxt
[rtd-badge]:                https://readthedocs.org/projects/unxt/badge/?version=latest
[rtd-link]:                 https://unxt.readthedocs.io/en/latest/?badge=latest
[zenodo-badge]:             https://zenodo.org/badge/734877295.svg
[zenodo-link]:              https://zenodo.org/doi/10.5281/zenodo.10850455

<!-- prettier-ignore-end -->
