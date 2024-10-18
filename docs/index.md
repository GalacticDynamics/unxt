# `unxt`

Unxt is unitful quantities and calculations in [JAX][jax].

Yes, it supports auto-differentiation (`grad`, `jacobian`, `hessian`) and
vectorization (`vmap`, etc).

---

## Installation

[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

::::{tab-set}

:::{tab-item} pip

```bash
pip install unxt
```

:::

:::{tab-item} uv

```bash
uv add unxt
```

:::

::::

## Quickstart

The starting point of `unxt` is the `Quantity` class. It combines a JAX array
with unit information.

```{code-block} python

>>> import jax.numpy as jnp
>>> from unxt import Quantity

>>> x = Quantity(jnp.array([1, 2, 3]), "m")
>>> x
Quantity([1. 2. 3.], 'm')

>>> x.value
ArrayImpl([1. 2. 3.])

>>> x.unit
Unit('m')

```

---

```{toctree}
:maxdepth: 1
:hidden:

getting_started.md
sharp_bits.md
conventions.md
tutorials/index.md
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: API

api/quantity.md
api/unitsystems.md
api/experimental.md
```

## Citation

[![DOI][zenodo-badge]][zenodo-link]

If you found this library to be useful and want to support the development and
maintenance of lower-level code libraries for the scientific community, please
consider citing this work.

## Ecosystem

### `unxt`'s Dependencies

- [Equinox][equinox]: one-stop JAX library, for everything that isn't already in
  core JAX.
- [Quax][quax]: JAX + multiple dispatch + custom array-ish objects.
- [Quaxed][quaxed]: pre-`quaxify`ed Jax.

### `unxt`'s Dependents

- [coordinax][coordinax]: Coordinates in JAX (built on `unxt`).
- [galax][galax]: Galactic dynamics in Jax (built on `unxt` and
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
