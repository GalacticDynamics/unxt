<h1 align='center'> unxt </h1>
<h3 align="center">Unitful Quantities in JAX</h3>

<p align="center">
    <a href="https://pypi.org/project/unxt/"> <img alt="PyPI: unxt" src="https://img.shields.io/pypi/v/unxt?style=flat" /> </a>
    <a href="https://pypi.org/project/unxt/"> <img alt="PyPI versions: unxt" src="https://img.shields.io/pypi/pyversions/unxt" /> </a>
    <a href="https://unxt.readthedocs.io/en/"> <img alt="ReadTheDocs" src="https://img.shields.io/badge/read_docs-here-orange" /> </a>
    <a href="https://pypi.org/project/unxt/"> <img alt="unxt license" src="https://img.shields.io/github/license/GalacticDynamics/unxt" /> </a>
</p>
<p align="center">
    <a href="https://scientific-python.org/specs/spec-0000/"> <img alt="ruff" src="https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038" /> </a>
    <a href="https://docs.astral.sh/ruff/"> <img alt="ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json" /> </a>
    <a href="https://pre-commit.com"> <img alt="pre-commit" src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit" /> </a>
    <a href="https://codspeed.io/GalacticDynamics/unxt"><img src="https://img.shields.io/endpoint?url=https://codspeed.io/badge.json" alt="CodSpeed Badge"/></a>
</p>
<p align="center">
    <a href="https://github.com/GalacticDynamics/unxt/actions"> <img alt="CI status" src="https://github.com/GalacticDynamics/unxt/actions/workflows/ci.yml/badge.svg?branch=main" /> </a>
    <a href="https://unxt.readthedocs.io/en/"> <img alt="ReadTheDocs" src="https://readthedocs.org/projects/unxt/badge/?version=latest" /> </a>
    <a href="https://codecov.io/gh/GalacticDynamics/unxt"> <img alt="codecov" src="https://codecov.io/gh/GalacticDynamics/unxt/graph/badge.svg" /> </a>
</p>
<p align="center">
    <a style="border-width:0" href="https://doi.org/10.21105/joss.07771"> <img src="https://joss.theoj.org/papers/10.21105/joss.07771/status.svg" alt="DOI badge" > </a>
</p>

---

Unxt is unitful quantities and calculations in [JAX][jax], built on
[Equinox][equinox] and [Quax][quax].

Unxt supports JAX's compelling features:

- JIT compilation (`jit`)
- vectorization (`vmap`, etc.)
- auto-differentiation (`grad`, `jacobian`, `hessian`)
- GPU/TPU/multi-host acceleration

And best of all, `unxt` doesn't force you to use special unit-compatible
re-exports of JAX libraries. You can use `unxt` with existing JAX code, and with
[quax][quax]'s simple decorator, JAX will work with `unxt.Quantity`.

## Installation

[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

```bash
pip install unxt
```

<details>
  <summary>using <code>uv</code></summary>

```bash
uv add unxt
```

</details>
<details>
  <summary>from source, using pip</summary>

```bash
pip install git+https://https://github.com/GalacticDynamics/unxt.git
```

</details>
<details>
  <summary>building from source</summary>

```bash
cd /path/to/parent
git clone https://https://github.com/GalacticDynamics/unxt.git
cd unxt
pip install -e .  # editable mode
```

</details>

## [Documentation][rtd-link]

[![Read The Docs](https://img.shields.io/badge/read_docs-here-orange)](https://unxt.readthedocs.io/en/)

### Quick example

```python
import unxt as u
import jax.numpy as jnp

x = u.Quantity(jnp.arange(1, 5, dtype=float), "km")
print(x)
# Quantity['length']([1., 2., 3., 4.], unit='km')
```

The constituent value and unit are accessible as attributes:

```python
repr(x.value)
# Array([1., 2., 3., 4.], dtype=float64)

repr(x.unit)
# Unit("m")
```

`Quantity` objects obey the rules of unitful arithmetic.

```python
# Addition / Subtraction
print(x + x)
# Quantity["length"]([2.0, 4.0, 6.0, 8.0], unit="km")

# Multiplication / Division
print(2 * x)
# Quantity["length"]([2.0, 4.0, 6.0, 8.0], unit="km")

y = u.Quantity(jnp.arange(4, 8, dtype=float), "yr")

print(x / y)
# Quantity['speed']([0.25, 0.4 , 0.5 , 0.57142857], unit='km / yr')

# Exponentiation
print(x**2)
# Quantity['area']([ 1.,  4.,  9., 16.], unit='km2')

# Unit checking on operations
try:
    x + y
except Exception as e:
    print(e)
# 'yr' (time) and 'km' (length) are not convertible
```

Quantities can be converted to different units:

```python
print(u.uconvert("m", x))  # via function
# Quantity['length']([1000., 2000., 3000., 4000.], unit='m')

print(x.uconvert("m"))  # via method
# Quantity['length']([1000., 2000., 3000., 4000.], unit='m')
```

Since `Quantity` is parametric, it can do runtime dimension checking!

```python
LengthQuantity = u.Quantity["length"]
print(LengthQuantity(2, "km"))
# Quantity['length'](2, unit='km')

try:
    LengthQuantity(2, "s")
except ValueError as e:
    print(e)
# Physical type mismatch.
```

`unxt` is built on [`quax`][quax], which enables custom array-ish objects in
JAX. For convenience we use the [`quaxed`][quaxed] library, which is just a
`quax.quaxify` wrapper around `jax` to avoid boilerplate code.

> [!NOTE]
>
> Using [`quaxed`][quaxed] is optional. You can directly use `quaxify`, and even
> apply it to the top-level function instead of individual functions.

```python
from quaxed import grad, vmap
import quaxed.numpy as jnp

print(jnp.square(x))
# Quantity['area']([ 1.,  4.,  9., 16.], unit='km2')

print(jnp.power(x, 3))
# Quantity['volume']([ 1.,  8., 27., 64.], unit='km3')

print(vmap(grad(lambda x: x**3))(x))
# Quantity['area']([ 3., 12., 27., 48.], unit='km2')
```

See the [documentation][rtd-link] for more examples and details of JIT and AD

## Citation

[![JOSS][joss-badge]][joss-link] [![DOI][zenodo-badge]][zenodo-link]

If you found this library to be useful and want to support the development and
maintenance of lower-level code libraries for the scientific community, please
consider citing this work.

## Contributing and Development

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]
[![codecov][codecov-badge]][codecov-link]
[![SPEC 0 — Minimum Supported Dependencies][spec0-badge]][spec0-link]
[![pre-commit][pre-commit-badge]][pre-commit-link]
[![ruff][ruff-badge]][ruff-link]
[![CodSpeed Badge](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://codspeed.io/GalacticDynamics/unxt)

We welcome contributions! Contributions are how open source projects improve and
grow.

To contribute to `unxt`, please
[fork](https://github.com/GalacticDynamics/unxt/fork) the repository, make a
development branch, develop on that branch, then
[open a pull request](https://github.com/GalacticDynamics/unxt/compare) from the
branch in your fork to main.

To report bugs, request features, or suggest other ideas, please
[open an issue](https://github.com/GalacticDynamics/unxt/issues/new/choose).

For more information, see [CONTRIBUTING.md](CONTRIBUTING.md).

<!-- prettier-ignore-start -->
[equinox]: https://docs.kidger.site/equinox/
[jax]: https://jax.readthedocs.io/en/latest/
[quax]: https://github.com/patrick-kidger/quax
[quaxed]: https://github.com/GalacticDynamics/quaxed

[actions-badge]:            https://github.com/GalacticDynamics/unxt/actions/workflows/ci.yml/badge.svg?branch=main
[actions-link]:             https://github.com/GalacticDynamics/unxt/actions
[codecov-badge]:            https://codecov.io/gh/GalacticDynamics/unxt/graph/badge.svg
[codecov-link]:             https://codecov.io/gh/GalacticDynamics/unxt
[joss-badge]:               https://joss.theoj.org/papers/10.21105/joss.07771/status.svg
[joss-link]:                https://doi.org/10.21105/joss.07771
[pre-commit-badge]:         https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit
[pre-commit-link]:          https://pre-commit.com
[pypi-link]:                https://pypi.org/project/unxt/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/unxt
[pypi-version]:             https://img.shields.io/pypi/v/unxt
[rtd-badge]:                https://readthedocs.org/projects/unxt/badge/?version=latest
[rtd-link]:                 https://unxt.readthedocs.io/en/
[ruff-badge]:               https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json
[ruff-link]:                https://docs.astral.sh/ruff/
[spec0-badge]:              https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038
[spec0-link]:               https://scientific-python.org/specs/spec-0000/
[zenodo-badge]:             https://zenodo.org/badge/734877295.svg
[zenodo-link]:              https://zenodo.org/doi/10.5281/zenodo.10850455

<!-- prettier-ignore-end -->
