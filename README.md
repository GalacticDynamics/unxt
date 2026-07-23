<h1 align='center'> unxt </h1>
<h3 align="center">Unitful Quantities in JAX</h3>

<p align="center">
    <a href="https://pypi.org/project/unxt/"><img alt="PyPI: unxt" src="https://img.shields.io/pypi/v/unxt?style=flat" /></a>
    <a href="https://pypi.org/project/unxt/"><img alt="PyPI versions: unxt" src="https://img.shields.io/pypi/pyversions/unxt" /></a>
    <a href="https://unxt.readthedocs.io/en/"><img alt="ReadTheDocs" src="https://img.shields.io/badge/read_docs-here-orange" /></a>
    <a href="https://pypi.org/project/unxt/"><img alt="unxt license" src="https://img.shields.io/github/license/GalacticDynamics/unxt" /></a>
</p>
<p align="center">
    <a href="https://scientific-python.org/specs/spec-0000/"><img alt="Scientific Python SPEC-0" src="https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038" /></a>
    <a href="https://docs.astral.sh/ruff/"><img alt="ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json" /></a>
    <a href="https://pre-commit.com"><img alt="pre-commit" src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit" /></a>
    <a href="https://codspeed.io/GalacticDynamics/unxt"><img src="https://img.shields.io/endpoint?url=https://codspeed.io/badge.json" alt="CodSpeed Badge"/></a>
</p>
<p align="center">
    <a href="https://github.com/GalacticDynamics/unxt/actions"><img alt="CI status" src="https://github.com/GalacticDynamics/unxt/actions/workflows/ci.yml/badge.svg?branch=main" /></a>
    <a href="https://unxt.readthedocs.io/en/"><img alt="ReadTheDocs" src="https://readthedocs.org/projects/unxt/badge/?version=latest" /></a>
    <a href="https://codecov.io/gh/GalacticDynamics/unxt"><img alt="codecov" src="https://codecov.io/gh/GalacticDynamics/unxt/graph/badge.svg" /></a>
</p>
<p align="center">
    <a style="border-width:0" href="https://doi.org/10.21105/joss.07771"><img src="https://joss.theoj.org/papers/10.21105/joss.07771/status.svg" alt="DOI badge" /></a>
</p>

---

Unxt is unitful quantities and calculations in [JAX][jax], built on [Equinox][equinox] and [Quax][quax].

Unxt supports JAX's compelling features:

- JIT compilation (`jit`)
- vectorization (`vmap`, etc.)
- auto-differentiation (`grad`, `jacobian`, `hessian`)
- GPU/TPU/multi-host acceleration

And best of all, `unxt` doesn't force you to use special unit-compatible re-exports of JAX libraries. You can use `unxt` with existing JAX code, and with [quax][quax]'s simple decorator, JAX will work with `unxt.Quantity`.

## Installation

[![PyPI version][pypi-version]][pypi-link] [![PyPI platforms][pypi-platforms]][pypi-link]

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
pip install git+https://github.com/GalacticDynamics/unxt.git
```

</details>
<details>
  <summary>building from source</summary>

```bash
cd /path/to/parent
git clone https://github.com/GalacticDynamics/unxt.git
cd unxt
pip install -e .  # editable mode
```

</details>

## [Documentation][rtd-link]

[![Read The Docs](https://img.shields.io/badge/read_docs-here-orange)](https://unxt.readthedocs.io/en/)

For full documentation, including installation instructions, tutorials, and API reference, please see the [unxt docs][rtd-link]. This README provides a brief overview and some quick examples.

### Dimensions

Dimensions represent the physical type of a quantity, such as length, time, or mass.

```{code-block} python
>>> import unxt as u
```

Create dimensions from strings:

```{code-block} python
>>> u.dimension("length")
PhysicalType('length')
```

Dimensions support mathematical expressions:

```{code-block} python
>>> u.dimension("length / time")
PhysicalType({'speed', 'velocity'})
```

Multi-word dimension names require parentheses in expressions:

```{code-block} python
>>> u.dimension("(amount of substance) / (time)")
PhysicalType('catalytic activity')
```

### Units

Units specify the scale and dimension of measurements.

```{code-block} python
>>> meter = u.unit("m")
>>> meter
Unit("m")
```

Units can be combined, either inside the expression string or by arithmetic on `Unit` objects:

```{code-block} python
>>> u.unit("km/h")  # in the expression
Unit("km / h")

>>> u.unit("km") / u.unit("h")  # via arithmetic
Unit("km / h")
```

Get the dimension of a unit:

```{code-block} python
>>> u.dimension_of(meter)
PhysicalType('length')
```

## Unit Systems

Unit systems define consistent sets of base units for specific domains. `unxt` provides built-in unit systems and tools for creating custom ones.

### Built-in Unit Systems

```{code-block} python
>>> u.unitsystem("si")  # SI (International System of Units)
unitsystem(m, kg, s, mol, A, K, cd, rad)

>>> u.unitsystem("cgs")  # CGS (centimeter-gram-second)
unitsystem(cm, g, s, dyn, erg, Ba, P, St, rad)

>>> u.unitsystem("galactic")  # galactic (astrophysics)
unitsystem(kpc, Myr, solMass, rad)
```

### Composing Units from a Unit System

Once you have a unit system, you can get units for any physical dimension by indexing the system:

```{code-block} python
>>> usys = u.unitsystem("si")
>>> usys["length"]
Unit("m")
```

### Custom Unit Systems

Create custom unit systems by specifying base units:

```{code-block} python
>>> custom_usys = u.unitsystem("km", "h", "tonne", "degree")
>>> custom_usys
unitsystem(km, h, t, deg)
```

Derived units are then available by dimension:

```{code-block} python
>>> custom_usys["velocity"]
Unit("km / h")
```

### Dynamical Unit Systems

For domains like gravitational dynamics, use dynamical unit systems where $G = 1$. Specify only 2 of (length, time, mass); the third is computed to make $G = 1$.

```{code-block} python
>>> from unxt.unitsystems import DynamicalSimUSysFlag

>>> dyn_usys = u.unitsystem(DynamicalSimUSysFlag, "kpc", "Myr")
>>> dyn_usys
LengthMassTimeUnitSystem(length=Unit("kpc"),
                         mass=Unit("1.49828e+10 kpc3 s2 kg / (Myr2 m3)"), time=Unit("Myr"))
```

The mass unit is the derived one — an exact composite expression, not a rounded label:

```{code-block} python
>>> dyn_usys["mass"]
Unit("1.49828e+10 kpc3 s2 kg / (Myr2 m3)")
```

### Quantities

Quantities combine values with units, providing type-safe unitful arithmetic.

`Quantity` (`u.Q`) is the lightweight, non-parametric default: a single class — and a single JAX pytree type — for all physical dimensions. `ParametricQuantity` (`up.PQ`) adds runtime dimension checking and dimension-specific `plum` dispatch by encoding each dimension in its own on-the-fly class (and pytree type), which grows the type/dispatch surface and adds per-construction overhead. (This is not about `jax.jit` cache misses: the `unit` is static, so a jitted function specializes per unit with either class — that part is inherent.) See the [Quantity guide](https://unxt.readthedocs.io/en/latest/guides/quantity.html) for full details; upgrading from an earlier version? See the [migration guide](https://unxt.readthedocs.io/en/latest/migration.html).

#### Basic Quantities

```{code-block} python
>>> import jax.numpy as jnp

>>> x = u.Q(jnp.arange(1, 5, dtype=float), "km")
>>> x
Quantity(Array([1., 2., 3., 4.], dtype=float32...), unit='km')
```

The constituent value and unit are accessible as attributes:

```{code-block} python
>>> x.value
Array([1., 2., 3., 4.], dtype=float32...)

>>> x.unit
Unit("km")
```

`Quantity` objects obey the rules of unitful arithmetic — addition, subtraction, multiplication, division, and exponentiation:

```{code-block} python
>>> x + x
Quantity(Array([2., 4., 6., 8.], dtype=float32...), unit='km')

>>> 2 * x
Quantity(Array([2., 4., 6., 8.], dtype=float32...), unit='km')

>>> y = u.Q(jnp.arange(4, 8, dtype=float), "yr")
>>> x / y
Quantity(
    Array([0.25     , 0.4      , 0.5      , 0.5714286], dtype=float32...), unit='km / yr'
)

>>> x**2
Quantity(Array([ 1.,  4.,  9., 16.], dtype=float32...), unit='km2')
```

Operations are unit-checked, so mixing incompatible dimensions raises:

```{code-block} python
>>> try:
...     x + y
... except Exception as e:
...     print(e)
'yr' (time) and 'km' (length) are not convertible
```

Quantities can be converted to different units, by function or by method:

```{code-block} python
>>> u.uconvert("m", x)  # via function
Quantity(Array([1000., 2000., 3000., 4000.], dtype=float32...), unit='m')

>>> x.uconvert("m")  # via method
Quantity(Array([1000., 2000., 3000., 4000.], dtype=float32...), unit='m')
```

`ParametricQuantity` — from the separate `unxts.parametric` package (`pip install unxts.parametric`, imported below as `up`) — adds runtime dimension checking on construction. Use `up.PQ["length"]` to create a parametric type that raises if the unit's physical type does not match:

```{code-block} python
>>> import unxts.parametric as up

>>> LengthQuantity = up.PQ["length"]
>>> LengthQuantity(2, "km")
ParametricQuantity(Array(2, dtype=int32...), unit='km')
```

A unit whose physical type does not match the parameter is rejected:

```{code-block} python
>>> try:
...     LengthQuantity(2, "s")
... except ValueError as e:
...     print(e)
Physical type mismatch.
```

By contrast, the default `u.Q["length"]` accepts the subscript but does **not** check dimensions — it silently builds a `Quantity` with the mismatched unit:

```{code-block} python
>>> u.Q["length"](2, "s")
Quantity(Array(2, dtype=int32...), unit='s')
```

Use `up.PQ["length"]` when you need the runtime guard. See the [`unxts.parametric` guide](https://unxt.readthedocs.io/en/latest/packages/unxts.parametric/index.html) for the full API.

#### Quantity

`Quantity` (aliased as `u.Q`) is the default lightweight class. It does **not** do runtime dimension checking on construction, which makes it the fastest option for performance-critical code:

```{code-block} python
>>> bq = u.quantity.Quantity(jnp.array([1.0, 2.0, 3.0]), "m")
>>> bq
Quantity(Array([1., 2., 3.], dtype=float32...), unit='m')

>>> bq * 2
Quantity(Array([2., 4., 6.], dtype=float32...), unit='m')
```

#### Angle

`Angle` is a specialized quantity with wrapping support for angular values:

```{code-block} python
>>> theta = u.Angle(jnp.array([0, 90, 180, 270, 360]), "deg")
>>> theta
Angle(Array([  0,  90, 180, 270, 360], dtype=int32...), unit='deg')
```

Angles can optionally be wrapped into a specified range:

```{code-block} python
>>> angle = u.Angle(jnp.array([370, -10]), "deg")
>>> angle.wrap_to(u.Q(0, "deg"), u.Q(360, "deg"))
Angle(Array([ 10, 350], dtype=int32...), unit='deg')
```

#### StaticQuantity

For static configuration values (e.g., JAX static arguments), use `StaticQuantity`, which stores NumPy values and rejects JAX arrays:

```{code-block} python
>>> import numpy as np
>>> from functools import partial
>>> import jax

>>> cfg = u.StaticQuantity(np.array([1.0, 2.0]), "m")

>>> @partial(jax.jit, static_argnames=("q",))
... def add(x, q):
...     return x + jnp.asarray(q.value)

>>> add(1.0, cfg)
Array([2., 3.], dtype=float32...)
```

#### StaticValue

If you want a `Quantity` that keeps a static value but still participates in regular arithmetic, wrap the value with `StaticValue`. Arithmetic behaves like the wrapped array, and `StaticValue + StaticValue` returns a `StaticValue`. Equality between two `StaticValue`s (`==` / `!=`) returns a scalar `bool` — which is what makes a `StaticValue`-backed quantity hashable and usable as a `jax.jit` static argument. Ordering (`<`, `<=`, `>`, `>=`), and `==` / `!=` against a raw array, return element-wise NumPy boolean arrays:

```{code-block} python
>>> sv = u.quantity.StaticValue(np.array([1.0, 2.0]))
>>> q_static = u.Q(sv, "m")
>>> q = u.Q(jnp.array([3.0, 4.0]), "m")

>>> q_static + q
Quantity(Array([4., 6.], dtype=float32...), unit='m')
```

Equality between two `StaticValue`s is a scalar `bool`:

```{code-block} python
>>> sv2 = u.quantity.StaticValue(np.array([2.0, 1.0]))
>>> sv == sv2
False
>>> sv == u.quantity.StaticValue(np.array([1.0, 2.0]))
True
```

Ordering, and equality against a raw array, are element-wise NumPy boolean arrays:

```{code-block} python
>>> sv < sv2
array([ True, False])

>>> sv == np.array([1.0, 2.0])
array([ True,  True])
```

### JAX Integration

`unxt` is built on [`quax`][quax], which enables custom array-ish objects in JAX. For convenience we use the [`quaxed`][quaxed] library, which is just a `quax.quaxify` wrapper around `jax` to avoid boilerplate code.

> [!NOTE]
>
> Using [`quaxed`][quaxed] is optional. You can directly use `quaxify`, and even apply it to the top-level function instead of individual functions.

Using the `x` quantity from the earlier examples:

```{code-block} python
>>> from quaxed import grad, vmap
>>> import quaxed.numpy as qnp

>>> qnp.square(x)
Quantity(Array([ 1.,  4.,  9., 16.], dtype=float32...), unit='km2')

>>> qnp.power(x, 3)
Quantity(Array([ 1.,  8., 27., 64.], dtype=float32...), unit='km3')

>>> vmap(grad(lambda x: x**3))(x)
Quantity(Array([ 3., 12., 27., 48.], dtype=float32...), unit='km2')
```

See the [documentation][rtd-link] for more examples and details of JIT and AD

## Citation

[![JOSS][joss-badge]][joss-link]

If you found this library to be useful and want to support the development and maintenance of lower-level code libraries for the scientific community, please consider citing this work.

## Contributing and Development

[![Actions Status][actions-badge]][actions-link] [![Documentation Status][rtd-badge]][rtd-link] [![codecov][codecov-badge]][codecov-link] [![SPEC 0 — Minimum Supported Dependencies][spec0-badge]][spec0-link] [![pre-commit][pre-commit-badge]][pre-commit-link] [![ruff][ruff-badge]][ruff-link] [![CodSpeed Badge](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://codspeed.io/GalacticDynamics/unxt)

We welcome contributions! Contributions are how open source projects improve and grow.

To contribute to `unxt`, please [fork](https://github.com/GalacticDynamics/unxt/fork) the repository, make a development branch, develop on that branch, then [open a pull request](https://github.com/GalacticDynamics/unxt/compare) from the branch in your fork to main.

To report bugs, request features, or suggest other ideas, please [open an issue](https://github.com/GalacticDynamics/unxt/issues/new/choose).

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
