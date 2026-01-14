---
sd_hide_title: true
---

<h1> <code> unxt </code> </h1>

```{toctree}
:maxdepth: 1
:hidden:
:caption: 📦 Packages

unxt-api <packages/unxt-api/index>
unxt <self>
unxt-hypothesis <packages/unxt-hypothesis/index>
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: 📚 Guides

guides/quantity
guides/dimensions
guides/units_and_systems
guides/type-checking
guides/sharp-bits
packages/unxt-api/extending
packages/unxt-hypothesis/testing-guide
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: 🤝 Interoperability
:glob:

interop/*
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: 🔌 API Reference

packages/unxt-api/api
api/index
packages/unxt-hypothesis/api
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: More

glossary
conventions
contributing
dev
```

# 🚀 Get Started

Unxt is unitful quantities and calculations in [JAX][jax].

Unxt supports JAX's compelling features:

- JIT compilation (`jit`)
- vectorization (`vmap`, etc.)
- auto-differentiation (`grad`, `jacobian`, `hessian`)
- GPU/TPU/multi-host acceleration

And best of all, `unxt` doesn't force you to use special unit-compatible
re-exports of JAX libraries. You can use `unxt` with existing JAX code, and with
one simple [decorator](#jax-functions), JAX will work with `unxt.Quantity`.

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

:::{tab-item} source, via pip

```bash
pip install git+https://https://github.com/GalacticDynamics/unxt.git
```

:::

:::{tab-item} building from source

```bash
cd /path/to/parent
git clone https://https://github.com/GalacticDynamics/unxt.git
cd unxt
pip install -e .  # editable mode
```

:::

::::

## Quickstart

### Dimensions

Dimensions represent the physical nature of quantities (length, time, velocity, etc.).
They're independent of the units used to measure them.

```{code-block} python

>>> import unxt as u

>>> length_dim = u.dimension("length")
>>> length_dim
PhysicalType('length')

```

Dimensions support mathematical expressions:

```{code-block} python

>>> speed_dim = u.dimension("length / time")
>>> speed_dim
PhysicalType({'speed', 'velocity'})

```

Multi-word dimension names require parentheses in expressions:

```{code-block} python

>>> activity_dim = u.dimension("(amount of substance) / (time)")
>>> activity_dim
PhysicalType('catalytic activity')

```

For more details, see the [Dimensions Guide](guides/dimensions.md).

### Units

Units specify the scale and dimension of measurements. The same dimension can be
measured in many different units:

```{code-block} python

>>> meter = u.unit("m")
>>> meter
Unit("m")

```

Units can be combined arithmetically:

```{code-block} python

>>> velocity_unit = u.unit("km") / u.unit("h")
>>> velocity_unit
Unit("km / h")

```

Get the dimension of a unit:

```{code-block} python

>>> u.dimension_of(meter)
PhysicalType('length')

```

For more details, see the [Units and Systems Guide](guides/units_and_systems.md).

### Unit Systems

Unit systems define consistent sets of base units for specific domains. `unxt`
provides built-in unit systems and tools for creating custom ones.

#### Built-in Unit Systems

```{code-block} python

>>> si = u.unitsystem("si")
>>> si
unitsystem(m, kg, s, mol, A, K, cd, rad)

>>> cgs = u.unitsystem("cgs")
>>> cgs
unitsystem(cm, g, s, dyn, erg, Ba, P, St, rad)

>>> galactic = u.unitsystem("galactic")
>>> galactic
unitsystem(kpc, Myr, solMass, rad)

>>> solarsystem = u.unitsystem("solarsystem")
>>> solarsystem
unitsystem(AU, yr, solMass, rad)

```

#### Composing Units from a Unit System

Once you have a unit system, you can get units for any physical dimension by
indexing the system:

```{code-block} python

>>> usys = u.unitsystem("si")

>>> usys["length"]
Unit("m")

>>> usys["velocity"]
Unit("m / s")

>>> usys["energy"]
Unit("m2 kg / s2")

```

You can use unit system units to create quantities:

```{code-block} python

>>> q = u.Quantity(10, usys["velocity"])
>>> q
Quantity(Array(10, dtype=int32, ...), unit='m / s')

```

#### Custom Unit Systems

Create custom unit systems by specifying base units:

```{code-block} python

>>> custom_usys = u.unitsystem("km", "h", "tonne", "degree")
>>> custom_usys
unitsystem(km, h, t, deg)

>>> custom_usys["velocity"]
Unit("km / h")

```

#### Dynamical Unit Systems

For domains like gravitational dynamics, use dynamical unit systems where
$G = 1$:

```{code-block} python

>>> from unxt.unitsystems import DynamicalSimUSysFlag

>>> usys = u.unitsystem(DynamicalSimUSysFlag, "kpc", "Myr")
>>> usys  # doctest: +SKIP
unitsystem(kpc, Myr, ...)

```

For more details, see the [Unit Systems Guide](guides/units_and_systems.md).


### Creating and Working with Quantity objects

The primary API of {mod}`unxt` is the {class}`~unxt.Quantity` class. It combines a
JAX array with unit information. We currently use [astropy.units][apyunits] for
unit handling.

Create a `Quantity` by passing a JAX array-compatible object and a unit:

```{code-block} python

>>> import unxt as u

>>> x = u.Quantity([1.0, 2.0, 3.0], unit="m")  # or u.Q(...) for short
>>> x
Quantity(Array([1., 2., 3.], dtype=float32), unit='m')
```

As a shorthand, we also support `u.Q` and specifying units as strings
(parsed by `unxt.unit`, using Astropy as the backend):

```{code-block} python

>>> y = u.Q([4.0, 5.0, 6.0], "m")
>>> y
Quantity(Array([4., 5., 6.], dtype=float32), unit='m')
```

The constituent value and unit are accessible as attributes:

```{code-block} python
>>> x.value
Array([1., 2., 3.], dtype=float32)

>>> x.unit
Unit("m")

```

`Quantity` objects obey the rules of unitful arithmetic. For example, adding,
multiplying, or dividing two quantities produces a new `Quantity` with the
correct units:

```{code-block} python

>>> x + y
Quantity(Array([5., 7., 9.], dtype=float32), unit='m')

>>> x * y
Quantity(Array([ 4., 10., 18.], dtype=float32), unit='m2')

>>> x / y
Quantity(Array([0.25, 0.4 , 0.5 ], dtype=float32), unit='')

```

Arithmetic will raise an error if the units are incompatible:

```{code-block} python

>>> z = u.Q(5.0, "second")
>>> try: x + z
... except Exception as e: print(e)
's' (time) and 'm' (length) are not convertible
```

### Converting Units

Quantities can be converted to different units:

::::{tab-set}

:::{tab-item} method

using the explicit syntax

```{code-block} python

>>> x.uconvert("cm")
Quantity(Array([100., 200., 300.], dtype=float32), unit='cm')

```

or Astropy's API

```{code-block} python

>>> x.to("cm")
Quantity(Array([100., 200., 300.], dtype=float32), unit='cm')

```

:::

:::{tab-item} function

or a function-oriented approach

```{code-block} python

>>> u.uconvert("cm", x)
Quantity(Array([100., 200., 300.], dtype=float32), unit='cm')

```

:::

::::


### JAX functions

JAX functions normally only support pure JAX arrays.

```{code-block} python

>>> import jax.numpy as jnp

>>> try: jnp.square(x)
... except TypeError: print("not a pure JAX array")
not a pure JAX array

```

We use `quax` to enable Quantity support across most of the JAX ecosystem! See
the [quax docs](https://docs.kidger.site/quax/) for implementation details. The
short explanation is that you can use `Quantity` in JAX functions so long they
pass through a
[`quax.quaxify`](https://docs.kidger.site/quax/api/quax/#quax.quaxify) call.
Here are a few examples:

::::{tab-set}

:::{tab-item} using `quaxify`

This is the way to "quaxify" a JAX function. A powerful feature of `quaxify` is
that it enables `Quantity` support through _all_ the JAX functions inside the
top function. With `unxt` you can use normal JAX!

```{code-block} python
:emphasize-lines: 4

>>> import jax.numpy as jnp  # regular JAX
>>> from quax import quaxify

>>> @quaxify  # Now it works with Quantity... that's it!
... def func(x, y):
...     return jnp.square(x) + jnp.multiply(x, y)  # normal JAX

>>> func(x, y)
Quantity(Array([ 5., 14., 27.], dtype=float32), unit='m2')

```

:::

:::{tab-item} convenience library

[`quaxed`][quaxed] is a convenience library that pre-"quaxify"s JAX functions.
It's a drop-in replacement for much of JAX.

```{code-block} python

>>> import quaxed.numpy as jnp  # pre-quaxified JAX

>>> jnp.square(x) + jnp.multiply(x, y)
Quantity(Array([ 5., 14., 27.], dtype=float32), unit='m2')

```

```{note}
`quaxed` is totally optional. You can [`quax.quaxify`](https://docs.kidger.site/quax/api/quax/#quax.quaxify) manually, to only decorate your top-level functions or to call 3rd party functions.
```

:::

::::

### JIT

`unxt.Quantity` works through `jax.jit`:

```{code-block} python

>>> from jax import jit

>>> jitted_func = jit(func)
>>> jitted_func(x, y)
Quantity(Array([ 5., 14., 27.], dtype=float32), unit='m2')

```

Static quantities are also available when you need JAX-static configuration or
constants:

```{code-block} python

>>> import numpy as np
>>> from functools import partial
>>> import jax.numpy as jnp
>>> from jax import jit
>>> import unxt as u

>>> sq = u.StaticQuantity(np.array([1.0, 2.0]), "m")
>>> jq = u.Q(jnp.array([1.0, 1.0]), "m")

>>> @partial(jit, static_argnames=("sq",))
... def add(jq, sq):
...     return jq + u.Q(jnp.asarray(sq.value), sq.unit)

>>> add(jq, sq)
Quantity(Array([2., 3.], dtype=float32), unit='m')

```

You can also keep a static value inside a regular `Quantity` by wrapping it
with `StaticValue`. Arithmetic behaves like the wrapped array, and
`StaticValue + StaticValue` returns a `StaticValue`:

```{code-block} python

>>> import numpy as np
>>> import jax.numpy as jnp
>>> import unxt as u

>>> sv = u.quantity.StaticValue(np.array([1.0, 2.0]))
>>> q_static = u.Q(sv, "m")
>>> q = u.Q(jnp.array([3.0, 4.0]), "m")

>>> q_static + q
Quantity(Array([4., 6.], dtype=float32), unit='m')

```

### Auto-Differentiation

JAX Auto-Differentiation (AD) is supported:

```{code-block} python

>>> def f(x: u.Q["length"], t: u.Q["time"]) -> u.Q["diffusivity"]:
...    return jnp.square(x) / t

>>> x = u.Q(1.0, "m")
>>> y = u.Q(4.0, "s")

```

::::{tab-set}

:::{tab-item} grad

```{code-block} python

>>> import jax
>>> from quax import quaxify

>>> grad_f = quaxify(jax.grad(f))
>>> grad_f(x, y)
Quantity(Array(0.5, dtype=float32, weak_type=True), unit='m / s')

```

or using the convenience library

```{code-block} python

>>> import quaxed as qjax

>>> grad_f = qjax.grad(f)
>>> grad_f(x, y)
Quantity(Array(0.5, dtype=float32, weak_type=True), unit='m / s')

```

:::

:::{tab-item} jacobian

```{code-block} python

>>> jac_f = quaxify(jax.jacfwd(f))
>>> jac_f(x, y)
Quantity(Array(0.5, dtype=float32, weak_type=True), unit='m / s')

```

or using the convenience library

```{code-block} python

>>> jac_f = qjax.jacfwd(f)
>>> jac_f(x, y)
Quantity(Array(0.5, dtype=float32, weak_type=True), unit='m / s')

```

:::

:::{tab-item} hessian

```{code-block} python

>>> hess_f = quaxify(jax.hessian(f))
>>> hess_f(x, y)
Quantity(Array(0.5, dtype=float32, weak_type=True), unit='1 / s')

```

or using the convenience library

```{code-block} python

>>> hess_f = qjax.hessian(f)
>>> hess_f(x, y)
Quantity(Array(0.5, dtype=float32, weak_type=True), unit='1 / s')

```

:::

::::

## Citation

[![JOSS][joss]][joss-link] [![DOI][zenodo-badge]][zenodo-link]

If you found this library to be useful and want to support the development and
maintenance of lower-level code libraries for the scientific community, please
consider citing this work.

---

## Ecosystem

### `unxt`'s Dependencies

- [Equinox][equinox]: one-stop JAX library, for everything that isn't already in
  core JAX.
- [Quax][quax]: JAX + multiple dispatch + custom array-ish objects.
- [Quaxed][quaxed]: pre-`quaxify`ed Jax.
- [plum][plum]: multiple dispatch in python
- [unxt-api][unxt-api]: the API for `unxt`.

### `unxt`'s Dependents

- [unxt-hypothesis][unxt-hypothesis]: `unxt` integration with `hypothesis` property-based testing.

- [coordinax][coordinax]: Coordinates in JAX.
- [galax][galax]: Galactic dynamics in JAX.

<!-- LINKS -->

[apyunits]: https://docs.astropy.org/en/stable/units/index.html
[coordinax]: https://github.com/GalacticDynamics/coordinax
[equinox]: https://docs.kidger.site/equinox/
[galax]: https://github.com/GalacticDynamics/galax
[jax]: https://jax.readthedocs.io/en/latest/
[joss]: https://joss.theoj.org/papers/10.21105/joss.07771/status.svg
[joss-link]: https://doi.org/10.21105/joss.07771
[plum]: https://pypi.org/project/plum-dispatch/
[quax]: https://github.com/patrick-kidger/quax
[quaxed]: https://quaxed.readthedocs.io/en/latest/
[unxt-api]: https://pypi.org/project/unxt-api/
[unxt-hypothesis]: https://pypi.org/project/unxt-hypothesis/
[pypi-link]: https://pypi.org/project/unxt/
[pypi-platforms]: https://img.shields.io/pypi/pyversions/unxt
[pypi-version]: https://img.shields.io/pypi/v/unxt
[zenodo-badge]: https://zenodo.org/badge/734877295.svg
[zenodo-link]: https://zenodo.org/doi/10.5281/zenodo.10850455
