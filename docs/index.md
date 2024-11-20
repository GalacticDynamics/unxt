---
sd_hide_title: true
---

<h1> <code> unxt </code> </h1>

```{toctree}
:maxdepth: 1
:hidden:
:caption: 📚 Guides

guides/quantity.md
guides/dimensions.md
guides/units_and_systems.md
guides/type-checking.md
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: 🤝 Interoperability
:glob:

interop/*
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: 🔌 API Reference

api/index.md
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: More

glossary.md
conventions.md
```

# 🚀 Get Started

Unxt is unitful quantities and calculations in [JAX][jax].

Unxt supports JAX's compelling features:

- JIT compilation (`jit`)
- vectorization (`vmap`, etc).
- auto-differentiation (`grad`, `jacobian`, `hessian`)
- GPU/TPU acceleration

And best of all, `unxt` doesn't force you to use special unit-compatible
re-exports of JAX libraries. You can use `unxt` with existing JAX code, and with
one simple [decorator](#jax-functions) it will work with `unxt.Quantity`.

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

### Construction

```{code-block} python

>>> import jax.numpy as jnp
>>> from unxt import Quantity

>>> x = Quantity(jnp.array([1, 2, 3]), "m")
>>> x
Quantity['length'](Array([1, 2, 3], dtype=int32), unit='m')
```

The constituent value and unit are accessible as attributes.

```{code-block} python
>>> x.value
Array([1, 2, 3], dtype=int32)

>>> x.unit
Unit("m")

```

### Conversion

Quantities can be converted to different units:

::::{tab-set}

:::{tab-item} method

using the explicit syntax

```{code-block} python

>>> x.uconvert("cm")
Quantity['length'](Array([100., 200., 300.], dtype=float32, weak_type=True), unit='cm')

```

or Astropy's API

```{code-block} python

>>> x.to("cm")
Quantity['length'](Array([100., 200., 300.], dtype=float32, weak_type=True), unit='cm')

```

:::

:::{tab-item} function

or a function-oriented approach

```{code-block} python

>>> from unxt import uconvert

>>> uconvert("cm", x)
Quantity['length'](Array([100., 200., 300.], dtype=float32, weak_type=True), unit='cm')

```

:::

::::

### Math

Quantities can be combined in calculations:

```{code-block} python

>>> y = Quantity([4, 5, 6], "m")
>>> x + y
Quantity['length'](Array([5, 7, 9], dtype=int32), unit='m')

>>> x * y
Quantity['area'](Array([ 4, 10, 18], dtype=int32), unit='m2')

>>> x / y
Quantity['dimensionless'](Array([0.25, 0.4 , 0.5 ], dtype=float32), unit='')

```

### JAX functions

JAX function normally only support pure JAX arrays.

```{code-block} python

>>> try: jnp.square(x)
... except TypeError: print("not a pure JAX array")
not a pure JAX array

```

We use `quax` to enable Quantity support across most of the JAX ecosystem! See
the [quax docs](https://docs.kidger.site/quax/) for implementation details. The
short version is that you can use `Quantity` in JAX functions so long they pass
through a [`quax.quaxify`](https://docs.kidger.site/quax/api/quax/#quax.quaxify)
call. Here are a few examples:

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
Quantity['area'](Array([ 5, 14, 27], dtype=int32), unit='m2')

```

:::

:::{tab-item} convenience library

[`quaxed`][quaxed] is a convenience library that pre-"quaxify"s JAX functions.
It's a drop-in replacement for much of JAX.

```{code-block} python

>>> import quaxed.numpy as jnp  # pre-quaxified JAX

>>> jnp.square(x) + jnp.multiply(x, y)
Quantity['area'](Array([ 5, 14, 27], dtype=int32), unit='m2')

```

```{note}
`quaxed` is totally optional. You can [`quax.quaxify`](https://docs.kidger.site/quax/api/quax/#quax.quaxify) manually, to only decorate your top-level functions or to call 3rd party functions.
```

:::

::::

### JIT

`unxt.Quantity` works through `jax.jit`.

```{code-block} python

>>> from jax import jit

>>> jitted_func = jit(func)
>>> jitted_func(x, y)
Quantity['area'](Array([ 5, 14, 27], dtype=int32), unit='m2')

```

### Auto-Differentiation

JAX Auto-Differentiation (AD) is supported:

```{code-block} python

>>> def f(x: Quantity["length"], t: Quantity["time"]) -> Quantity["diffusivity"]:
...    return jnp.square(x) / t

>>> x = Quantity(1.0, "m")
>>> y = Quantity(4.0, "s")

```

::::{tab-set}

:::{tab-item} grad

```{code-block} python

>>> import jax
>>> from quax import quaxify

>>> grad_f = quaxify(jax.grad(f))
>>> grad_f(x, y)
Quantity['speed'](Array(0.5, dtype=float32, weak_type=True), unit='m / s')

```

or using the convenience library

```{code-block} python

>>> import quaxed as qjax

>>> grad_f = qjax.grad(f)
>>> grad_f(x, y)
Quantity['speed'](Array(0.5, dtype=float32, weak_type=True), unit='m / s')

```

:::

:::{tab-item} jacobian

```{code-block} python

>>> jac_f = quaxify(jax.jacfwd(f))
>>> jac_f(x, y)
Quantity['speed'](Array(0.5, dtype=float32, weak_type=True), unit='m / s')

```

or using the convenience library

```{code-block} python

>>> jac_f = qjax.jacfwd(f)
>>> jac_f(x, y)
Quantity['speed'](Array(0.5, dtype=float32, weak_type=True), unit='m / s')

```

:::

:::{tab-item} hessian

```{code-block} python

>>> hess_f = quaxify(jax.hessian(f))
>>> hess_f(x, y)
Quantity['frequency'](Array(0.5, dtype=float32, weak_type=True), unit='1 / s')

```

or using the convenience library

```{code-block} python

>>> hess_f = qjax.hessian(f)
>>> hess_f(x, y)
Quantity['frequency'](Array(0.5, dtype=float32, weak_type=True), unit='1 / s')

```

:::

::::

---

## Citation

[![DOI][zenodo-badge]][zenodo-link]

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

### `unxt`'s Dependents

- [coordinax][coordinax]: Coordinates in JAX.
- [galax][galax]: Galactic dynamics in JAX.

<!-- LINKS -->

[coordinax]: https://github.com/GalacticDynamics/coordinax
[equinox]: https://docs.kidger.site/equinox/
[galax]: https://github.com/GalacticDynamics/galax
[jax]: https://jax.readthedocs.io/en/latest/
[plum]: https://pypi.org/project/plum-dispatch/
[quax]: https://github.com/patrick-kidger/quax
[quaxed]: https://quaxed.readthedocs.io/en/latest/
[pypi-link]: https://pypi.org/project/unxt/
[pypi-platforms]: https://img.shields.io/pypi/pyversions/unxt
[pypi-version]: https://img.shields.io/pypi/v/unxt
[zenodo-badge]: https://zenodo.org/badge/734877295.svg
[zenodo-link]: https://zenodo.org/doi/10.5281/zenodo.10850455
