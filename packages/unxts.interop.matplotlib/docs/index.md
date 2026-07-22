# `unxts.interop.matplotlib`

```{toctree}
:maxdepth: 1
:hidden:

guide
api
```

`unxts.interop.matplotlib` is the canonical location for [matplotlib](https://matplotlib.org/) integration. Importing the package — or `unxt` itself, which imports it automatically when installed — registers a `matplotlib.units.ConversionInterface` so that `unxt.Quantity` objects can be plotted directly.

## Installation

The recommended install pins a compatible `matplotlib` version via the `interop-mpl` [extra](https://peps.python.org/pep-0508/#extras):

::::{tab-set}

:::{tab-item} uv

```bash
uv add "unxt[interop-mpl]"
```

:::

:::{tab-item} pip

```bash
pip install "unxt[interop-mpl]"
```

:::

::::

Or install the package directly:

::::{tab-set}

:::{tab-item} uv

```bash
uv add unxts.interop.matplotlib
```

:::

:::{tab-item} pip

```bash
pip install unxts.interop.matplotlib
```

:::

::::

## Quick example

Once installed, plot `Quantity` objects directly with `matplotlib`:

```{code-block} python

import matplotlib.pyplot as plt
import jax.numpy as jnp
import unxt as u

x = u.Q(jnp.linspace(0, 360, 100), "deg")
y = u.Q(jnp.sin(x.ustrip("rad")), "")

plt.plot(x, y)
```

See the [guide](guide) for more plotting patterns, and the [API reference](api) for toggling the converter.
