# `unxts.interop.matplotlib`

```{toctree}
:maxdepth: 1
:hidden:

tutorial-first-plot
guide
api
```

`unxts.interop.matplotlib` is the canonical location for [matplotlib](https://matplotlib.org/) integration. Importing the package — or `unxt` itself, which imports it automatically when installed — registers a `matplotlib.units.ConversionInterface` for `AbstractQuantity`, so `unxt.Quantity` objects (and the other quantity types) can be plotted directly.

## Install

The recommended install adds `unxts.interop.matplotlib` alongside `unxt` via the `interop-mpl` [extra](https://peps.python.org/pep-0508/#extras), so it, `unxt`, and `matplotlib` are resolved together as a compatible set:

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

## At a glance

Once installed, plot `Quantity` objects directly with `matplotlib`:

```{code-block} python

import matplotlib.pyplot as plt
import jax.numpy as jnp
import unxt as u

x = u.Q(jnp.linspace(0, 360, 100), "deg")
y = u.Q(jnp.sin(x.ustrip("rad")), "")

plt.plot(x, y)
```

## Pages

**Tutorial**

- [Plot data that knows its units](tutorial-first-plot) — start here: plot two quantities, watch the axis labels write themselves, and see a dataset in the wrong unit land in the right place.

**How-to**

- [How to plot quantities with matplotlib](guide) — plotting directly, keeping units through the computation with `quaxed`, and turning the converter off.

**Reference**

- [API](api) — the converter and its setup function.
