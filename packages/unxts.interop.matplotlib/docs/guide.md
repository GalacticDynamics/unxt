# How to plot quantities with matplotlib

This guide shows you how to plot `unxt.Quantity` objects with [matplotlib](https://matplotlib.org/).

## Setup

Importing `unxts.interop.matplotlib` registers a `matplotlib.units.ConversionInterface` for `unxt.quantity.AbstractQuantity` (so `Quantity` and the other quantity types work). `unxt` imports the package automatically when it is installed, so plotting usually just works once the package is present:

```{code-block} python

import matplotlib.pyplot as plt
import unxt as u
```

## Plotting quantities

Pass `Quantity` objects straight to `matplotlib` — the registered converter strips the units to their magnitudes for the axes:

```{code-block} python

import jax.numpy as jnp

x = u.Q(jnp.linspace(0, 360, 100), "deg")
y = u.Q(jnp.sin(x.ustrip("rad")), "")

fig, ax = plt.subplots()
ax.plot(x, y)
```

## Using `quaxed.numpy`

With [`quaxed`](https://github.com/GalacticDynamics/quaxed)'s unit-aware `numpy` namespace, the intermediate values stay `Quantity` objects and units propagate through the computation:

```{code-block} python

import quaxed.numpy as jnp

x = u.Q(jnp.linspace(0, 360, 100), "deg")
y = jnp.sin(x)

fig, ax = plt.subplots()
ax.plot(x, y)
```

## Disabling the converter

The converter is enabled on import. To turn it off (or back on) at runtime, use `setup_matplotlib_support_for_unxt` — see the [API reference](api):

```{code-block} python

from unxts.interop.matplotlib import setup_matplotlib_support_for_unxt

setup_matplotlib_support_for_unxt(enable=False)  # stop converting Quantity
setup_matplotlib_support_for_unxt(enable=True)  # re-enable
```

## See also

- [Plot data that knows its units](tutorial-first-plot) — the lesson, if you have not plotted a quantity before.
- [API](api) — the converter and its setup function.
- [matplotlib units documentation](https://matplotlib.org/stable/gallery/units/index.html)
