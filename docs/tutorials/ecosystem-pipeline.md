# Carry units through a whole analysis

In this tutorial we will take a dataset whose units are only labels, turn them into real units, compute a derived physical quantity, and plot it — passing the data through `xarray`, `unxt` and `matplotlib` without ever writing a conversion factor or an axis label by hand.

Three libraries that know nothing about each other, and the units survive all of it. That is the point of the `unxts.*` packages, and it is easier to see than to describe.

You need `unxt`, `unxts.interop.xarray`, `unxts.interop.matplotlib`, `xarray` and `matplotlib` installed. `pip install "unxt[interop-xarray,interop-mpl]"` gets all of it.

## Set up

```{code-block} python
>>> import matplotlib
>>> matplotlib.use("Agg")  # draw to memory, so no window opens

>>> import matplotlib.pyplot as plt
>>> import numpy as np
>>> import xarray as xr
>>> from astropy import constants as const

>>> import unxt as u
>>> import unxts.interop.xarray       # registers the .unxt accessor
>>> import unxts.interop.matplotlib   # registers the plotting converter

```

Those two imports are the entire integration. Each registers itself with the library it bridges, and then gets out of the way — see [`unxts.interop.xarray`](../packages/unxts.interop.xarray/index) and [`unxts.interop.matplotlib`](../packages/unxts.interop.matplotlib/index).

## Load data whose units are only labels

Here is a rotation curve: how fast things orbit the centre of a galaxy, as a function of how far out they are. Recorded the way scientific data usually is, with the units written in the attributes:

```{code-block} python
>>> ds = xr.Dataset(
...     {
...         "radius": ("i", np.array([2.0, 4.0, 6.0, 8.0, 10.0]),
...                    {"units": "kpc"}),
...         "v_circ": ("i", np.array([120.0, 180.0, 210.0, 220.0, 218.0]),
...                    {"units": "km/s"}),
...     }
... )

>>> ds["v_circ"].attrs
{'units': 'km/s'}

```

That `'km/s'` is a string sitting beside the numbers. Nothing enforces it — the values will happily be doubled into something that is no longer km/s and keep the label anyway:

```{code-block} python
>>> ds["v_circ"].values * 2
array([240., 360., 420., 440., 436.])

```

## Make the units real

`quantify()` reads each `units` attribute and replaces the plain arrays with `unxt` quantities:

```{code-block} python
>>> qds = ds.unxt.quantify()

>>> r = qds["radius"].data
>>> r
Quantity(Array([ 2.,  4.,  6.,  8., 10.], dtype=float32), unit='kpc')

>>> v = qds["v_circ"].data
>>> v
Quantity(Array([120., 180., 210., 220., 218.], dtype=float32), unit='km / s')

```

The unit is part of the data now, not a note attached to it.

## Compute something new

The mass enclosed within radius $r$ for a circular orbit is $M = v^2 r / G$. We need the gravitational constant, in whatever units it comes in:

```{code-block} python
>>> G = u.Q(const.G.value, "m3 / (kg s2)")
>>> G
Quantity(Array(6.6743e-11, dtype=float32), unit='m3 / (kg s2)')

```

Now just write the formula. Kiloparsecs, kilometres per second and SI metres all in one expression:

```{code-block} python
>>> M = v**2 * r / G

```

Look at what came out:

```{code-block} python
>>> M.unit
Unit("km2 kg kpc / m3")

```

That is an ugly unit, and it is exactly right — `unxt` did the algebra without tidying up after itself. Ask what it _is_, though, and the answer is clean:

```{code-block} python
>>> u.dimension_of(M)
PhysicalType('mass')

```

A mass. Notice we have verified the formula is dimensionally correct before converting anything, using the dimensions rather than the units. Now ask for it in units an astronomer reads:

```{code-block} python
>>> M_sun = M.uconvert("solMass")
>>> M_sun[3]
Quantity(Array(9.00273e+10, dtype=float32), unit='solMass')

```

About 9 × 10¹⁰ solar masses inside 8 kpc — which is roughly the right answer for the Milky Way inside the Sun's orbit. We never wrote down a single conversion factor between kiloparsecs, kilometres, metres, kilograms and solar masses.

## Plot it

Hand the quantities straight to `matplotlib`:

```{code-block} python
>>> fig, ax = plt.subplots()
>>> _ = ax.plot(r, v)

```

and look at the axes:

```{code-block} python
>>> print(ax.get_xlabel())
$\mathrm{kpc}$

>>> print(ax.get_ylabel())
$\mathrm{km\,s^{-1}}$

```

We did not call `set_xlabel`. The converter read the unit off each quantity and labelled the axis with it, typeset for the figure.

Plot the mass we derived, and the label follows the same route:

```{code-block} python
>>> fig2, ax2 = plt.subplots()
>>> _ = ax2.plot(r, M_sun)
>>> print(ax2.get_ylabel())
$\mathrm{M_{\odot}}$

```

The solar-mass symbol, from a quantity we computed rather than declared.

## Put it back

Store the derived column alongside the originals and hand the dataset back to the world it came from:

```{code-block} python
>>> qds["m_enc"] = ("i", M_sun)
>>> out = qds.unxt.dequantify()

>>> out["m_enc"].attrs
{'units': 'solMass'}

```

`dequantify()` moved the unit back into the attributes, ready to be written to NetCDF. The label on the way out was _derived_, not typed.

## What we built

A complete analysis — load, compute, plot, save — in which the units were attached to the data at the start and carried themselves through three libraries to the end. The only place a unit name was written by hand was the one place it was a genuine choice: asking for the answer in solar masses.

## The rest of the ecosystem

`unxt` is the core; each `unxts.*` package bridges it to something else. Two of them appeared above.

| Package | For | Docs |
| --- | --- | --- |
| `unxts.interop.xarray` | labelled arrays and datasets | [docs](../packages/unxts.interop.xarray/index) |
| `unxts.interop.matplotlib` | plotting quantities | [docs](../packages/unxts.interop.matplotlib/index) |
| `unxts.interop.gala` | unit systems from `gala` | [docs](../packages/unxts.interop.gala/index) |
| `unxts.parametric` | the physical dimension in the type, checked at runtime | [docs](../packages/unxts.parametric/index) |
| `unxts.linalg` | matrices whose elements carry different units | [docs](../packages/unxts.linalg/index) |
| `unxts.hypothesis` | property-based testing over quantities | [docs](../packages/unxts.hypothesis/index) |
| `unxts.api` | the abstract API, for making your own types speak `unxt` | [docs](../packages/unxts.api/index) |

Each has its own tutorial; any of them will drop into a pipeline like this one the same way these two did.

## Where to go next

- {doc}`../how-to/interoperate-with-astropy` — the other direction: bringing `astropy` quantities in.
- {doc}`../how-to/optimize-performance` — what all this costs, and where.
- {doc}`dimensional-analysis` — checking a formula the way we checked $v^2 r / G$.
