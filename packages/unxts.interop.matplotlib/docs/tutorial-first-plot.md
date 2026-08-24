# Plot data that knows its units

We are going to plot two `unxt` quantities and never once tell `matplotlib` what the axes are measured in. The units come from the data, the axis labels write themselves, and a second dataset in a different unit still lands in the right place on the same axes.

You need `unxt`, `unxts.interop.matplotlib` and `matplotlib` installed, and nothing else.

## Set up

```{code-block} python
>>> import matplotlib
>>> matplotlib.use("Agg")  # draw to memory, so no window opens

>>> import matplotlib.pyplot as plt
>>> import quaxed.numpy as jnp
>>> import unxt as u
>>> import unxts.interop.matplotlib  # registers the converter

```

That last import is the whole setup. It registers a converter with `matplotlib`, and from then on `matplotlib` knows what to do with a `Quantity`. (`unxt` imports it for you when the package is installed, so in your own code you will usually not write that line at all.)

`matplotlib.use("Agg")` just keeps this page from trying to open a window.

## Make some unitful data

Let's plot a journey: distances in kilometres against times in hours.

```{code-block} python
>>> t = u.Q(jnp.asarray([0.0, 1.0, 2.0]), "hr")
>>> d = u.Q(jnp.asarray([0.0, 90.0, 180.0]), "km")

>>> t
Quantity(Array([0., 1., 2.], dtype=float32), unit='h')

```

Notice `matplotlib` has not been mentioned yet. These are ordinary quantities.

## Plot them

```{code-block} python
>>> fig, ax = plt.subplots()
>>> _ = ax.plot(t, d)

```

Now look at the axes:

```{code-block} python
>>> print(ax.get_xlabel())
$\mathrm{h}$

>>> print(ax.get_ylabel())
$\mathrm{km}$

```

We never called `set_xlabel`. The converter read the unit off each quantity and labelled the axis with it, typeset as LaTeX so it renders properly in the figure.

The axis has also _remembered_ its unit:

```{code-block} python
>>> ax.xaxis.get_units()
Unit("h")

```

## Add data in a different unit

Now for the interesting bit. Let's add a second journey: same distances, but the times given in **minutes** instead of hours.

```{code-block} python
>>> t_minutes = u.Q(jnp.asarray([0.0, 60.0, 120.0]), "min")
>>> _ = ax.plot(t_minutes, d)

```

The x-axis is in hours and we just handed it minutes. Check what happened to the label:

```{code-block} python
>>> print(ax.get_xlabel())
$\mathrm{h}$

```

Still hours. And check where the data landed:

```{code-block} python
>>> [round(float(v), 1) for v in ax.get_xlim()]
[-0.1, 2.1]

```

The axis still spans about 0 to 2, not 0 to 120. Our 120 minutes was **converted to 2 hours** and drawn on top of the first line, because the axis already had a unit and the converter reconciled the new data with it.

Had these been plain arrays, 120 would have been plotted at 120 and the figure would have been silently wrong.

## Let units flow through a computation

Because the data is unitful all the way, you can compute with it and plot the result. Here is speed, derived rather than typed in:

```{code-block} python
>>> speed = d / t_minutes.uconvert("hr")
>>> speed.unit
Unit("km / h")

>>> fig2, ax2 = plt.subplots()
>>> _ = ax2.plot(t, speed)
>>> print(ax2.get_ylabel())
$\mathrm{km\,h^{-1}}$

```

The `km / h` label came out of the division. Nothing in the plotting code knows what a speed is.

Using [`quaxed`](https://quaxed.readthedocs.io/)'s `numpy` keeps units through maths functions too:

```{code-block} python
>>> angle = u.Q(jnp.linspace(0.0, 360.0, 100), "deg")
>>> wave = jnp.sin(angle)
>>> wave.unit
Unit(dimensionless)

>>> fig3, ax3 = plt.subplots()
>>> _ = ax3.plot(angle, wave)
>>> print(ax3.get_xlabel())
$\mathrm{{}^{\circ}}$

>>> plt.close("all")

```

`jnp.sin` took degrees and returned a dimensionless result, and the x-axis picked up the degree symbol on its own.

## What we built

Three figures, no axis labels written by hand, and a dataset in the wrong unit placed correctly instead of silently misplotted. All of it came from one import.

## Where to go next

- [How to plot quantities with matplotlib](guide) — turning the converter off, and using it alongside `quaxed`.
- [API](api) — the converter and its setup function.
