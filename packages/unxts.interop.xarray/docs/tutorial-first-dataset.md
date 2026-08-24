# Turn unit labels into real units

An `xarray` object can carry a `units` string in its attributes, but that string is only a label — nothing checks it, and `xarray` will happily add kilometres to hours. In this tutorial we will watch exactly that happen, then turn those labels into real `unxt` quantities so the same mistake becomes an error, compute with them, and put the labels back.

You need `unxt`, `unxts.interop.xarray` and `xarray` installed, and nothing else.

## Set up

```{code-block} python
>>> import numpy as np
>>> import xarray as xr
>>> import unxt as u
>>> import unxts.interop.xarray  # registers the .unxt accessor

```

That import is the setup. It attaches a `.unxt` accessor to `DataArray` and `Dataset`.

## Build some labelled data

Three legs of a journey — distances in kilometres, times in hours — with the units recorded the way `xarray` users normally record them, as an attribute:

```{code-block} python
>>> dist = xr.DataArray(np.array([90.0, 180.0, 270.0]),
...                     dims="leg", attrs={"units": "km"})
>>> time = xr.DataArray(np.array([1.0, 2.0, 3.0]),
...                     dims="leg", attrs={"units": "h"})

```

## Watch the labels fail to protect you

Add a distance to a time. This is meaningless, so let's see what `xarray` does about it:

```{code-block} python
>>> (dist + time).values
array([ 91., 182., 273.])

```

It did the addition. 90 km plus 1 hour is 91 of something, and we now have an array of nonsense that looks exactly like data.

Worse, look at what happened to the units:

```{code-block} python
>>> (dist + time).attrs
{}

```

Gone. The attribute could not survive an operation it had no part in, so the result is not even mislabelled — it is unlabelled. Nothing raised, nothing warned.

## Make the units real

Now `quantify()`. It reads the `units` attribute and replaces the plain array with an `unxt.Quantity`:

```{code-block} python
>>> qd = dist.unxt.quantify()
>>> qt = time.unxt.quantify()

>>> qd.data
Quantity(Array([ 90., 180., 270.], dtype=float32), unit='km')

```

The unit is no longer a string beside the numbers — it is part of them. You can ask an object what units it found, before or after:

```{code-block} python
>>> dist.unxt.units
{None: Unit("km")}

```

(The data itself lives under the `None` key; named coordinates appear under their own names.)

## Try the mistake again

Same addition, now on quantified data:

```{code-block} python
>>> try:
...     qd + qt
... except Exception as e:
...     print(type(e).__name__)
UnitConversionError

```

That is the whole point of the exercise. The operation that silently produced nonsense a moment ago now refuses to run.

## Compute something that does make sense

Distance divided by time is a speed, and the unit follows from the arithmetic:

```{code-block} python
>>> speed = qd / qt
>>> speed.data
Quantity(Array([90., 90., 90.], dtype=float32), unit='km / h')

```

Notice `km / h` — we never wrote it down. Ninety kilometres per hour on every leg, which is what the numbers say.

## Put the labels back

When you are done computing — writing to NetCDF, handing off to a library that does not know about quantities — `dequantify()` reverses the first step, moving the unit back into the attributes:

```{code-block} python
>>> speed.unxt.dequantify().attrs
{'units': 'km / h'}

```

The derived unit was written out as a label, ready to be stored.

## Do it to a whole dataset

Both methods work on a `Dataset`, converting every variable that has a `units` attribute:

```{code-block} python
>>> ds = xr.Dataset({"d": dist, "t": time})
>>> qds = ds.unxt.quantify()

>>> qds["d"].data
Quantity(Array([ 90., 180., 270.], dtype=float32), unit='km')

>>> qds["t"].data
Quantity(Array([1., 2., 3.], dtype=float32), unit='h')

```

And the round trip returns the labels exactly as they started:

```{code-block} python
>>> qds.unxt.dequantify()["d"].attrs
{'units': 'km'}

```

## Convert while you are in there

One thing worth knowing before you go: once quantified, the data converts like any other quantity.

```{code-block} python
>>> qd.data.uconvert("m")
Quantity(Array([ 90000., 180000., 270000.], dtype=float32), unit='m')

```

## What we built

You took a dataset whose units were decorative, made them real, watched a bug that had been silent become an exception, computed a derived quantity whose unit came out of the arithmetic, and wrote the labels back for storage. That is the whole `quantify` / `dequantify` cycle, and it is the shape of nearly every workflow with this package.

## Where to go next

- [How to use unxt with xarray](xarray-guide) — coordinates, per-variable units, JAX transforms, and NetCDF round trips.
- [The xarray sharp bits](sharp-bits) — why dimension coordinates cannot hold quantities.
- [API](api) — the accessor's parameters and the lower-level functions.
