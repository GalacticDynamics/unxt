# The xarray sharp bits

Two things about the integration surprise people. Both follow from how `xarray` itself works rather than from anything `unxts.interop.xarray` chose, which is why neither is going to be fixed.

## Dimension coordinates cannot hold quantities

`xarray` backs every _dimension coordinate_ (one named like its dimension, shown with a `*` in the repr) with a `pandas.Index`. Building that index coerces the data to a plain `numpy` array, so a dimension coordinate cannot hold a `Quantity`. This is inherent to `xarray`'s indexing model, not something `unxts.interop.xarray` can override, and it affects every duck-array unit library (including `pint-xarray`) the same way.

Assigning one **raises** rather than dropping the unit on the floor — `Quantity.__array__` refuses to hand a dimensionful value to a consumer that cannot see its unit. `quantify()` handles this for you by leaving dimension coordinates plain.

**Workaround**: store the unitful values on a _non-dimension_ coordinate, keeping a plain index on the dimension itself:

```{code-block} python
>>> import unxt as u
>>> import xarray as xr

>>> data = [10.0, 20.0, 30.0]
>>> quantities = u.Q([1.0, 2.0, 3.0], "m")

>>> # Dimension coordinate: refused, because the unit could not survive
>>> try:
...     xr.DataArray(data, dims=["x"], coords={"x": quantities})
... except Exception as e:
...     print(type(e).__name__)
UnitConversionError

>>> # Non-dimension coordinate: the Quantity (and its unit) is preserved
>>> da = xr.DataArray(data, dims=["i"],
...                   coords={"i": [0, 1, 2], "x": ("i", quantities)})
>>> da.coords["x"].data
Quantity(Array([1., 2., 3.], dtype=float32), unit='m')

```

## Operations that drop units

A few `xarray` operations route through code paths that cannot preserve a `Quantity`:

- **`rolling` / sliding-window reductions** use `numpy.lib.stride_tricks`, which has no Array API (or `jax.numpy`) equivalent, so they are unsupported on JAX-backed data generally — not specific to units.
- **`interp`** delegates to `scipy`/`numpy` interpolation internally and returns a plain array (the same behavior as `pint-xarray`).

For these, `dequantify`, operate, then re-`quantify`, or work on `.data` with `unxt`/`quaxed` directly.
