# The xarray sharp bits

Two things about the integration surprise people. Both follow from how `xarray` itself works rather than from anything `unxts.interop.xarray` chose, which is why neither is going to be fixed.

## Dimension Coordinates Cannot Hold Quantities

`xarray` backs every _dimension coordinate_ (one named like its dimension, shown with a `*` in the repr) with a `pandas.Index`. Building that index coerces the data to a plain `numpy` array, so a `Quantity` assigned to a dimension coordinate is silently unwrapped — its unit is lost. This is inherent to `xarray`'s indexing model, not something `unxts.interop.xarray` can override, and it affects every duck-array unit library (including `pint-xarray`) the same way.

**Workaround**: store the unitful values on a _non-dimension_ coordinate, keeping a plain index on the dimension itself:

```python
import unxt as u
import xarray as xr

data = [10.0, 20.0, 30.0]
quantities = u.Quantity([1.0, 2.0, 3.0], "m")

# Dimension coordinate: ``x`` is unwrapped to a plain array, unit lost
da = xr.DataArray(data, dims=["x"], coords={"x": quantities})
print(type(da.coords["x"].data).__name__)
# ndarray

# Non-dimension coordinate: the Quantity (and its unit) is preserved
da = xr.DataArray(data, dims=["i"], coords={"i": [0, 1, 2], "x": ("i", quantities)})
print(da.coords["x"].data)
# Quantity(Array([1., 2., 3.], dtype=float32), unit='m')
```

## Operations That Drop Units

A few `xarray` operations route through code paths that cannot preserve a `Quantity`:

- **`rolling` / sliding-window reductions** use `numpy.lib.stride_tricks`, which has no Array API (or `jax.numpy`) equivalent, so they are unsupported on JAX-backed data generally — not specific to units.
- **`interp`** delegates to `scipy`/`numpy` interpolation internally and returns a plain array (the same behavior as `pint-xarray`).

For these, `dequantify`, operate, then re-`quantify`, or work on `.data` with `unxt`/`quaxed` directly.
