# API

For task-shaped guidance see [How to use unxt with xarray](xarray-guide).

## The `.unxt` accessor

Registered on both `DataArray` and `Dataset` when `unxts.interop.xarray` is imported.

| Member | Signature | Returns |
| --- | --- | --- |
| `.unxt.quantify` | `(units=None, **unit_kwargs)` | a copy with `Quantity` data |
| `.unxt.dequantify` | `(format=None, unit_attribute="units")` | a copy with plain arrays and unit attributes |
| `.unxt.units` | property | mapping of name → unit; the data itself is under the `None` key (`DataArray` only) |

`quantify`'s `units` accepts a unit string or object (applied to the data), or a mapping of names to units; `**unit_kwargs` is the keyword spelling of the same mapping, so `quantify(time="s")` and `quantify({"time": "s"})` are equivalent. On a `Dataset`, `units` must be a mapping — variables are named. Anything not named is read from that object's `units` attribute, and variables with no `units` attribute are left as plain arrays.

`dequantify`'s `format` is passed to {func}`format` to render each unit; the default `None` uses `str(unit)`. `unit_attribute` is the attribute the unit string is written to.

## Lower-level functions

These are the layer underneath the accessor, for inspecting or manipulating units without going through a `DataArray` or `Dataset`.

The `.unxt` accessor covers most workflows, but the four underlying functions are also exported for use in pipelines, custom integrations, or cases where you need direct control.

## `extract_unit_attributes`

Reads `"units"` attrs from each variable and coordinate — without converting anything to a Quantity. Use this to inspect declared units before committing to a conversion.

```python
import xarray as xr
from unxts.interop.xarray import extract_unit_attributes

ds = xr.Dataset(
    {
        "temperature": ("time", [273.0, 293.0], {"units": "K"}),
        "pressure": ("time", [101325.0, 102000.0]),
    }
)
print(extract_unit_attributes(ds))
# {'temperature': Unit("K")}
```

## `attach_units`

Attaches units to a DataArray or Dataset, converting plain array data into Quantities. Use `None` as the key for a DataArray's own data (as opposed to a named coordinate).

```python
import xarray as xr
from unxts.interop.xarray import attach_units

da = xr.DataArray([1.0, 2.0, 3.0], dims=["x"])
quantified = attach_units(da, {None: "m"})
print(quantified.data)
# Quantity(Array([1., 2., 3.], dtype=float32), unit='m')
```

Use `attach_units` directly when you already have a units mapping (e.g., from a file header or a prior `extract_unit_attributes` call) and want to skip the attribute-reading step.

## `extract_units`

Reads the units from **existing Quantities** in a DataArray or Dataset. This is the inverse of `attach_units` — use it when you need the units for computation before stripping them.

```python
import xarray as xr
import unxt as u
from unxts.interop.xarray import extract_units

q = u.Quantity([1.0, 2.0], "m")
da = xr.DataArray(q, dims=["x"])
print(extract_units(da))
# {None: Unit("m")}
```

## `strip_units`

Removes Quantity wrappers, returning plain arrays. The unit information is discarded unless you capture it with `extract_units` first.

```python
import xarray as xr
import unxt as u
from unxts.interop.xarray import strip_units

q = u.Quantity([1.0, 2.0], "m")
da = xr.DataArray(q, dims=["x"])
stripped = strip_units(da)
print(stripped.data)
# Array([1., 2.], dtype=float32)
```

## When to use the low-level API

| Task | Use |
| --- | --- |
| Interactive quantify/dequantify | `.unxt.quantify()` / `.unxt.dequantify()` |
| Inspect declared units without converting | `extract_unit_attributes` |
| Attach a pre-built units mapping | `attach_units` |
| Read units from already-quantified data | `extract_units` |
| Strip Quantities to plain arrays | `strip_units` |
| Build a custom quantify/dequantify pipeline | All four, composed manually |
