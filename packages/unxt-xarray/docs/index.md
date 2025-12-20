# unxt-xarray

```{toctree}
:maxdepth: 1
:hidden:

xarray-guide
```

`xarray` integration for unxt - JAX-based physical quantities with `xarray`'s
labeled arrays.

## Installation

::::{tab-set}

:::{tab-item} pip

```bash
pip install unxt-xarray
```

:::

:::{tab-item} uv

````bash

```bash
uv add unxt-xarray
````

:::

::::

## Quick Start

```python
import xarray as xr
import unxt as u
import unxt_xarray  # This registers the .unxt accessor

# Create DataArray with unit attributes
da = xr.DataArray(
    [1.0, 2.0, 3.0],
    dims=["time"],
    coords={"time": [0.0, 1.0, 2.0]},
    attrs={"units": "m"},
)
da.coords["time"].attrs["units"] = "s"

# Convert to unxt Quantities
quantified = da.unxt.quantify()
print(quantified.data)
# Quantity['length'](Array([1., 2., 3.], dtype=float32), unit='m')

print(quantified.coords["time"].data)
# Quantity['time'](Array([0., 1., 2.], dtype=float32), unit='s')

# Convert back to plain arrays with unit attributes
dequantified = quantified.unxt.dequantify()
print(dequantified.attrs["units"])
# 'm'
```

## Usage

### DataArray Operations

The `.unxt` accessor provides two main methods for `DataArray`:

#### `quantify()`

Convert a DataArray with unit attributes into one containing unxt Quantities:

```python
import xarray as xr
import unxt_xarray

# From attributes
da = xr.DataArray([1.0, 2.0, 3.0], dims=["x"], attrs={"units": "m"})
q = da.unxt.quantify()

# With explicit units
da = xr.DataArray([1.0, 2.0, 3.0], dims=["x"])
q = da.unxt.quantify("km")

# With coordinate units
da = xr.DataArray(
    [1.0, 2.0],
    dims=["x"],
    coords={"x": [0.0, 1.0]},
    attrs={"units": "m"},
)
q = da.unxt.quantify(x="s")  # Specify coord units
```

#### `dequantify()`

Convert Quantities back to plain arrays with unit attributes:

```python
import unxt as u

q = u.Quantity([1.0, 2.0, 3.0], "m")
da = xr.DataArray(q, dims=["x"])

plain = da.unxt.dequantify()
print(plain.attrs["units"])  # 'm'
print(type(plain.data))  # Array (not Quantity)
```

### Dataset Operations

The accessor works similarly for `Dataset` objects:

```python
import xarray as xr
import unxt_xarray

# Create Dataset with unit attributes
ds = xr.Dataset(
    {
        "temperature": ("time", [20.0, 25.0, 30.0], {"units": "deg_C"}),
        "pressure": ("time", [1.0, 1.1, 1.2], {"units": "bar"}),
    }
)

# Quantify all variables
q_ds = ds.unxt.quantify()
print(q_ds["temperature"].data)
# Quantity(Array([20., 25., 30.], dtype=float32), unit='degC')

# Dequantify back
plain_ds = q_ds.unxt.dequantify()
print(plain_ds["temperature"].attrs["units"])  # 'degC'
```

## Advanced Usage

### Custom Unit Attributes

By default, units are stored in the `"units"` attribute. You can customize this:

```python
da = xr.DataArray([1.0, 2.0], dims=["x"])
q_da = xr.DataArray(u.Quantity([1.0, 2.0], "m"), dims=["x"])

# Use custom attribute name
plain = q_da.unxt.dequantify(unit_attribute="unit_str")
print(plain.attrs["unit_str"])  # 'm'
```

### Format Strings

Control how units are formatted when dequantifying:

```python
import unxt as u

q = u.Quantity([1.0, 2.0], "m/s")
da = xr.DataArray(q, dims=["x"])

# Default format
plain = da.unxt.dequantify()
print(plain.attrs["units"])  # 'm / s'

# Custom format (if supported by unit system)
# plain = da.unxt.dequantify(format="{:~}")  # Compact format
```

### Partial Quantification

You can quantify only specific variables in a Dataset:

```python
ds = xr.Dataset(
    {
        "with_units": ("x", [1.0, 2.0], {"units": "m"}),
        "without_units": ("y", [3.0, 4.0]),
    }
)

q_ds = ds.unxt.quantify()
# Only "with_units" is quantified
```

## Best Practices

1. **Consistent Units**: Ensure unit attributes are consistent across your
   workflow
2. **Explicit is Better**: Use explicit units in `quantify()` when possible for
   clarity
3. **Preserve Attributes**: Other attributes (like `long_name`, `description`)
   are preserved
4. **JAX Compatibility**: Remember that quantified data is JAX arrays - use
   `jax.numpy` operations

## See Also

- [unxt documentation](https://unxt.readthedocs.io/)
- [xarray documentation](https://docs.xarray.dev/)
- [pint-xarray](https://pint-xarray.readthedocs.io/) - The original inspiration
- [JAX documentation](https://jax.readthedocs.io/)
