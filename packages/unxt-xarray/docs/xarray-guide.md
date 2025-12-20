# `xarray` Integration Guide

This guide shows how to use `unxt-xarray` to integrate JAX-based physical
quantities with `xarray`'s labeled multi-dimensional arrays.

## Overview

`unxt-xarray` provides seamless integration between:

- **`unxt`**: JAX-based physical quantities with dimension checking
- **`xarray`**: N-dimensional labeled arrays for scientific computing

The integration enables you to:

- Attach physical units to `xarray` DataArrays and Datasets
- Preserve units through `xarray` operations
- Convert between unit-aware (Quantity) and plain arrays with metadata
- Use JAX transformations (jit, vmap, grad) on unit-aware `xarray` objects

## Installation

::::{tab-set}

:::{tab-item} pip

```bash
pip install unxt-xarray
```

:::

:::{tab-item} uv

```bash
uv add unxt-xarray
```

:::

::::

## Basic Usage

### The `.unxt` Accessor

After importing `unxt_xarray`, all DataArrays and Datasets gain a `.unxt`
accessor with two main methods:

- `quantify()`: Convert attrs to Quantities
- `dequantify()`: Convert Quantities back to plain arrays with attrs

```python
import xarray as xr
import unxt as u
import unxt_xarray  # Registers the .unxt accessor

# Create a DataArray with unit metadata
da = xr.DataArray(
    [1.0, 2.0, 3.0],
    dims=["x"],
    attrs={"units": "m"},
)

# Convert to Quantities
quantified = da.unxt.quantify()
print(quantified.data)
# Quantity['length'](Array([1., 2., 3.], dtype=float32), unit='m')

# Convert back
dequantified = quantified.unxt.dequantify()
print(dequantified.attrs["units"])
# 'm'
```

## Working with DataArrays

### Quantifying from Attributes

The most common workflow starts with `xarray` objects that have unit information
stored in their `attrs`:

```python
import jax.numpy as jnp
import xarray as xr
import unxt as u
import unxt_xarray

# Temperature data with metadata
temp = xr.DataArray(
    jnp.array([20.0, 25.0, 30.0]),
    dims=["time"],
    coords={"time": [0, 1, 2]},
    attrs={"units": "K", "description": "Temperature measurements"},
)

# Convert to Quantities - units become part of the data
q_temp = temp.unxt.quantify()
print(q_temp.data)
# Quantity['temperature'](Array([20., 25., 30.], dtype=float32), unit='K')

# Other attributes are preserved
print(q_temp.attrs["description"])
# 'Temperature measurements'
```

### Explicit Unit Specification

You can override or set units explicitly:

```python
# Override the units attribute
da = xr.DataArray([100.0, 200.0], dims=["x"], attrs={"units": "cm"})
quantified = da.unxt.quantify(units={"<this-array>": "m"})
print(quantified.data)
# Quantity['length'](Array([100., 200.], dtype=float32), unit='m')
```

The special key `"<this-array>"` refers to the DataArray's data itself.

### Coordinates with Units

Coordinates can also have units:

```python
import xarray as xr
from xarray import Variable
import unxt as u

# Create coordinate with units using Variable
time_coord = Variable(dims=["time"], data=[0.0, 1.0, 2.0], attrs={"units": "s"})

da = xr.DataArray(
    [10.0, 20.0, 30.0],
    dims=["time"],
    coords={"time": time_coord},
    attrs={"units": "m"},
)

# Quantify both data and coordinates
quantified = da.unxt.quantify()
print(quantified.data)
# Quantity['length'](Array([10., 20., 30.], dtype=float32), unit='m')
print(quantified.coords["time"].data)
# Quantity['time'](Array([0., 1., 2.], dtype=float32), unit='s')
```

**Important**: Use non-dimension coordinates (coordinates not marked with `*` in
`xarray` output) to preserve Quantity objects. Dimension coordinates are
automatically converted to plain arrays by `xarray`.

## Working with Datasets

Datasets work similarly but handle multiple data variables:

```python
import xarray as xr
import unxt as u
import unxt_xarray

# Create Dataset with multiple variables
ds = xr.Dataset(
    {
        "temperature": (["time"], [273.0, 293.0, 313.0], {"units": "K"}),
        "pressure": (["time"], [101325.0, 102000.0, 103000.0], {"units": "Pa"}),
    },
    coords={"time": [0, 1, 2]},
)

# Quantify all variables at once
q_ds = ds.unxt.quantify()
print(q_ds["temperature"].data)
# Quantity['temperature'](Array([273., 293., 313.], dtype=float32), unit='K')
print(q_ds["pressure"].data)
# Quantity['pressure'](Array([101325., 102000., 103000.], dtype=float32), unit='Pa')
```

### Per-Variable Units

You can specify units for specific variables:

```python
ds = xr.Dataset(
    {
        "distance": (["x"], [1.0, 2.0, 3.0]),
        "velocity": (["x"], [10.0, 20.0, 30.0]),
    }
)

q_ds = ds.unxt.quantify(
    units={
        "distance": "m",
        "velocity": "m/s",
    }
)
```

## Dequantification

Converting back to plain arrays with unit metadata:

```python
import xarray as xr
import unxt as u
import unxt_xarray

# Start with quantified data
q = u.Quantity([1.0, 2.0, 3.0], "m")
da = xr.DataArray(q, dims=["x"])

# Convert to plain arrays with unit attributes
plain = da.unxt.dequantify()
print(plain.data)
# Array([1., 2., 3.], dtype=float32)
print(plain.attrs["units"])
# 'm'
```

The `unit_attribute` parameter controls the attribute name (default: `"units"`):

```python
plain = da.unxt.dequantify(unit_attribute="unit_str")
print(plain.attrs["unit_str"])
# 'm'
```

## JAX Integration

Since `unxt` uses JAX arrays, all JAX transformations work seamlessly:

### JIT Compilation

```python
import jax
import jax.numpy as jnp
import xarray as xr
import unxt as u
import unxt_xarray


@jax.jit
def process_data(da):
    """JIT-compiled function operating on DataArray."""
    return da * 2.0


# Create quantified DataArray
q = u.Quantity([1.0, 2.0, 3.0], "m")
da = xr.DataArray(q, dims=["x"])

# JIT works with the underlying data
result = process_data(da.data)
print(result)
# Quantity['length'](Array([2., 4., 6.], dtype=float32), unit='m')
```

### Vectorization

```python
import jax
import xarray as xr
import unxt as u


@jax.vmap
def square(x):
    return x**2


q = u.Quantity([[1.0, 2.0], [3.0, 4.0]], "m")
da = xr.DataArray(q, dims=["time", "space"])

# vmap over the data
squared = square(da.data)
print(squared)
# Quantity['area'](Array([[ 1.,  4.],
#        [ 9., 16.]], dtype=float32), unit='m2')
```

### Auto-differentiation

```python
import jax
import xarray as xr
import unxt as u


def kinetic_energy(v):
    """Kinetic energy: KE = 0.5 * m * v^2."""
    m = u.Quantity(2.0, "kg")
    return 0.5 * m * v**2


v = u.Quantity([1.0, 2.0, 3.0], "m/s")
da = xr.DataArray(v, dims=["time"])

# Gradient with respect to velocity
grad_fn = jax.grad(lambda v_val: jax.numpy.sum(kinetic_energy(v_val).value))
dKE_dv = grad_fn(da.data)
```

## Roundtrip Conversions

The quantify/dequantify operations are designed to roundtrip:

```python
import xarray as xr
import unxt as u
import unxt_xarray

# Start with attrs
original = xr.DataArray([1.0, 2.0], dims=["x"], attrs={"units": "m"})

# Roundtrip: attrs → Quantity → attrs
roundtrip = original.unxt.quantify().unxt.dequantify()

assert roundtrip.attrs["units"] == original.attrs["units"]
assert jnp.allclose(roundtrip.data, original.data)
```

## Best Practices

### 1. Import unxt_xarray Early

Always import `unxt_xarray` before using the `.unxt` accessor:

```python
import unxt_xarray  # Registers the accessor
```

This registers the accessor on `xarray`'s DataArray and Dataset classes.

### 2. Use Non-Dimension Coordinates for Units

When working with coordinates that need to preserve Quantities:

```python
from xarray import Variable

# ✓ Good: non-dimension coordinate
coords = {"i": [0, 1], "x": ("i", u.Quantity([1.0, 2.0], "m"))}

# ✗ Bad: dimension coordinate (xarray will extract values)
coords = {"x": u.Quantity([1.0, 2.0], "m")}  # x is marked as dimension
```

### 3. Consistent Unit Attributes

Use consistent attribute names throughout your workflow. The default `"units"`
is standard in many scientific data formats (CF conventions, NetCDF, etc.).

### 4. Preserve Other Metadata

The `quantify()` and `dequantify()` methods preserve all other attributes:

```python
da = xr.DataArray(
    [1.0, 2.0],
    dims=["x"],
    attrs={
        "units": "m",
        "long_name": "Distance",
        "standard_name": "distance",
    },
)

quantified = da.unxt.quantify()
# All non-unit attrs are preserved
assert quantified.attrs["long_name"] == "Distance"
```

## Common Patterns

### Loading from NetCDF

```python
import xarray as xr
import unxt_xarray
from pathlib import Path

# Load dataset with unit metadata
# Get the path to the sample data file relative to this document
docs_dir = Path("packages/unxt-xarray/docs")
data_path = docs_dir / "_data" / "sample_data.nc"

ds = xr.open_dataset(data_path)
print(ds)
# <xarray.Dataset> Size: 144B
# Dimensions:      (time: 2, location: 3)
# Coordinates:
#   * time         (time) float64 16B 0.0 3.6e+03
#   * location     (location) int64 24B 0 1 2
# Data variables:
#     temperature  (time, location) float64 48B 273.1 293.1 313.1 275.0 295.0 315.0
#     pressure     (time, location) float64 48B 1.013e+05 1.02e+05 ... 1.032e+05
#     distance     (location) float64 24B 0.0 100.0 200.0
# Attributes: (12/13)
#     ...

# Variables have unit metadata
print(ds["temperature"].attrs["units"])
# 'K'

# Convert all variables with units to Quantities
q_ds = ds.unxt.quantify()
print(q_ds["temperature"].data)
# Quantity(Array([[273.15, 293.15, 313.15],
#                 [275.  , 295.  , 315.  ]], dtype=float64), unit='K')
print(q_ds["pressure"].data)
# Quantity(Array([[101325., 102000., 103000.],
#                 [101500., 102200., 103200.]], dtype=float64), unit='Pa')
```

### Saving to NetCDF

```python
import xarray as xr
import unxt as u
import unxt_xarray

# Create a quantified dataset
q_ds = xr.Dataset(
    {
        "distance": (["x"], u.Quantity([1.0, 2.0, 3.0], "m")),
        "velocity": (["x"], u.Quantity([10.0, 20.0, 30.0], "m/s")),
    }
)

# Dequantify before saving
plain_ds = q_ds.unxt.dequantify()
print(plain_ds["distance"].attrs["units"])
# 'm'

# Save to file
# plain_ds.to_netcdf("output.nc")
```

### Unit Conversion

Use `unxt`'s conversion functions:

```python
import unxt as u
import xarray as xr

q = u.Quantity([1.0, 2.0, 3.0], "m")
da = xr.DataArray(q, dims=["x"])

# Convert to centimeters
cm_data = u.uconvert(u.unit("cm"), da.data)
da_cm = xr.DataArray(cm_data, dims=da.dims, coords=da.coords)
```

## Limitations

### Dimension Coordinates

`xarray`'s dimension coordinates (those marked with `*` in the string
representation) always extract the underlying array values, even for duck array
types like Quantity. This is a limitation of `xarray` itself.

**Workaround**: Use non-dimension coordinates:

```python
import unxt as u
import xarray as xr

data = [10.0, 20.0, 30.0]
quantities = u.Quantity([1.0, 2.0, 3.0], "m")

# Instead of this (dimension coordinate):
# da = xr.DataArray(data, dims=["x"], coords={"x": quantities})

# Do this (non-dimension coordinate):
da = xr.DataArray(data, dims=["i"], coords={"i": [0, 1, 2], "x": ("i", quantities)})
```

## See Also

- [unxt documentation](../../../index.md) - Core unitful quantities
- [xarray documentation](https://docs.xarray.dev/) - Labeled arrays
- [JAX documentation](https://jax.readthedocs.io/) - Composable transformations
- [Astropy units](https://docs.astropy.org/en/stable/units/) - Unit definitions
