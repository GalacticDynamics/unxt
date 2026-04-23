# unxt-xarray

xarray integration for [unxt](https://github.com/GalacticDynamics/unxt).

This package provides xarray accessors that enable seamless integration between unxt's JAX-based quantities and xarray's labeled multi-dimensional arrays.

## Installation

```bash
pip install unxt-xarray
```

or with uv:

```bash
uv add unxt-xarray
```

## Quick Start

```python
import xarray as xr
import unxt as u
import unxt_xarray  # registers the .unxt accessor

# Create a DataArray with unit attributes
da = xr.DataArray(
    data=[1.0, 2.0, 3.0],
    dims=["x"],
    coords={"x": ("x", [0.0, 1.0, 2.0], {"units": "s"})},
    attrs={"units": "m"},
)

# Quantify: convert to unxt Quantities
quantified = da.unxt.quantify()
print(quantified.data)
# Quantity(Array([1., 2., 3.], dtype=float32), unit='m')

# Dequantify: convert back to plain arrays with unit attributes
dequantified = quantified.unxt.dequantify()
print(dequantified.attrs)
# {'units': 'm'}
```

## Features

- **`quantify()`**: Convert xarray objects with unit attributes to unxt Quantities
- **`dequantify()`**: Convert unxt Quantities back to plain arrays with unit attributes
- Supports both `DataArray` and `Dataset` objects
- Preserves coordinates and their units
- JAX-compatible for JIT compilation, vectorization, and differentiation

## Documentation

For full documentation, see the [main unxt documentation](https://unxt.readthedocs.io/).
