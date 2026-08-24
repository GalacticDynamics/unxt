# `unxts.interop.xarray`

```{toctree}
:maxdepth: 1
:hidden:

tutorial-first-dataset
xarray-guide
api
sharp-bits
```

`unxts.interop.xarray` is the canonical location for [xarray](https://docs.xarray.dev/) integration. It adds a `.unxt` accessor to `DataArray` and `Dataset` that converts between `xarray`'s unit _metadata_ — a `units` string in `.attrs` — and real `unxt.Quantity` values, so units become part of the data rather than a label beside it.

Importing the package registers the accessor as an import side effect.

## Install

The recommended install adds `unxts.interop.xarray` alongside `unxt` via the `interop-xarray` [extra](https://peps.python.org/pep-0508/#extras), so it, `unxt` and `xarray` are resolved together as a compatible set:

::::{tab-set}

:::{tab-item} uv

```bash
uv add "unxt[interop-xarray]"
```

:::

:::{tab-item} pip

```bash
pip install "unxt[interop-xarray]"
```

:::

::::

Or install the package directly:

::::{tab-set}

:::{tab-item} uv

```bash
uv add unxts.interop.xarray
```

:::

:::{tab-item} pip

```bash
pip install unxts.interop.xarray
```

:::

::::

## At a glance

```python
import numpy as np
import xarray as xr

import unxts.interop.xarray  # registers the .unxt accessor

da = xr.DataArray(np.array([1.0, 2.0, 3.0]), dims="x", attrs={"units": "m"})

q = da.unxt.quantify()
q.data
# Quantity(Array([1., 2., 3.], dtype=float32), unit='m')

plain = q.unxt.dequantify()
plain.attrs["units"]
# 'm'
```

`quantify()` reads the `units` attribute and turns the values into a `Quantity`; `dequantify()` reverses it, putting the unit back in `.attrs`.

## Pages

**Tutorial**

- [Turn unit labels into real units](tutorial-first-dataset) — start here: watch `xarray` add kilometres to hours without complaint, then quantify and watch the same operation refuse.

**How-to**

- [How to use unxt with xarray](xarray-guide) — quantifying and dequantifying `DataArray`s and `Dataset`s, coordinates, unit conversion, JAX transforms, and round-tripping through NetCDF.

**Reference**

- [API](api) — the functions underneath the accessor: `extract_unit_attributes`, `attach_units`, `extract_units`, `strip_units`.

**Discussion**

- [The xarray sharp bits](sharp-bits) — why dimension coordinates cannot hold quantities, and which operations drop units.

## See also

- [xarray documentation](https://docs.xarray.dev/)
- [unxt documentation](https://unxt.readthedocs.io/)
