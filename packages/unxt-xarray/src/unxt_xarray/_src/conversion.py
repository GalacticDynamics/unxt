"""Unit conversion functions for xarray objects.

Copyright (c) 2023 Galactic Dynamics. All rights reserved.
"""

__all__ = (
    "attach_units",
    "extract_units",
    "extract_unit_attributes",
    "strip_units",
)

from collections.abc import Hashable, Mapping
from typing import Any, Final

from xarray import DataArray, Dataset, Variable

import unxt as u
from unxt.quantity import AllowValue

# Name of the attribute used to store units
UNIT_ATTR: Final = "units"

# Sentinel for temporary name in conversion
TEMPORARY_NAME: Final = "<this-array>"


def _array_attach_units(
    data: Any, /, unit: u.AbstractUnit | str | None
) -> u.Quantity | Any:
    """Attach units to array data.

    Parameters
    ----------
    data : array-like
        The data to attach units to.
    unit : AbstractUnit | str | None
        The unit to attach. If None, returns data unchanged.

    Returns
    -------
    Quantity | array-like
        If unit is not None, returns a Quantity. Otherwise returns data unchanged.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> from unxt_xarray._src.conversion import _array_attach_units

    >>> data = jnp.array([1.0, 2.0, 3.0])
    >>> q = _array_attach_units(data, "m")
    >>> q
    Quantity(Array([1., 2., 3.], dtype=float32), unit='m')

    >>> _array_attach_units(data, None) is data
    True

    """
    return data if unit is None else u.Q(data, u.unit(unit))


def extract_unit_attributes(obj: DataArray | Dataset, /) -> dict[Hashable, str | None]:
    """Extract unit attributes from xarray object.

    Parameters
    ----------
    obj : DataArray | Dataset
        The xarray object to extract unit attributes from.

    Returns
    -------
    dict[Hashable, str | None]
        Mapping of variable names to their unit attribute values.

    Examples
    --------
    >>> import xarray as xr
    >>> from unxt_xarray._src.conversion import extract_unit_attributes

    >>> da = xr.DataArray([1.0, 2.0], dims=["x"], attrs={"units": "m"})
    >>> extract_unit_attributes(da)
    {'<this-array>': 'm'}

    >>> ds = xr.Dataset({"a": ("x", [1, 2], {"units": "m"}), "b": ("y", [3, 4])})
    >>> extract_unit_attributes(ds)
    {'a': 'm'}

    """
    units: dict[Hashable, str | None] = {}

    if isinstance(obj, DataArray):
        # For DataArray, use temporary name for the data variable
        if (v := obj.attrs.get(UNIT_ATTR)) is not None:
            units[TEMPORARY_NAME] = v

        # Extract from coordinates
        for name, coord in obj.coords.items():
            if (v := coord.attrs.get(UNIT_ATTR)) is not None:
                units[name] = v

    elif isinstance(obj, Dataset):
        # Extract from all variables (data and coords)
        for name, var in obj.variables.items():
            if (v := var.attrs.get(UNIT_ATTR)) is not None:
                units[name] = v

    else:
        msg = f"Cannot extract unit attributes from type: {type(obj)}"
        raise TypeError(msg)

    return units


def extract_units(obj: DataArray | Dataset, /) -> dict[Hashable, u.AbstractUnit | None]:
    """Extract units from Quantities in xarray object.

    Parameters
    ----------
    obj : DataArray | Dataset
        The xarray object to extract units from.

    Returns
    -------
    dict[Hashable, AbstractUnit | None]
        Mapping of variable names to their units.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import xarray as xr
    >>> import unxt as u
    >>> from unxt_xarray._src.conversion import extract_units

    >>> q = u.Quantity([1.0, 2.0], "m")
    >>> da = xr.DataArray(q, dims=["x"])
    >>> units = extract_units(da)
    >>> units["<this-array>"]
    Unit("m")

    """
    units: dict[Hashable, u.AbstractUnit | None] = {}

    if isinstance(obj, DataArray):
        # Extract from data
        if (unit := u.unit_of(obj.data)) is not None:
            units[TEMPORARY_NAME] = unit

        # Extract from coordinates
        for name, coord in obj.coords.items():
            if (unit := u.unit_of(coord.data)) is not None:
                units[name] = unit

    elif isinstance(obj, Dataset):
        # Extract from all variables
        for name, var in obj.variables.items():
            if (unit := u.unit_of(var.data)) is not None:
                units[name] = unit

    else:
        msg = f"Cannot extract units from type: {type(obj)}"
        raise TypeError(msg)

    return units


def attach_units(
    obj: DataArray | Dataset, units: Mapping[Hashable, u.AbstractUnit | str | None]
) -> DataArray | Dataset:
    """Attach units to xarray object variables.

    Parameters
    ----------
    obj : DataArray | Dataset
        The xarray object to attach units to.
    units : Mapping[Hashable, AbstractUnit | str | None]
        Mapping of variable names to units.

    Returns
    -------
    DataArray | Dataset
        New xarray object with units attached as Quantities.

    Examples
    --------
    >>> import xarray as xr
    >>> from unxt_xarray._src.conversion import attach_units

    >>> da = xr.DataArray([1.0, 2.0], dims=["x"])
    >>> quantified = attach_units(da, {"<this-array>": "m"})
    >>> quantified.data
    Quantity(Array([1., 2.], dtype=float32), unit='m')

    """
    if isinstance(obj, DataArray):
        # Handle the data array itself
        data_unit = units.get(TEMPORARY_NAME)
        new_data = _array_attach_units(obj.data, data_unit)

        # Handle coordinates - need to preserve as Variable objects
        new_coords = {}
        for name, coord in obj.coords.items():
            unit = units.get(name)
            new_coords[name] = (
                coord
                if unit is None
                else Variable(coord.dims, u.Q(coord.data, unit), coord.attrs)
            )

        return DataArray(
            data=new_data,
            coords=new_coords,
            dims=obj.dims,
            name=obj.name,
            attrs=obj.attrs,
        )

    if isinstance(obj, Dataset):
        # Handle all variables in dataset
        new_vars = {}
        for name, var in obj.data_vars.items():
            new_data = _array_attach_units(var.data, units.get(name))
            new_vars[name] = (var.dims, new_data, var.attrs)

        # Handle coordinates
        new_coords = {}
        for name, coord in obj.coords.items():
            unit = units.get(name)
            new_coords[name] = (
                coord
                if unit is None
                else Variable(coord.dims, u.Q(coord.data, unit), coord.attrs)
            )

        return Dataset(data_vars=new_vars, coords=new_coords, attrs=obj.attrs)

    msg = f"Cannot attach units to type: {type(obj)}"
    raise TypeError(msg)


def strip_units(obj: DataArray | Dataset) -> DataArray | Dataset:
    """Strip units from xarray object, converting Quantities to plain arrays.

    Parameters
    ----------
    obj : DataArray | Dataset
        The xarray object to strip units from.

    Returns
    -------
    DataArray | Dataset
        New xarray object with plain arrays instead of Quantities.

    Examples
    --------
    >>> import xarray as xr
    >>> import unxt as u
    >>> from unxt_xarray._src.conversion import strip_units

    >>> q = u.Quantity([1.0, 2.0], "m")
    >>> da = xr.DataArray(q, dims=["x"])
    >>> stripped = strip_units(da)
    >>> stripped.data
    Array([1., 2.], dtype=float32)

    """
    if isinstance(obj, DataArray):
        # Strip units from data
        new_data = u.ustrip(AllowValue, obj.data)

        # Strip units from coordinates
        new_coords = {}
        for name, coord in obj.coords.items():
            new_coord_data = u.ustrip(AllowValue, coord.data)
            new_coords[name] = Variable(coord.dims, new_coord_data, coord.attrs)

        return DataArray(
            data=new_data,
            coords=new_coords,
            dims=obj.dims,
            name=obj.name,
            attrs=obj.attrs,
        )

    if isinstance(obj, Dataset):
        # Strip units from all variables
        new_vars = {}
        for name, var in obj.data_vars.items():
            new_data = u.ustrip(AllowValue, var.data)
            new_vars[name] = (var.dims, new_data, var.attrs)

        # Strip units from coordinates
        new_coords = {}
        for name, coord in obj.coords.items():
            new_coord_data = u.ustrip(AllowValue, coord.data)
            new_coords[name] = Variable(coord.dims, new_coord_data, coord.attrs)

        return Dataset(data_vars=new_vars, coords=new_coords, attrs=obj.attrs)

    msg = f"Cannot strip units from type: {type(obj)}"
    raise TypeError(msg)
