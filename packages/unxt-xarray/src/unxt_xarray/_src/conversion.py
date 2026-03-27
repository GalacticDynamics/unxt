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

from plum import dispatch
from xarray import DataArray, Dataset, Variable

import unxt as u
from unxt.quantity import AllowValue

# Name of the attribute used to store units
UNIT_ATTR: Final = "units"


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


@dispatch
def extract_unit_attributes(obj: DataArray, /) -> dict[Hashable, str | None]:
    """Extract unit attributes from a DataArray.

    Parameters
    ----------
    obj : DataArray
        The DataArray to extract unit attributes from.

    Returns
    -------
    dict[Hashable, str | None]
        Mapping of variable names to their unit attribute values.
        The DataArray's own data units are stored under the ``None`` key.

    Examples
    --------
    >>> import xarray as xr
    >>> from unxt_xarray._src.conversion import extract_unit_attributes

    >>> da = xr.DataArray([1.0, 2.0], dims=["x"], attrs={"units": "m"})
    >>> extract_unit_attributes(da)
    {None: 'm'}

    """
    units: dict[Hashable, str | None] = {}

    # For DataArray, use None as the key for the data variable
    if (v := obj.attrs.get(UNIT_ATTR)) is not None:
        units[None] = v

    # Extract from coordinates
    for name, coord in obj.coords.items():
        if (v := coord.attrs.get(UNIT_ATTR)) is not None:
            units[name] = v

    return units


@dispatch
def extract_unit_attributes(obj: Dataset, /) -> dict[Hashable, str | None]:
    """Extract unit attributes from a Dataset.

    Parameters
    ----------
    obj : Dataset
        The Dataset to extract unit attributes from.

    Returns
    -------
    dict[Hashable, str | None]
        Mapping of variable names to their unit attribute values.

    Examples
    --------
    >>> import xarray as xr
    >>> from unxt_xarray._src.conversion import extract_unit_attributes

    >>> ds = xr.Dataset({"a": ("x", [1, 2], {"units": "m"}), "b": ("y", [3, 4])})
    >>> extract_unit_attributes(ds)
    {'a': 'm'}

    """
    units: dict[Hashable, str | None] = {}

    # Extract from all variables (data and coords)
    for name, var in obj.variables.items():
        if (v := var.attrs.get(UNIT_ATTR)) is not None:
            units[name] = v

    return units


@dispatch
def extract_unit_attributes(obj: object, /) -> dict:
    msg = f"Cannot extract unit attributes from type: {type(obj)}"
    raise TypeError(msg)


@dispatch
def extract_units(obj: DataArray, /) -> dict[Hashable, u.AbstractUnit | None]:
    """Extract units from Quantities in a DataArray.

    Parameters
    ----------
    obj : DataArray
        The DataArray to extract units from.

    Returns
    -------
    dict[Hashable, AbstractUnit | None]
        Mapping of variable names to their units.
        The DataArray's own data units are stored under the ``None`` key.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import xarray as xr
    >>> import unxt as u
    >>> from unxt_xarray._src.conversion import extract_units

    >>> q = u.Quantity([1.0, 2.0], "m")
    >>> da = xr.DataArray(q, dims=["x"])
    >>> units = extract_units(da)
    >>> units[None]
    Unit("m")

    """
    units: dict[Hashable, u.AbstractUnit | None] = {}

    # Extract from data
    if (unit := u.unit_of(obj.data)) is not None:
        units[None] = unit

    # Extract from coordinates
    for name, coord in obj.coords.items():
        if (unit := u.unit_of(coord.data)) is not None:
            units[name] = unit

    return units


@dispatch
def extract_units(obj: Dataset, /) -> dict[Hashable, u.AbstractUnit | None]:
    """Extract units from Quantities in a Dataset.

    Parameters
    ----------
    obj : Dataset
        The Dataset to extract units from.

    Returns
    -------
    dict[Hashable, AbstractUnit | None]
        Mapping of variable names to their units.

    """
    units: dict[Hashable, u.AbstractUnit | None] = {}

    # Extract from all variables
    for name, var in obj.variables.items():
        if (unit := u.unit_of(var.data)) is not None:
            units[name] = unit

    return units


@dispatch
def extract_units(obj: object, /) -> dict:
    msg = f"Cannot extract units from type: {type(obj)}"
    raise TypeError(msg)


@dispatch
def attach_units(
    obj: DataArray, units: Mapping[Hashable, u.AbstractUnit | str | None]
) -> DataArray:
    """Attach units to a DataArray's variables.

    Parameters
    ----------
    obj : DataArray
        The DataArray to attach units to.
    units : Mapping[Hashable, AbstractUnit | str | None]
        Mapping of variable names to units.
        Use ``None`` as the key to attach units to the DataArray's own data.

    Returns
    -------
    DataArray
        New DataArray with units attached as Quantities.

    Examples
    --------
    >>> import xarray as xr
    >>> from unxt_xarray._src.conversion import attach_units

    >>> da = xr.DataArray([1.0, 2.0], dims=["x"])
    >>> quantified = attach_units(da, {None: "m"})
    >>> quantified.data
    Quantity(Array([1., 2.], dtype=float32), unit='m')

    """
    # Handle the data array itself (None key = the DataArray's own data)
    data_unit = units.get(None)
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


@dispatch
def attach_units(
    obj: Dataset, units: Mapping[Hashable, u.AbstractUnit | str | None]
) -> Dataset:
    """Attach units to a Dataset's variables.

    Parameters
    ----------
    obj : Dataset
        The Dataset to attach units to.
    units : Mapping[Hashable, AbstractUnit | str | None]
        Mapping of variable names to units.

    Returns
    -------
    Dataset
        New Dataset with units attached as Quantities.

    """
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


@dispatch
def attach_units(obj: object, units: Mapping) -> object:
    msg = f"Cannot attach units to type: {type(obj)}"
    raise TypeError(msg)


@dispatch
def strip_units(obj: DataArray) -> DataArray:
    """Strip units from a DataArray, converting Quantities to plain arrays.

    Parameters
    ----------
    obj : DataArray
        The DataArray to strip units from.

    Returns
    -------
    DataArray
        New DataArray with plain arrays instead of Quantities.

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


@dispatch
def strip_units(obj: Dataset) -> Dataset:
    """Strip units from a Dataset, converting Quantities to plain arrays.

    Parameters
    ----------
    obj : Dataset
        The Dataset to strip units from.

    Returns
    -------
    Dataset
        New Dataset with plain arrays instead of Quantities.

    """
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


@dispatch
def strip_units(obj: object) -> object:
    msg = f"Cannot strip units from type: {type(obj)}"
    raise TypeError(msg)
