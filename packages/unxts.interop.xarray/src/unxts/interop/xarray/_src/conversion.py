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
    >>> from unxts.interop.xarray._src.conversion import _array_attach_units

    >>> data = jnp.array([1.0, 2.0, 3.0])
    >>> q = _array_attach_units(data, "m")
    >>> q
    Quantity(Array([1., 2., 3.], dtype=float32), unit='m')

    >>> _array_attach_units(data, None) is data
    True

    Data that is already a Quantity cannot be re-quantified: attaching a unit
    to it raises `ValueError` naming the unit the data already carries, rather
    than the opaque "Cannot convert 'Quantity' to a value" `TypeError` that
    `unxt.Quantity` would otherwise raise.

    """
    if unit is None:
        return data

    if (existing := u.unit_of(data)) is not None:
        msg = (
            f"cannot attach unit '{unit}': the data is already a Quantity with "
            f"unit '{existing}'. Convert it with `unxt.uconvert` instead of "
            "re-quantifying."
        )
        raise ValueError(msg)

    return u.Q(data, u.unit(unit))


def _consume_unit_attrs(obj: DataArray | Dataset, /) -> None:
    """Drop the unit attribute wherever the data itself now carries the unit.

    Once a unit lives on the data as a `Quantity`, the ``units`` attribute is
    a stale duplicate: a later conversion would leave it contradicting the data
    (so plot labels and CF serialization report the pre-conversion unit), and it
    would make a second ``quantify()`` try to re-quantify a `Quantity`.

    The attribute is dropped only where the unit demonstrably survived onto the
    data. xarray coerces *dimension* coordinates back to plain index arrays,
    discarding the Quantity; there the attribute remains the only record of the
    unit and must be kept. Mutates ``obj`` in place -- callers pass an object
    they have just constructed.
    """

    def drop_if_carried(node: Any, /) -> None:
        """Drop the attr from one variable/coordinate if its data carries a unit."""
        if u.unit_of(node.data) is not None:
            node.attrs.pop(UNIT_ATTR, None)

    # A Dataset holds many data variables; a DataArray *is* the single one.
    # Branching explicitly (rather than unifying into one list) keeps the two
    # element types apart, so this needs no ``type: ignore``.
    if isinstance(obj, Dataset):
        for var in obj.data_vars.values():
            drop_if_carried(var)
    else:
        drop_if_carried(obj)

    for coord in obj.coords.values():
        drop_if_carried(coord)


@dispatch
def extract_unit_attributes(obj: DataArray, /) -> dict[Hashable, u.AbstractUnit | None]:
    """Extract unit attributes from a DataArray.

    Parameters
    ----------
    obj : DataArray
        The DataArray to extract unit attributes from.

    Returns
    -------
    dict[Hashable, AbstractUnit | None]
        Mapping of variable names to normalized unit objects.
        The DataArray's own data units are stored under the ``None`` key.

    Examples
    --------
    >>> import xarray as xr
    >>> from unxts.interop.xarray._src.conversion import extract_unit_attributes

    >>> da = xr.DataArray([1.0, 2.0], dims=["x"], attrs={"units": "m"})
    >>> extract_unit_attributes(da)
    {None: Unit("m")}

    """
    units: dict[Hashable, u.AbstractUnit | None] = {}

    # For DataArray, use None as the key for the data variable
    if (v := obj.attrs.get(UNIT_ATTR)) is not None:
        units[None] = u.unit(v)

    # Extract from coordinates
    for name, coord in obj.coords.items():
        if (v := coord.attrs.get(UNIT_ATTR)) is not None:
            units[name] = u.unit(v)

    return units


@dispatch
def extract_unit_attributes(obj: Dataset, /) -> dict[Hashable, u.AbstractUnit | None]:
    """Extract unit attributes from a Dataset.

    Parameters
    ----------
    obj : Dataset
        The Dataset to extract unit attributes from.

    Returns
    -------
    dict[Hashable, AbstractUnit | None]
        Mapping of variable names to normalized unit objects.

    Examples
    --------
    >>> import xarray as xr
    >>> from unxts.interop.xarray._src.conversion import extract_unit_attributes

    >>> ds = xr.Dataset({"a": ("x", [1, 2], {"units": "m"}), "b": ("y", [3, 4])})
    >>> extract_unit_attributes(ds)
    {'a': Unit("m")}

    """
    units: dict[Hashable, u.AbstractUnit | None] = {}

    # Extract from all variables (data and coords)
    for name, var in obj.variables.items():
        if (v := var.attrs.get(UNIT_ATTR)) is not None:
            units[name] = u.unit(v)

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
    >>> from unxts.interop.xarray._src.conversion import extract_units

    >>> q = u.Q([1.0, 2.0], "m")
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


def _reject_unknown_unit_names(
    units: "Mapping[Hashable, Any]", valid_names: "set[Any]", obj_kind: str
) -> None:
    """Raise if units names something that is not a variable/coordinate.

    Without this, a typo in a **unit_kwargs name (e.g.
    quantify(temperatrue="K")) is silently dropped by units.get(name),
    leaving the data unquantified with no error or warning.
    """
    unknown = [repr(k) for k in units if k not in valid_names]
    if not unknown:
        return
    # ``None`` is the DataArray's own-data key; render it explicitly so the
    # message stays helpful (and non-empty) when there are no other names.
    names = ("None (the data)" if n is None else repr(n) for n in valid_names)
    valid = ", ".join(sorted(names)) or "(none)"
    msg = (
        f"got unit(s) for name(s) not in the {obj_kind}: {', '.join(unknown)}. "
        f"Valid names: {valid}."
    )
    raise ValueError(msg)


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
    >>> from unxts.interop.xarray._src.conversion import attach_units

    >>> da = xr.DataArray([1.0, 2.0], dims=["x"])
    >>> quantified = attach_units(da, {None: "m"})
    >>> quantified.data
    Quantity(Array([1., 2.], dtype=float32), unit='m')

    The consumed ``units`` attribute does not survive onto the result:

    >>> da = xr.DataArray([1.0, 2.0], dims=["x"], attrs={"units": "m"})
    >>> attach_units(da, {None: "m"}).attrs
    {}

    """
    _reject_unknown_unit_names(units, {None} | set(obj.coords), "DataArray")

    # Handle the data array itself (None key = the DataArray's own data)
    data_unit = units.get(None)
    new_data = _array_attach_units(obj.data, data_unit)

    # Handle coordinates - need to preserve as Variable objects
    new_coords = {}
    for name, coord in obj.coords.items():
        unit = units.get(name)
        # Always build a fresh Variable with a copied attrs dict, even when no
        # unit is attached. ``_consume_unit_attrs`` mutates the result's attrs
        # in place, so reusing ``coord`` would be safe only for as long as the
        # DataArray/Dataset constructor keeps copying coordinate attrs -- an
        # xarray implementation detail, not a guarantee. Copying here makes
        # non-mutation of the caller's object structural instead.
        new_coords[name] = Variable(
            coord.dims,
            coord.data if unit is None else _array_attach_units(coord.data, unit),
            dict(coord.attrs),
        )

    result = DataArray(
        data=new_data,
        coords=new_coords,
        dims=obj.dims,
        name=obj.name,
        attrs=dict(obj.attrs),
    )
    _consume_unit_attrs(result)
    return result


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
    _reject_unknown_unit_names(units, set(obj.data_vars) | set(obj.coords), "Dataset")

    # Handle all variables in dataset
    new_vars = {}
    for name, var in obj.data_vars.items():
        new_data = _array_attach_units(var.data, units.get(name))
        new_vars[name] = (var.dims, new_data, dict(var.attrs))

    # Handle coordinates
    new_coords = {}
    for name, coord in obj.coords.items():
        unit = units.get(name)
        # Always build a fresh Variable with a copied attrs dict, even when no
        # unit is attached. ``_consume_unit_attrs`` mutates the result's attrs
        # in place, so reusing ``coord`` would be safe only for as long as the
        # DataArray/Dataset constructor keeps copying coordinate attrs -- an
        # xarray implementation detail, not a guarantee. Copying here makes
        # non-mutation of the caller's object structural instead.
        new_coords[name] = Variable(
            coord.dims,
            coord.data if unit is None else _array_attach_units(coord.data, unit),
            dict(coord.attrs),
        )

    result = Dataset(data_vars=new_vars, coords=new_coords, attrs=dict(obj.attrs))
    _consume_unit_attrs(result)
    return result


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
    >>> from unxts.interop.xarray._src.conversion import strip_units

    >>> q = u.Q([1.0, 2.0], "m")
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
        new_coords[name] = Variable(coord.dims, new_coord_data, dict(coord.attrs))

    return DataArray(
        data=new_data,
        coords=new_coords,
        dims=obj.dims,
        name=obj.name,
        attrs=dict(obj.attrs),
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
        new_vars[name] = (var.dims, new_data, dict(var.attrs))

    # Strip units from coordinates
    new_coords = {}
    for name, coord in obj.coords.items():
        new_coord_data = u.ustrip(AllowValue, coord.data)
        new_coords[name] = Variable(coord.dims, new_coord_data, dict(coord.attrs))

    return Dataset(data_vars=new_vars, coords=new_coords, attrs=dict(obj.attrs))


@dispatch
def strip_units(obj: object) -> object:
    msg = f"Cannot strip units from type: {type(obj)}"
    raise TypeError(msg)
