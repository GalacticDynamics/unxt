"""xarray accessors for unxt integration.

Copyright (c) 2023 Galactic Dynamics. All rights reserved.
"""

__all__ = ("UnxtDataArrayAccessor", "UnxtDatasetAccessor")

from collections.abc import Hashable, Mapping

from xarray import (
    DataArray,
    Dataset,
    register_dataarray_accessor,
    register_dataset_accessor,
)

import unxt as u
from .conversion import (
    TEMPORARY_NAME,
    UNIT_ATTR,
    attach_units,
    extract_unit_attributes,
    extract_units,
    strip_units,
)


@register_dataarray_accessor("unxt")
class UnxtDataArrayAccessor:
    """Access methods for DataArrays with unxt units.

    Methods and attributes can be accessed through the `.unxt` attribute.

    Examples
    --------
    >>> import xarray as xr
    >>> import unxt as u
    >>> import unxt_xarray  # registers the accessor

    >>> da = xr.DataArray([1.0, 2.0, 3.0], dims=["x"], attrs={"units": "m"})
    >>> q = da.unxt.quantify()
    >>> q.data
    Quantity(Array([1., 2., 3.], dtype=float32), unit='m')

    """

    def __init__(self, da: DataArray) -> None:
        self._da = da

    def quantify(
        self,
        units: str
        | u.AbstractUnit
        | Mapping[Hashable, str | u.AbstractUnit | None]
        | None = None,
        **unit_kwargs: str | u.AbstractUnit | None,
    ) -> DataArray:
        """Attach units to the DataArray.

        Units can be specified as a string, unit object, or mapping of coordinate
        names to units. If not specified, units are read from the "units" attribute.

        Parameters
        ----------
        units : str | AbstractUnit | Mapping | None, optional
            Units to attach. Can be:
            - A string or unit object to apply to the data array
            - A dict-like mapping coordinate/variable names to units
            - None to use the "units" attribute
        **unit_kwargs
            Keyword form of units for coordinate names.

        Returns
        -------
        DataArray
            DataArray with data and coordinates as unxt Quantities.

        Examples
        --------
        >>> import xarray as xr
        >>> import unxt_xarray

        Quantify using attribute:

        >>> da = xr.DataArray([1.0, 2.0], dims=["x"], attrs={"units": "m"})
        >>> q = da.unxt.quantify()
        >>> q.data
        Quantity(Array([1., 2.], dtype=float32), unit='m')

        Quantify with explicit unit:

        >>> da = xr.DataArray([1.0, 2.0], dims=["x"])
        >>> q = da.unxt.quantify("km")
        >>> q.data
        Quantity(Array([1., 2.], dtype=float32), unit='km')

        Quantify with coordinate units:

        >>> da = xr.DataArray(
        ...     [1.0, 2.0],
        ...     dims=["x"],
        ...     coords={"x": [0.0, 1.0]},
        ...     attrs={"units": "m"},
        ... )
        >>> da.coords["x"].attrs["units"] = "s"
        >>> q = da.unxt.quantify()
        >>> q.data
        Quantity(Array([1., 2.], dtype=float32), unit='m')
        >>> q.coords["x"].data  # Note: dimension coordinates are unwrapped
        array([0., 1.], dtype=float32)

        """
        # Combine explicit units with unit_kwargs
        if units is None:
            units = unit_kwargs
        elif isinstance(units, (str, u.AbstractUnit)):
            # Single unit for the data array
            combined_units = {TEMPORARY_NAME: units}
            combined_units.update(unit_kwargs)
            units = combined_units
        elif isinstance(units, Mapping):
            combined_units = dict(units)
            combined_units.update(unit_kwargs)
            units = combined_units
        else:
            msg = f"units must be a string, AbstractUnit, or Mapping, got {type(units)}"
            raise TypeError(msg)

        # Extract unit attributes if no explicit units provided
        unit_attrs = extract_unit_attributes(self._da)

        # Merge: explicit units override attributes
        final_units: dict[Hashable, str | u.AbstractUnit | None] = {}
        for name in set(list(unit_attrs.keys()) + list(units.keys())):
            if name in units:
                final_units[name] = units[name]
            elif name in unit_attrs:
                final_units[name] = unit_attrs[name]

        # Attach units
        return attach_units(self._da, final_units)

    def dequantify(
        self, format: str | None = None, unit_attribute: str = UNIT_ATTR
    ) -> DataArray:
        """Convert quantities back to plain arrays with unit attributes.

        Parameters
        ----------
        format : str | None, optional
            Format string for unit representation. If None, uses default string format.
        unit_attribute : str, optional
            Name of attribute to store units in. Default is "units".

        Returns
        -------
        DataArray
            DataArray with plain arrays and unit attributes.

        Examples
        --------
        >>> import xarray as xr
        >>> import unxt as u
        >>> import unxt_xarray

        >>> q = u.Quantity([1.0, 2.0], "m")
        >>> da = xr.DataArray(q, dims=["x"])
        >>> dequantified = da.unxt.dequantify()
        >>> dequantified.data
        Array([1., 2.], dtype=float32)
        >>> dequantified.attrs["units"]
        'm'

        """
        # Extract units from quantities
        units = extract_units(self._da)

        # Strip quantities, converting to plain arrays
        stripped = strip_units(self._da)

        # Format units as strings for attributes
        unit_strs: dict[Hashable, str] = {}
        for name, unit in units.items():
            if unit is not None:
                if format is not None:
                    unit_strs[name] = format.format(unit)
                else:
                    unit_strs[name] = str(unit)

        # Add unit attributes
        if TEMPORARY_NAME in unit_strs:
            stripped.attrs[unit_attribute] = unit_strs[TEMPORARY_NAME]

        for name, unit_str in unit_strs.items():
            if name == TEMPORARY_NAME:
                continue
            if name in stripped.coords:
                stripped.coords[name].attrs[unit_attribute] = unit_str

        return stripped


@register_dataset_accessor("unxt")
class UnxtDatasetAccessor:
    """Access methods for Datasets with unxt units.

    Methods and attributes can be accessed through the `.unxt` attribute.

    Examples
    --------
    >>> import xarray as xr
    >>> import unxt_xarray

    >>> ds = xr.Dataset(
    ...     {
    ...         "a": ("x", [1.0, 2.0], {"units": "m"}),
    ...         "b": ("y", [3.0, 4.0], {"units": "s"}),
    ...     }
    ... )
    >>> q = ds.unxt.quantify()
    >>> q["a"].data
    Quantity(Array([1., 2.], dtype=float32), unit='m')
    >>> q["b"].data
    Quantity(Array([3., 4.], dtype=float32), unit='s')

    """

    def __init__(self, ds: Dataset, /) -> None:
        self._ds = ds

    def quantify(
        self,
        units: Mapping[Hashable, str | u.AbstractUnit | None] | None = None,
        **unit_kwargs: str | u.AbstractUnit | None,
    ) -> Dataset:
        """Attach units to the Dataset.

        Units can be specified as a mapping of variable names to units.
        If not specified, units are read from the "units" attribute of each variable.

        Parameters
        ----------
        units : Mapping | None, optional
            Mapping of variable/coordinate names to units.
            If None, uses the "units" attribute.
        **unit_kwargs
            Keyword form of units for variable names.

        Returns
        -------
        Dataset
            Dataset with variables and coordinates as unxt Quantities.

        Examples
        --------
        >>> import xarray as xr
        >>> import unxt_xarray

        Quantify using attributes:

        >>> ds = xr.Dataset(
        ...     {
        ...         "a": ("x", [1.0, 2.0], {"units": "m"}),
        ...         "b": ("y", [3.0, 4.0], {"units": "s"}),
        ...     }
        ... )
        >>> q = ds.unxt.quantify()
        >>> q["a"].data
        Quantity(Array([1., 2.], dtype=float32), unit='m')

        Quantify with explicit units:

        >>> ds = xr.Dataset({"a": ("x", [1.0, 2.0])})
        >>> q = ds.unxt.quantify(a="km")
        >>> q["a"].data
        Quantity(Array([1., 2.], dtype=float32), unit='km')

        """
        # Combine explicit units with unit_kwargs
        if units is None:
            units = unit_kwargs
        else:
            combined_units = dict(units)
            combined_units.update(unit_kwargs)
            units = combined_units

        # Extract unit attributes
        unit_attrs = extract_unit_attributes(self._ds)

        # Merge: explicit units override attributes
        final_units: dict[Hashable, str | u.AbstractUnit | None] = {}
        for name in set(list(unit_attrs.keys()) + list(units.keys())):
            if name in units:
                final_units[name] = units[name]
            elif name in unit_attrs:
                final_units[name] = unit_attrs[name]

        # Attach units
        return attach_units(self._ds, final_units)

    def dequantify(
        self, format: str | None = None, unit_attribute: str = UNIT_ATTR
    ) -> Dataset:
        """Convert quantities back to plain arrays with unit attributes.

        Parameters
        ----------
        format : str | None, optional
            Format string for unit representation. If None, uses default string format.
        unit_attribute : str, optional
            Name of attribute to store units in. Default is "units".

        Returns
        -------
        Dataset
            Dataset with plain arrays and unit attributes.

        Examples
        --------
        >>> import xarray as xr
        >>> import unxt as u
        >>> import unxt_xarray

        >>> q1 = u.Quantity([1.0, 2.0], "m")
        >>> q2 = u.Quantity([3.0, 4.0], "s")
        >>> ds = xr.Dataset({"a": ("x", q1), "b": ("y", q2)})
        >>> dequantified = ds.unxt.dequantify()
        >>> dequantified["a"].attrs["units"]
        'm'
        >>> dequantified["b"].attrs["units"]
        's'

        """
        # Extract units from quantities
        units = extract_units(self._ds)

        # Strip quantities, converting to plain arrays
        stripped = strip_units(self._ds)

        # Format units as strings for attributes
        unit_strs: dict[Hashable, str] = {}
        for name, unit in units.items():
            if unit is not None:
                if format is not None:
                    unit_strs[name] = format.format(unit)
                else:
                    unit_strs[name] = str(unit)

        # Add unit attributes to variables
        for name, unit_str in unit_strs.items():
            if name in stripped.data_vars:
                stripped[name].attrs[unit_attribute] = unit_str
            elif name in stripped.coords:
                stripped.coords[name].attrs[unit_attribute] = unit_str

        return stripped
