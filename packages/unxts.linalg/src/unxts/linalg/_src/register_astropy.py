"""Astropy conversions for the linalg containers (registered on import).

`UnitsMatrix` and `astropy.units.StructuredUnit` are the same idea in two
libraries -- a nested layout of units -- and `QuantityMatrix` is that layout
with values attached, which astropy spells as a `Quantity` over a structured
dtype. So all three conversions are relabelings, not computations.

``astropy`` is a hard dependency of ``unxt``, which ``unxts.linalg`` requires,
so these are always registered.
"""

__all__: tuple[str, ...] = ()

from typing import Any

import astropy.units as apyu
import numpy as np
from plum import conversion_method

from ._quantity_matrix import QuantityMatrix
from ._units_matrix import UnitsMatrix


def _structured_unit_to_tuple(obj: apyu.StructuredUnit, /) -> tuple:
    """Convert an astropy ``StructuredUnit`` to a nested tuple of units."""
    return tuple(
        _structured_unit_to_tuple(v) if isinstance(v, apyu.StructuredUnit) else v
        for v in obj.values()
    )


def _structured_depth(unit: apyu.StructuredUnit, /) -> int:
    """How many axes of a value the layout of *unit* describes."""
    sub = next(iter(unit.values()))
    return 1 + _structured_depth(sub) if isinstance(sub, apyu.StructuredUnit) else 1


def _structured_dtype(unit: apyu.StructuredUnit, base: np.dtype, /) -> np.dtype:
    """Build the numpy dtype for *unit*'s layout, with *base* at every leaf."""
    return np.dtype(
        [
            (
                name,
                _structured_dtype(sub, base)
                if isinstance(sub, apyu.StructuredUnit)
                else base,
            )
            for name, sub in unit.items()
        ]
    )


def _records(value: np.ndarray, /) -> Any:
    """Group *value* into nested tuples -- how numpy spells a structured scalar."""
    return tuple(_records(v) for v in value) if value.ndim else value[()]


@conversion_method(type_from=UnitsMatrix, type_to=apyu.StructuredUnit)
def unitsmatrix_to_structured_unit(obj: UnitsMatrix, /) -> apyu.StructuredUnit:
    """Convert a `UnitsMatrix` to an `astropy.units.StructuredUnit`.

    Examples
    --------
    >>> import astropy.units as apyu
    >>> from plum import convert
    >>> from unxts.linalg import UnitsMatrix

    >>> convert(UnitsMatrix(("km", "s")), apyu.StructuredUnit)
    Unit("(km, s)")

    >>> convert(UnitsMatrix((("m", "s"), ("kg", "rad"))), apyu.StructuredUnit)
    Unit("((m, s), (kg, rad))")

    """
    return apyu.StructuredUnit(obj.to_tuple())


@conversion_method(type_from=apyu.StructuredUnit, type_to=UnitsMatrix)
def structured_unit_to_unitsmatrix(obj: apyu.StructuredUnit, /) -> UnitsMatrix:
    """Convert an `astropy.units.StructuredUnit` to a `UnitsMatrix`.

    Examples
    --------
    >>> import astropy.units as apyu
    >>> from plum import convert
    >>> from unxts.linalg import UnitsMatrix

    >>> um = convert(apyu.StructuredUnit(("m", "s", "kg")), UnitsMatrix)
    >>> um.shape, um[0]
    ((3,), Unit("m"))

    >>> convert(apyu.StructuredUnit((("m", "s"), ("kg", "rad"))), UnitsMatrix).shape
    (2, 2)

    """
    return UnitsMatrix(_structured_unit_to_tuple(obj))


@conversion_method(type_from=QuantityMatrix, type_to=apyu.Quantity)
def quantitymatrix_to_astropy_quantity(q: QuantityMatrix, /) -> apyu.Quantity:
    """Convert a `QuantityMatrix` to an `astropy.units.Quantity`.

    The unit layout claims the trailing axes -- one per level of nesting -- and
    whatever is left in front of them is batch, which is the one shape astropy
    allows over a structured dtype.

    Examples
    --------
    >>> import astropy.units as apyu
    >>> import jax.numpy as jnp
    >>> from plum import convert
    >>> from unxts.linalg import QuantityMatrix

    >>> qmat = QuantityMatrix(jnp.array([1.0, 2.0]), unit=("km", "s"))
    >>> convert(qmat, apyu.Quantity)
    <Quantity (1., 2.) (km, s)>

    A nested unit layout nests the record to match:

    >>> qmat = QuantityMatrix(
    ...     jnp.array([[1.0, 2.0], [3.0, 4.0]]), unit=(("m", "s"), ("kg", "rad"))
    ... )
    >>> convert(qmat, apyu.Quantity)
    <Quantity ((1., 2.), (3., 4.)) ((m, s), (kg, rad))>

    A flat layout instead reads one leading axis as batch:

    >>> qmat = QuantityMatrix(jnp.arange(6.0).reshape(2, 3), unit=("m", "s", "kg"))
    >>> convert(qmat, apyu.Quantity)
    <Quantity [(0., 1., 2.), (3., 4., 5.)] (m, s, kg)>

    """
    unit = apyu.StructuredUnit(q.unit.to_tuple())
    value = np.asarray(q.value)

    depth = _structured_depth(unit)
    dtype = _structured_dtype(unit, value.dtype)
    if value.ndim == depth:
        data = _records(value)
    elif value.ndim == depth + 1:
        data = [_records(row) for row in value]
    else:
        msg = (
            f"cannot lay a value of shape {value.shape} out under the unit "
            f"{unit}: the layout claims the last {depth} of those axes and "
            f"astropy allows one batch axis in front of them, not "
            f"{value.ndim - depth}."
        )
        raise ValueError(msg)

    return apyu.Quantity(np.array(data, dtype=dtype), unit=unit)
