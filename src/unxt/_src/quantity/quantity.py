# pylint: disable=import-error, no-member, unsubscriptable-object
#    b/c it doesn't understand dataclass fields

__all__ = ("Quantity", "Q")

from dataclasses import replace
from typing import ClassVar, final

import equinox as eqx
from jaxtyping import Array, ArrayLike, Shaped
from plum import parametric

from .base import AbstractQuantity
from .base_parametric import AbstractParametricQuantity
from .value import convert_to_quantity_value
from unxt.units import AbstractUnit, unit as parse_unit


@final
@parametric
class Quantity(AbstractParametricQuantity):
    """Arrays with associated units.

    This class is parametrized by the dimensions of the units.

    Attributes
    ----------
    short_name : str
        Short name 'Q' used for compact wadler-lindig printing.

    Examples
    --------
    >>> import unxt as u

    From an integer:

    >>> u.Quantity(1, "m")
    Quantity(Array(1, dtype=int32, ...), unit='m')

    From a float:

    >>> u.Quantity(1.0, "m")
    Quantity(Array(1., dtype=float32, ...), unit='m')

    From a list:

    >>> u.Quantity([1, 2, 3], "m")
    Quantity(Array([1, 2, 3], dtype=int32), unit='m')

    From a tuple:

    >>> u.Quantity((1, 2, 3), "m")
    Quantity(Array([1, 2, 3], dtype=int32), unit='m')

    From a `numpy.ndarray`:

    >>> import numpy as np
    >>> u.Quantity(np.array([1, 2, 3]), "m")
    Quantity(Array([1, 2, 3], dtype=int32), unit='m')

    From a `jax.Array`:

    >>> import jax.numpy as jnp
    >>> u.Quantity(jnp.array([1, 2, 3]), "m")
    Quantity(Array([1, 2, 3], dtype=int32), unit='m')

    The unit can also be given as a units object:

    >>> u.Quantity(1, u.unit("m"))
    Quantity(Array(1, dtype=int32, ...), unit='m')

    In the previous examples, the dimension parameter was inferred from the
    values. It can also be given explicitly:

    >>> u.Quantity["length"](1, "m")
    Quantity(Array(1, dtype=int32, ...), unit='m')

    This can be used for runtime checking of the input dimension!

    >>> try:
    ...     u.Quantity["length"](1, "s")
    ... except Exception as e:
    ...     print(e)
    Physical type mismatch.

    The dimension can also be given as a dimension object:

    >>> dims = u.dimension("length")
    >>> dims
    PhysicalType('length')
    >>> u.Quantity[dims](1.0, "m")
    Quantity(Array(1., dtype=float32, ...), unit='m')

    Or as a unit:

    >>> u.Quantity[u.unit("m")](1.0, "m")
    Quantity(Array(1., dtype=float32, ...), unit='m')

    Some tricky cases are when the physical type is unknown:

    >>> unit = u.unit("m2 / (kg s2)")
    >>> u.dimension_of(unit)
    PhysicalType('unknown')

    The dimension can be given as a string in all cases, but is necessary when
    the physical type is unknown:

    >>> print(u.Quantity["m2 kg-1 s-2"](1.0, unit))  # to show the [dim]
    Quantity['m2 kg-1 s-2'](1., unit='m2 / (kg s2)')

    """

    short_name: ClassVar[str] = "Q"
    """Short name for compact printing."""

    value: Shaped[Array, "*shape"] = eqx.field(converter=convert_to_quantity_value)
    """The value of the `AbstractQuantity`."""

    unit: AbstractUnit = eqx.field(static=True, converter=parse_unit)
    """The unit associated with this value."""


Q = Quantity  # convenience alias


@AbstractQuantity.__mod__.dispatch  # type: ignore[misc, attr-defined]
def mod(self: Quantity["dimensionless"], other: ArrayLike) -> Quantity["dimensionless"]:
    """Take the mod.

    Examples
    --------
    >>> import unxt as u

    >>> q = u.Quantity(480, "deg")
    >>> q % u.Quantity(360, "deg")
    Quantity(Array(120, dtype=int32, ...), unit='deg')

    """
    return replace(self, value=self.value % other)
