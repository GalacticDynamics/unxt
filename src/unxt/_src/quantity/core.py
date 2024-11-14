# pylint: disable=import-error, no-member, unsubscriptable-object
#    b/c it doesn't understand dataclass fields

__all__ = ["Quantity"]

from dataclasses import replace
from typing import final

from jaxtyping import ArrayLike
from plum import parametric

from .base import AbstractQuantity
from .base_parametric import AbstractParametricQuantity


@final
@parametric
class Quantity(AbstractParametricQuantity):
    """Arrays with associated units.

    This class is parametrized by the dimensions of the units.

    Examples
    --------
    >>> import unxt as u

    From an integer:

    >>> u.Quantity(1, "m")
    Quantity['length'](Array(1, dtype=int32, ...), unit='m')

    From a float:

    >>> u.Quantity(1.0, "m")
    Quantity['length'](Array(1., dtype=float32, ...), unit='m')

    From a list:

    >>> u.Quantity([1, 2, 3], "m")
    Quantity['length'](Array([1, 2, 3], dtype=int32), unit='m')

    From a tuple:

    >>> u.Quantity((1, 2, 3), "m")
    Quantity['length'](Array([1, 2, 3], dtype=int32), unit='m')

    From a `numpy.ndarray`:

    >>> import numpy as np
    >>> u.Quantity(np.array([1, 2, 3]), "m")
    Quantity['length'](Array([1, 2, 3], dtype=int32), unit='m')

    From a `jax.Array`:

    >>> import jax.numpy as jnp
    >>> u.Quantity(jnp.array([1, 2, 3]), "m")
    Quantity['length'](Array([1, 2, 3], dtype=int32), unit='m')

    The unit can also be given as a units object:

    >>> u.Quantity(1, u.unit("m"))
    Quantity['length'](Array(1, dtype=int32, ...), unit='m')

    In the previous examples, the dimension parameter was inferred from the
    values. It can also be given explicitly:

    >>> u.Quantity["length"](1, "m")
    Quantity['length'](Array(1, dtype=int32, ...), unit='m')

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
    Quantity['length'](Array(1., dtype=float32, ...), unit='m')

    Or as a unit:

    >>> u.Quantity[u.unit("m")](1.0, "m")
    Quantity['length'](Array(1., dtype=float32, ...), unit='m')

    Some tricky cases are when the physical type is unknown:

    >>> unit = u.unit("m2 / (kg s2)")
    >>> u.dimension_of(unit)
    PhysicalType('unknown')

    The dimension can be given as a string in all cases, but is necessary when
    the physical type is unknown:

    >>> u.Quantity["m2 kg-1 s-2"](1.0, unit)
    Quantity['m2 kg-1 s-2'](Array(1., dtype=float32, ...), unit='m2 / (kg s2)')

    """


@AbstractQuantity.__mod__.dispatch  # type: ignore[misc]
def mod(self: Quantity["dimensionless"], other: ArrayLike) -> Quantity["dimensionless"]:
    """Take the mod.

    Examples
    --------
    >>> import unxt as u

    >>> q = u.Quantity(480, "deg")
    >>> q % u.Quantity(360, "deg")
    Quantity['angle'](Array(120, dtype=int32, ...), unit='deg')

    """
    return replace(self, value=self.value % other)
