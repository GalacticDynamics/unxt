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

    Parameters
    ----------
    value : array-like
        The array of values. Anything that can be converted to an array by
        `jax.numpy.asarray`.
    unit : Unit-like
        The unit of the array. Anything that can be converted to a unit by
        `astropy.units.Unit`.

    Examples
    --------
    >>> from unxt import Quantity

    From an integer:

    >>> Quantity(1, "m")
    Quantity['length'](Array(1, dtype=int32, ...), unit='m')

    From a float:

    >>> Quantity(1.0, "m")
    Quantity['length'](Array(1., dtype=float32, ...), unit='m')

    From a list:

    >>> Quantity([1, 2, 3], "m")
    Quantity['length'](Array([1, 2, 3], dtype=int32), unit='m')

    From a tuple:

    >>> Quantity((1, 2, 3), "m")
    Quantity['length'](Array([1, 2, 3], dtype=int32), unit='m')

    From a :class:`numpy.ndarray`:

    >>> import numpy as np
    >>> Quantity(np.array([1, 2, 3]), "m")
    Quantity['length'](Array([1, 2, 3], dtype=int32), unit='m')

    From a :class:`jax.Array`:

    >>> import jax.numpy as jnp
    >>> Quantity(jnp.array([1, 2, 3]), "m")
    Quantity['length'](Array([1, 2, 3], dtype=int32), unit='m')

    The unit can also be given as a :class:`astropy.units.Unit`:

    >>> import astropy.units as u
    >>> Quantity(1, u.m)
    Quantity['length'](Array(1, dtype=int32, ...), unit='m')

    In the previous examples, the dimensions parameter was inferred from the
    values. It can also be given explicitly:

    >>> Quantity["length"](1, "m")
    Quantity['length'](Array(1, dtype=int32, ...), unit='m')

    This can be used for runtime checking of the input dimensions!

    >>> try: Quantity["length"](1, "s")
    ... except Exception as e: print(e)
    Physical type mismatch.

    The dimensions can also be given as a :class:`astropy.units.PhysicalType`:

    >>> dimensions = u.km.physical_type
    >>> dimensions
    PhysicalType('length')
    >>> Quantity[dimensions](1.0, "m")
    Quantity['length'](Array(1., dtype=float32, ...), unit='m')

    Or as a unit:

    >>> Quantity[u.m](1.0, "m")
    Quantity['length'](Array(1., dtype=float32, ...), unit='m')

    Some tricky cases are when the physical type is unknown:

    >>> unit = u.m ** 2 / (u.kg * u.s ** 2)
    >>> unit.physical_type
    PhysicalType('unknown')

    The dimensions can be given as a string in all cases, but is necessary when
    the physical type is unknown:

    >>> Quantity['m2 kg-1 s-2'](1.0, unit)
    Quantity['m2 kg-1 s-2'](Array(1., dtype=float32, ...), unit='m2 / (kg s2)')

    """


@AbstractQuantity.__mod__.dispatch  # type: ignore[misc]
def mod(self: Quantity["dimensionless"], other: ArrayLike) -> Quantity["dimensionless"]:
    """Take the mod.

    Examples
    --------
    >>> from unxt import Quantity

    >>> q = Quantity(480, "deg")
    >>> q % Quantity(360, "deg")
    Quantity['angle'](Array(120, dtype=int32, ...), unit='deg')

    """
    return replace(self, value=self.value % other)
