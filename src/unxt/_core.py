# pylint: disable=import-error, no-member, unsubscriptable-object
#    b/c it doesn't understand dataclass fields

__all__ = ["AbstractParametricQuantity", "Quantity"]

from typing import Any, TypeVar, final

import equinox as eqx
import jax
import jax.core
from astropy.units import PhysicalType, Unit, UnitBase, get_physical_type
from jaxtyping import Array, ArrayLike, Shaped
from plum import parametric

from quaxed.array_api._dispatch import dispatcher

from ._base import AbstractQuantity

FMT = TypeVar("FMT")


##############################################################################


@parametric
class AbstractParametricQuantity(AbstractQuantity):
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

    """

    value: Shaped[Array, "*shape"] = eqx.field(converter=jax.numpy.asarray)
    unit: Unit = eqx.field(static=True, converter=Unit)

    def __check_init__(self) -> None:
        """Check whether the arguments are valid."""
        expected_dimensions = self._type_parameter._physical_type_id  # noqa: SLF001
        got_dimensions = self.unit.physical_type._physical_type_id  # noqa: SLF001
        if got_dimensions != expected_dimensions:
            msg = "Physical type mismatch."  # TODO: better error message
            raise ValueError(msg)

    # ---------------------------------------------------------------
    # Plum stuff

    @classmethod
    @dispatcher
    def __init_type_parameter__(cls, dimensions: PhysicalType) -> tuple[PhysicalType]:
        """Check whether the type parameters are valid."""
        return (dimensions,)

    @classmethod  # type: ignore[no-redef]
    @dispatcher
    def __init_type_parameter__(cls, dimensions: str) -> tuple[PhysicalType]:
        """Check whether the type parameters are valid."""
        try:
            dims = get_physical_type(dimensions)
        except ValueError:
            dims = PhysicalType(Unit(dimensions), dimensions)
        return (dims,)

    @classmethod  # type: ignore[no-redef]
    @dispatcher
    def __init_type_parameter__(cls, unit: UnitBase) -> tuple[PhysicalType]:
        """Infer the type parameter from the arguments."""
        if unit.physical_type != "unknown":
            return (unit.physical_type,)
        return (PhysicalType(unit, unit.to_string(fraction=False)),)

    @classmethod
    def __infer_type_parameter__(
        cls, value: ArrayLike, unit: Any
    ) -> tuple[PhysicalType]:
        """Infer the type parameter from the arguments."""
        return (Unit(unit).physical_type,)

    @classmethod
    @dispatcher  # type: ignore[misc]
    def __le_type_parameter__(
        cls,
        left: tuple[PhysicalType],
        right: tuple[PhysicalType],
    ) -> bool:
        """Define an order on type parameters.

        That is, check whether `left <= right` or not.
        """
        (dim_left,) = left
        (dim_right,) = right
        return dim_left == dim_right

    def __repr__(self) -> str:
        # fmt: off
        if self._type_parameter == "unknown":
            dim = " ".join(
                f"{unit}{power}" if power != 1 else unit
                for unit, power in self._type_parameter._physical_type_id  # noqa: SLF001
            )
        else:
            dim = self._type_parameter._name_string_as_ordered_set().split("'")[1]  # noqa: SLF001
        return f"Quantity[{dim!r}]({self.value!r}, unit={self.unit.to_string()!r})"
        # fmt: on


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
