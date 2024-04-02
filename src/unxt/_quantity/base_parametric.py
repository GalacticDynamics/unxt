# pylint: disable=import-error, no-member, unsubscriptable-object
#    b/c it doesn't understand dataclass fields

__all__ = ["AbstractParametricQuantity"]

from typing import Any

import equinox as eqx
import jax
import jax.core
from astropy.units import PhysicalType, Unit, UnitBase, get_physical_type
from jaxtyping import Array, ArrayLike, Shaped
from plum import parametric

from quaxed.array_api._dispatch import dispatcher

from .base import AbstractQuantity


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
