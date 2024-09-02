# pylint: disable=import-error, no-member, unsubscriptable-object
#    b/c it doesn't understand dataclass fields

__all__ = ["AbstractParametricQuantity"]

from typing import Any

import equinox as eqx
import jax
import jax.core
from astropy.units import PhysicalType, Unit, get_physical_type
from jaxtyping import Array, ArrayLike, Shaped
from plum import parametric

from quaxed.array_api._dispatch import dispatcher

from .base import AbstractQuantity
from unxt._unxt.dimensions.core import dimensions_of
from unxt._unxt.typing_ext import Unit as UnitTypes
from unxt._unxt.units.core import units


@parametric
class AbstractParametricQuantity(AbstractQuantity):
    """Arrays with associated units.

    This class is parametrized by the dimensions of the units.

    """

    value: Shaped[Array, "*shape"] = eqx.field(converter=jax.numpy.asarray)
    """The value of the Quantity."""

    unit: Unit = eqx.field(static=True, converter=Unit)
    """The unit associated with this value."""

    def __post_init__(self) -> None:
        """Check whether the arguments are valid."""
        self._type_parameter: PhysicalType

    def __check_init__(self) -> None:
        """Check whether the arguments are valid."""
        expected_dimensions = self._type_parameter._physical_type_id  # noqa: SLF001
        got_dimensions = dimensions_of(self)._physical_type_id  # noqa: SLF001
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
            dims = PhysicalType(units(dimensions), dimensions)
        return (dims,)

    @classmethod  # type: ignore[no-redef]
    @dispatcher
    def __init_type_parameter__(cls, unit: UnitTypes) -> tuple[PhysicalType]:
        """Infer the type parameter from the arguments."""
        dims = dimensions_of(unit)
        if dims != "unknown":
            return (dims,)
        return (PhysicalType(unit, unit.to_string(fraction=False)),)

    @classmethod
    def __infer_type_parameter__(
        cls, value: ArrayLike, unit: Any
    ) -> tuple[PhysicalType]:
        """Infer the type parameter from the arguments."""
        return (dimensions_of(units(unit)),)

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
