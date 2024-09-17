# pylint: disable=import-error, no-member, unsubscriptable-object
#    b/c it doesn't understand dataclass fields

__all__ = ["AbstractParametricQuantity"]

from typing import Any

import equinox as eqx
import jax
import jax.core
from astropy.units import PhysicalType as Dimensions, Unit
from jaxtyping import Array, ArrayLike, Shaped
from plum import dispatch, parametric

from .base import AbstractQuantity
from unxt._unxt.dimensions.core import dimensions, dimensions_of
from unxt._unxt.typing_ext import Unit as UnitTypes
from unxt._unxt.units.core import units


@parametric
class AbstractParametricQuantity(AbstractQuantity):
    """Arrays with associated units.

    This class is parametrized by the dimensions of the units.

    """

    value: Shaped[Array, "*shape"] = eqx.field(converter=jax.numpy.asarray)
    """The value of the `AbstractQuantity`."""

    unit: Unit = eqx.field(static=True, converter=units)
    """The unit associated with this value."""

    def __post_init__(self) -> None:
        """Check whether the arguments are valid."""
        self._type_parameter: Dimensions

    def __check_init__(self) -> None:
        """Check whether the arguments are valid."""
        expected_dimensions = self._type_parameter
        got_dimensions = dimensions_of(self.unit)
        if got_dimensions != expected_dimensions:
            msg = "Physical type mismatch."  # TODO: better error message
            raise ValueError(msg)

    # ---------------------------------------------------------------
    # Plum stuff

    @classmethod
    @dispatch
    def __init_type_parameter__(cls, dims: Dimensions, /) -> tuple[Dimensions]:
        """Check whether the type parameters are valid."""
        return (dims,)

    @classmethod  # type: ignore[no-redef]
    @dispatch
    def __init_type_parameter__(cls, dims: str, /) -> tuple[Dimensions]:
        """Check whether the type parameters are valid."""
        dims_: Dimensions
        try:
            dims_ = dimensions(dims)
        except ValueError:
            dims_ = dimensions_of(units(dims))
        return (dims_,)

    @classmethod  # type: ignore[no-redef]
    @dispatch
    def __init_type_parameter__(cls, unit: UnitTypes, /) -> tuple[Dimensions]:
        """Infer the type parameter from the arguments."""
        dims = dimensions_of(unit)
        if dims != "unknown":
            return (dims,)
        return (Dimensions(unit, unit.to_string(fraction=False)),)

    @classmethod
    def __infer_type_parameter__(cls, value: ArrayLike, unit: Any) -> tuple[Dimensions]:
        """Infer the type parameter from the arguments."""
        return (dimensions_of(units(unit)),)

    @classmethod
    @dispatch  # type: ignore[misc]
    def __le_type_parameter__(
        cls, left: tuple[Dimensions], right: tuple[Dimensions]
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
            try:  # Astropy v6-
                ptid = self._type_parameter._physical_type_id  # noqa: SLF001
            except AttributeError:  # Astropy v7+
                ptid = self._type_parameter._unit._physical_type_id  # noqa: SLF001
            dim = " ".join(
                f"{unit}{power}" if power != 1 else unit
                for unit, power in ptid
            )
        else:
            dim = self._type_parameter._name_string_as_ordered_set().split("'")[1]  # noqa: SLF001
        return f"Quantity[{dim!r}]({self.value!r}, unit={self.unit.to_string()!r})"
        # fmt: on
