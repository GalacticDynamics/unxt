# pylint: disable=import-error, no-member, unsubscriptable-object
#    b/c it doesn't understand dataclass fields

__all__ = ("Q", "Quantity")

from typing import Any, ClassVar

import equinox as eqx
from jaxtyping import Array, Shaped

from .base import AbstractQuantity
from .value import StaticValue, convert_to_quantity_value
from unxt.units import AbstractUnit, unit as parse_unit


class Quantity(AbstractQuantity):
    """The default quantity: units without dimension parametrization.

    This class is not parametrized by its dimensionality, making it a single
    class (and a single JAX pytree type) regardless of dimension. For runtime
    dimension checking and dimension-specific dispatch, see
    `unxt.ParametricQuantity`.

    Examples
    --------
    >>> import unxt as u
    >>> u.Quantity(1, "m")
    Quantity(Array(1, dtype=int32), unit='m')

    """

    value: Shaped[Array | StaticValue, "*shape"] = eqx.field(
        converter=convert_to_quantity_value
    )
    """The value of the `Quantity`."""

    unit: AbstractUnit = eqx.field(static=True, converter=parse_unit)
    """The unit associated with this value."""

    short_name: ClassVar[str] = "Q"

    def __class_getitem__(cls: "type[Quantity]", item: Any) -> "type[Quantity]":
        """No-op support for ``Quantity[...]`` syntax.

        The dimension parameter is accepted for interchangeability with
        `unxt.ParametricQuantity` but is NOT checked.

        >>> from unxt.quantity import Quantity
        >>> Quantity["length"]
        <class 'unxt...quantity...Quantity'>

        """
        return cls


Q = Quantity
"""Convenience alias for `Quantity`."""
