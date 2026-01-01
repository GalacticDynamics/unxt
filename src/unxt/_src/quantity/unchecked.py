# pylint: disable=import-error, no-member, unsubscriptable-object
#    b/c it doesn't understand dataclass fields

__all__ = ("BareQuantity",)

from typing import Any

import equinox as eqx
from jaxtyping import Array, Shaped
from plum import add_promotion_rule

from .base import AbstractQuantity
from .quantity import Quantity
from .value import StaticValue, convert_to_quantity_value
from unxt.units import AbstractUnit, unit as parse_unit


class BareQuantity(AbstractQuantity):
    """A fast implementation of the Quantity class.

    This class is not parametrized by its dimensionality.
    """

    value: Shaped[Array | StaticValue, "*shape"] = eqx.field(
        converter=convert_to_quantity_value
    )
    """The value of the `AbstractQuantity`."""

    unit: AbstractUnit = eqx.field(static=True, converter=parse_unit)
    """The unit associated with this value."""

    def __class_getitem__(cls: "type[BareQuantity]", item: Any) -> "type[BareQuantity]":
        """No-op support for `BareQuantity[...]` syntax.

        This method is called when the class is subscripted, e.g.:

        >>> from unxt.quantity import BareQuantity
        >>> BareQuantity["length"]
        <class 'unxt...quantity...BareQuantity'>

        """
        return cls


add_promotion_rule(BareQuantity, Quantity, Quantity)
