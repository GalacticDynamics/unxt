# pylint: disable=import-error, no-member, unsubscriptable-object
#    b/c it doesn't understand dataclass fields

__all__ = ("BareQuantity", "UncheckedQuantity")

import warnings
from typing import Any, Final
from typing_extensions import deprecated

import equinox as eqx
from jaxtyping import Array, Shaped
from plum import add_promotion_rule

from .base import AbstractQuantity
from .quantity import Quantity
from .value import convert_to_quantity_value
from unxt.units import AbstractUnit, unit as parse_unit


class BareQuantity(AbstractQuantity):
    """A fast implementation of the Quantity class.

    This class is not parametrized by its dimensionality.
    """

    value: Shaped[Array, "*shape"] = eqx.field(converter=convert_to_quantity_value)
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


# =====================

_deprecation_msg: Final = (
    "`UncheckedQuantity` is deprecated since v1.1 "
    "and will be removed in a future version. "
    "Use `BareQuantity` instead."
)


@deprecated("Use `BareQuantity` instead.")
class UncheckedQuantity(BareQuantity):
    """Deprecated version of `BareQuantity`."""

    value: Shaped[Array, "*shape"] = eqx.field(converter=convert_to_quantity_value)
    """The value of the `AbstractQuantity`."""

    unit: AbstractUnit = eqx.field(static=True, converter=parse_unit)
    """The unit associated with this value."""

    def __init__(self, value: Any, unit: Any) -> None:
        warnings.warn(_deprecation_msg, DeprecationWarning, stacklevel=2)
        BareQuantity.__init__(self, value=value, unit=unit)
