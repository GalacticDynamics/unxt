# pylint: disable=import-error, no-member, unsubscriptable-object
#    b/c it doesn't understand dataclass fields

__all__ = ["UncheckedQuantity"]

from typing import Any

import equinox as eqx
from jaxtyping import Array, Shaped

from .base import AbstractQuantity
from .value import convert_to_quantity_value
from unxt._src.units import AstropyUnits, unit as parse_unit
from unxt.units import unit as parse_unit


class UncheckedQuantity(AbstractQuantity):
    """A fast implementation of the Quantity class.

    This class is not parametrized by its dimensionality.
    """

    value: Shaped[Array, "*shape"] = eqx.field(converter=convert_to_quantity_value)
    """The value of the `AbstractQuantity`."""

    unit: AstropyUnits = eqx.field(static=True, converter=parse_unit)
    """The unit associated with this value."""

    def __class_getitem__(
        cls: type["UncheckedQuantity"], item: Any
    ) -> type["UncheckedQuantity"]:
        """No-op support for `UncheckedQuantity[...]` syntax.

        This method is called when the class is subscripted, e.g.:

        >>> from unxt.quantity import UncheckedQuantity
        >>> UncheckedQuantity["length"]
        <class 'unxt...quantity...UncheckedQuantity'>

        """
        return cls
