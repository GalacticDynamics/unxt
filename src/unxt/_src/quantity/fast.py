# pylint: disable=import-error, no-member, unsubscriptable-object
#    b/c it doesn't understand dataclass fields

__all__ = ["UncheckedQuantity"]

from typing import Any

from .base import AbstractQuantity


class UncheckedQuantity(AbstractQuantity):
    """A fast implementation of the Quantity class.

    This class is not parametrized by its dimensionality.
    """

    def __class_getitem__(
        cls: type["UncheckedQuantity"], item: Any
    ) -> type["UncheckedQuantity"]:
        """No-op support for `UncheckedQuantity[...]` syntax.

        This method is called when the class is subscripted, e.g.:

        >>> from unxt.quantity import UncheckedQuantity
        >>> UncheckedQuantity["length"]
        <class 'unxt._src.quantity.fast.UncheckedQuantity'>

        """
        return cls
