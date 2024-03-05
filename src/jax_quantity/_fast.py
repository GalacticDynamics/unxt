# pylint: disable=import-error, no-member, unsubscriptable-object
#    b/c it doesn't understand dataclass fields

__all__ = ["FastQuantity"]


from ._base import AbstractQuantity


class FastQuantity(AbstractQuantity):
    """A fast implementation of the Quantity class.

    This class is not parametrized by its dimensionality.
    """
