# pylint: disable=import-error, no-member, unsubscriptable-object
#    b/c it doesn't understand dataclass fields

__all__ = ["Distance"]

from typing import TypeVar, final

import astropy.units as u

import quaxed.numpy as qnp

from ._base import AbstractQuantity
from ._core import Quantity

FMT = TypeVar("FMT")

parallax_base_length = Quantity(1, "AU")
length_dimension = u.get_physical_type("length")


##############################################################################


@final
class Distance(AbstractQuantity):
    """Distance quantities."""

    def __check_init__(self) -> None:
        """Check the initialization."""
        if self.unit.physical_type != length_dimension:
            msg = "Distance must have dimensions length."
            raise ValueError(msg)

    @property
    def parallax(  # noqa: PLR0206  (needed for quax boundary)
        self, base_length: Quantity["length"] = parallax_base_length
    ) -> Quantity["angle"]:
        """The parallax of the distance."""
        return qnp.arctan2(base_length, self)
