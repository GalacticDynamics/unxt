# pylint: disable=import-error, no-member, unsubscriptable-object
#    b/c it doesn't understand dataclass fields

__all__ = ["Distance"]

from typing import TypeVar, final

from plum import parametric

import quaxed.numpy as qnp

from ._core import AbstractParametricQuantity, Quantity

FMT = TypeVar("FMT")

parallax_base_length = Quantity(1, "AU")


##############################################################################


@final
@parametric
class Distance(AbstractParametricQuantity):
    """Distance quantities."""

    @property
    def parallax(  # noqa: PLR0206  (needed for quax boundary)
        self, base_length: Quantity["length"] = parallax_base_length
    ) -> Quantity["angle"]:
        """The parallax of the distance."""
        return qnp.arctan2(base_length, self)
