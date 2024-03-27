# pylint: disable=import-error, no-member, unsubscriptable-object
#    b/c it doesn't understand dataclass fields

__all__ = ["Distance"]

from typing import Any, TypeVar, final

import astropy.units as u

import quaxed.array_api as xp
import quaxed.numpy as qnp

from ._base import AbstractQuantity
from ._core import Quantity

FMT = TypeVar("FMT")

parallax_base_length = Quantity(1, "AU")
distance_modulus_base_distance = Quantity(10, "pc")
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

    @property
    def distance_modulus(  # noqa: PLR0206  (needed for quax boundary)
        self, base_length: Quantity["length"] = distance_modulus_base_distance
    ) -> Quantity:
        """The distance modulus."""
        return 5 * qnp.log10(self / base_length)


@Distance.constructor._f.register  # type: ignore[attr-defined,misc]  # noqa: SLF001
def constructor(
    cls: type[Distance], value: Quantity["angle"], /, *, dtype: Any = None
) -> Distance:
    """Construct a `Distance` from an angle through the parallax."""
    d = parallax_base_length / xp.tan(value)
    return cls(xp.asarray(d.value, dtype=dtype), d.unit)
