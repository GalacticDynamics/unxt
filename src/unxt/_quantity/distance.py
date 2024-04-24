# pylint: disable=import-error, no-member, unsubscriptable-object
#    b/c it doesn't understand dataclass fields

__all__ = ["AbstractDistance", "Distance", "Parallax"]

from abc import abstractmethod
from dataclasses import KW_ONLY
from typing import Any, TypeVar, final

import astropy.units as u
import equinox as eqx
import jax.numpy as jnp
from plum import add_conversion_method, add_promotion_rule

import quaxed.array_api as xp
import quaxed.numpy as qnp

from .base import AbstractQuantity
from .core import Quantity

FMT = TypeVar("FMT")

parallax_base_length = Quantity(1, "AU")
distance_modulus_base_distance = Quantity(10, "pc")
angle_dimension = u.get_physical_type("angle")
length_dimension = u.get_physical_type("length")

##############################################################################


class AbstractDistance(AbstractQuantity):
    """Distance quantities."""

    @property
    @abstractmethod
    def distance(self) -> "Distance":
        """The distance."""

    @property
    @abstractmethod
    def parallax(self) -> "Parallax":
        """The parallax."""

    @property
    @abstractmethod
    def distance_modulus(self) -> Quantity:
        """The distance modulus."""


# ============================================================================
# Conversion and Promotion

# Add a rule that when a AbstractDistance interacts with a Quantity, the
# distance degrades to a Quantity. This is necessary for many operations, e.g.
# division of a distance by non-dimensionless quantity where the resulting units
# are not those of a distance.
add_promotion_rule(AbstractDistance, Quantity, Quantity)


def _convert_distance_to_quantity(x: AbstractDistance) -> Quantity:
    """Convert a distance to a quantity."""
    return Quantity(x.value, x.unit)


add_conversion_method(AbstractDistance, Quantity, _convert_distance_to_quantity)


##############################################################################


@final
class Distance(AbstractDistance):
    """Distance quantities."""

    def __check_init__(self) -> None:
        """Check the initialization."""
        if self.unit.physical_type != length_dimension:
            msg = "Distance must have dimensions length."
            raise ValueError(msg)

    @property
    def distance(self) -> "Distance":
        """The distance."""
        return self

    @property
    def parallax(  # noqa: PLR0206  (needed for quax boundary)
        self, base_length: Quantity["length"] = parallax_base_length
    ) -> "Parallax":
        """The parallax of the distance."""
        v = qnp.arctan2(base_length, self)
        return Parallax(v.value, v.unit)

    @property
    def distance_modulus(  # noqa: PLR0206  (needed for quax boundary)
        self, base_length: Quantity["length"] = distance_modulus_base_distance
    ) -> Quantity:
        """The distance modulus."""
        return 5 * xp.log10(self / base_length)


##############################################################################


@final
class Parallax(AbstractDistance):
    """Parallax distance quantity."""

    _: KW_ONLY
    check_negative: bool = eqx.field(default=True, static=True, compare=False)
    """Whether to check that the parallax is strictly non-negative.

    Theoretically the parallax must be strictly non-negative (:math:`\tan(p) = 1
    AU / d`), however noisy direct measurements of the parallax can be negative.
    """

    def __check_init__(self) -> None:
        """Check the initialization."""
        if self.unit.physical_type != angle_dimension:
            msg = "Parallax must have dimensions angle."
            raise ValueError(msg)

        if self.check_negative:
            eqx.error_if(
                self.value,
                jnp.any(jnp.less(self.value, 0)),
                "Parallax must be non-negative.",
            )

    @property
    def distance(  # noqa: PLR0206  (needed for quax boundary)
        self, base_length: Quantity["length"] = parallax_base_length
    ) -> Distance:
        """The distance."""
        v = base_length / xp.tan(self)
        return Distance(v.value, v.unit)

    @property
    def parallax(self) -> "Parallax":
        """The parallax of the distance."""
        return self

    @property
    def distance_modulus(self) -> Quantity:
        """The distance modulus."""
        return self.distance.distance_modulus  # TODO: shortcut


# ============================================================================
# Additional constructors


@Distance.constructor._f.register  # noqa: SLF001
def constructor(
    cls: type[Distance], value: Parallax | Quantity["angle"], /, *, dtype: Any = None
) -> Distance:
    """Construct a `Distance` from an angle through the parallax."""
    d = parallax_base_length / xp.tan(value)
    return cls(xp.asarray(d.value, dtype=dtype), d.unit)


@Parallax.constructor._f.register  # type: ignore[no-redef]  # noqa: SLF001
def constructor(
    cls: type[Parallax], value: Distance | Quantity["length"], /, *, dtype: Any = None
) -> Parallax:
    """Construct a `Parallax` from a distance."""
    p = xp.atan2(parallax_base_length, value)
    return cls(xp.asarray(p.value, dtype=dtype), p.unit)
