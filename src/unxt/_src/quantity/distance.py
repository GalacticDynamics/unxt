# pylint: disable=import-error, no-member, unsubscriptable-object
#    b/c it doesn't understand dataclass fields

__all__ = ["AbstractDistance", "Distance", "Parallax", "DistanceModulus"]

from abc import abstractmethod
from dataclasses import KW_ONLY
from typing import Any, TypeVar, final

import astropy.units as u
import equinox as eqx
import jax.numpy as jnp
from plum import add_promotion_rule, conversion_method

import quaxed.numpy as jnp

from .api import ustrip
from .base import AbstractQuantity
from .core import Quantity
from unxt._src.dimensions.core import dimensions_of

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
    def distance_modulus(self) -> "DistanceModulus":
        """The distance modulus."""


# ============================================================================
# Conversion and Promotion

# Add a rule that when a AbstractDistance interacts with a Quantity, the
# distance degrades to a Quantity. This is necessary for many operations, e.g.
# division of a distance by non-dimensionless quantity where the resulting units
# are not those of a distance.
add_promotion_rule(AbstractDistance, Quantity, Quantity)


@conversion_method(type_from=AbstractDistance, type_to=Quantity)  # type: ignore[misc]
def _convert_distance_to_quantity(x: AbstractDistance) -> Quantity:
    """Convert a distance to a quantity."""
    return Quantity(x.value, x.unit)


##############################################################################


@final
class Distance(AbstractDistance):
    """Distance quantities.

    The distance is a quantity with dimensions of length.

    Examples
    --------
    >>> from unxt import Distance
    >>> Distance(10, "pc")
    Distance(Array(10, dtype=int32, ...), unit='pc')

    The units are checked to have length dimensions.

    >>> try: Distance(10, "s")
    ... except ValueError as e: print(e)
    Distance must have dimensions length.

    """

    def __check_init__(self) -> None:
        """Check the initialization."""
        if dimensions_of(self) != length_dimension:
            msg = "Distance must have dimensions length."
            raise ValueError(msg)

    @property
    def distance(self) -> "Distance":
        """The distance.

        Examples
        --------
        >>> from unxt import Distance
        >>> d = Distance(10, "pc")
        >>> d.distance is d
        True

        """
        return self

    @property
    def parallax(  # noqa: PLR0206  (needed for quax boundary)
        self, base_length: Quantity["length"] = parallax_base_length
    ) -> "Parallax":
        r"""The parallax of the distance.

        The parallax is calculated as :math:`\arctan(1 AU / d)`.

        Examples
        --------
        >>> import quaxed.numpy as jnp
        >>> from unxt import Distance
        >>> d = Distance(1, "pc")
        >>> jnp.round(d.parallax.to("arcsec"), 2)
        Parallax(Array(1., dtype=float32, ...), unit='arcsec')

        """
        v = jnp.arctan2(base_length, self)
        return Parallax(v.value, v.unit)

    @property
    def distance_modulus(  # noqa: PLR0206  (needed for quax boundary)
        self, base_length: Quantity["length"] = distance_modulus_base_distance
    ) -> "DistanceModulus":
        """The distance modulus.

        Examples
        --------
        >>> from unxt import Distance
        >>> d = Distance(10, "pc")
        >>> d.distance_modulus
        DistanceModulus(Array(0., dtype=float32), unit='mag')

        """
        return DistanceModulus(5 * jnp.log10(self / base_length).value, "mag")


##############################################################################


@final
class Parallax(AbstractDistance):
    """Parallax distance quantity.

    Examples
    --------
    >>> from unxt import Parallax
    >>> Parallax(1, "mas")
    Parallax(Array(1, dtype=int32, ...), unit='mas')

    The units are checked to have angle dimensions.

    >>> try: Parallax(1, "pc")
    ... except ValueError as e: print(e)
    Parallax must have angular dimensions.

    The parallax is checked to be non-negative by default.

    >>> try: Parallax(-1, "mas")
    ... except Exception: print("negative")
    negative

    To disable this check, set `check_negative=False`.

    >>> Parallax(-1, "mas", check_negative=False)
    Parallax(Array(-1, dtype=int32, ...), unit='mas')

    """

    _: KW_ONLY
    check_negative: bool = eqx.field(default=True, static=True, compare=False)
    """Whether to check that the parallax is strictly non-negative.

    Theoretically the parallax must be strictly non-negative (:math:`\tan(p) = 1
    AU / d`), however noisy direct measurements of the parallax can be negative.
    """

    def __check_init__(self) -> None:
        """Check the initialization."""
        if dimensions_of(self) != angle_dimension:
            msg = "Parallax must have angular dimensions."
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
        r"""The distance.

        The distance is calculated as :math:`1 AU / \tan(p)`.

        Examples
        --------
        >>> from unxt import Parallax
        >>> p = Parallax(1, "mas")
        >>> p.distance.to("kpc")
        Distance(Array(1., dtype=float32, ...), unit='kpc')

        """
        v = base_length / jnp.tan(self)
        return Distance(v.value, v.unit)

    @property
    def parallax(self) -> "Parallax":
        """The parallax of the distance.

        Examples
        --------
        >>> from unxt import Parallax
        >>> p = Parallax(1, "mas")
        >>> p.parallax is p
        True

        """
        return self

    @property
    def distance_modulus(self) -> "DistanceModulus":
        """The distance modulus.

        Examples
        --------
        >>> from unxt import Parallax
        >>> Parallax(1, "mas").distance_modulus
        DistanceModulus(Array(10., dtype=float32), unit='mag')

        """
        return self.distance.distance_modulus  # TODO: specific shortcut


##############################################################################


class DistanceModulus(AbstractDistance):
    """Distance modulus quantity.

    Examples
    --------
    >>> from unxt import DistanceModulus
    >>> DistanceModulus(10, "mag")
    DistanceModulus(Array(10, dtype=int32, ...), unit='mag')

    The units are checked to have magnitude dimensions.

    >>> try: DistanceModulus(10, "pc")
    ... except ValueError as e: print(e)
    Distance modulus must have units of magnitude.

    """

    def __check_init__(self) -> None:
        """Check the initialization."""
        if self.unit != u.mag:
            msg = "Distance modulus must have units of magnitude."
            raise ValueError(msg)

    @property
    def distance(self) -> Distance:
        """The distance.

        The distance is calculated as :math:`10^{(m / 5 + 1)}`.

        Examples
        --------
        >>> from unxt import DistanceModulus
        >>> DistanceModulus(10, "mag").distance
        Distance(Array(1000., dtype=float32, ...), unit='pc')

        """
        return Distance(10 ** (self.value / 5 + 1), "pc")

    @property
    def parallax(self) -> Parallax:
        """The parallax.

        Examples
        --------
        >>> from unxt import DistanceModulus
        >>> DistanceModulus(10, "mag").parallax.to("mas")
        Parallax(Array(0.99999994, dtype=float32, ...), unit='mas')

        """
        return self.distance.parallax  # TODO: specific shortcut

    @property
    def distance_modulus(self) -> "DistanceModulus":
        """The distance modulus.

        Examples
        --------
        >>> from unxt import DistanceModulus
        >>> DistanceModulus(10, "mag").distance_modulus
        DistanceModulus(Array(10, dtype=int32, ...), unit='mag')

        """
        return self


# ============================================================================
# Additional constructors


@Distance.from_._f.register  # noqa: SLF001
def from_(
    cls: type[Distance], value: Parallax | Quantity["angle"], /, *, dtype: Any = None
) -> Distance:
    """Construct a `Distance` from an angle through the parallax.

    Examples
    --------
    >>> from unxt import Distance, Parallax, Quantity

    >>> Distance.from_(Parallax(1, "mas")).to("kpc")
    Distance(Array(1., dtype=float32, ...), unit='kpc')

    >>> Distance.from_(Quantity(1, "mas")).to("kpc")
    Distance(Array(1., dtype=float32, ...), unit='kpc')

    """
    d = parallax_base_length / jnp.tan(value)
    return cls(jnp.asarray(d.value, dtype=dtype), d.unit)


@Distance.from_._f.register  # type: ignore[no-redef]  # noqa: SLF001
def from_(
    cls: type[Distance],
    value: DistanceModulus | Quantity["mag"],
    /,
    *,
    dtype: Any = None,
) -> Distance:
    """Construct a `Distance` from a mag through the dist mod.

    Examples
    --------
    >>> from unxt import Distance, DistanceModulus, Quantity

    >>> Distance.from_(DistanceModulus(10, "mag")).to("pc")
    Distance(Array(1000., dtype=float32, ...), unit='pc')

    >>> Distance.from_(Quantity(10, "mag")).to("pc")
    Distance(Array(1000., dtype=float32, ...), unit='pc')

    """
    d = 10 ** (ustrip("mag", value) / 5 + 1)
    return cls(jnp.asarray(d, dtype=dtype), "pc")


@Parallax.from_._f.register  # type: ignore[no-redef]  # noqa: SLF001
def from_(
    cls: type[Parallax], value: Distance | Quantity["length"], /, *, dtype: Any = None
) -> Parallax:
    """Construct a `Parallax` from a distance.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from unxt import Parallax, Distance, Quantity

    >>> jnp.round(Parallax.from_(Distance(1, "pc")).to("mas"))
    Parallax(Array(1000., dtype=float32, ...), unit='mas')

    >>> jnp.round(Parallax.from_(Quantity(1, "pc")).to("mas"), 2)
    Parallax(Array(1000., dtype=float32, ...), unit='mas')

    """
    p = jnp.atan2(parallax_base_length, value)
    return cls(jnp.asarray(p.value, dtype=dtype), p.unit)
