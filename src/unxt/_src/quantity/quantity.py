# pylint: disable=import-error, no-member, unsubscriptable-object
#    b/c it doesn't understand dataclass fields

__all__ = ("Quantity", "Q")

from typing import ClassVar, final
from typing_extensions import override

import equinox as eqx
import numpy as np
import quax_blocks
from jaxtyping import Array, Shaped
from plum import parametric

from .base import AbstractQuantity
from .base_parametric import AbstractParametricQuantity
from .value import StaticValue, convert_to_quantity_value
from unxt.units import AbstractUnit, unit as parse_unit
from unxt_api import is_unit_convertible, uconvert_value


@final
@parametric
class Quantity(AbstractParametricQuantity):
    """Arrays with associated units.

    This class is parametrized by the dimensions of the units.

    Attributes
    ----------
    short_name : str
        Short name 'Q' used for compact wadler-lindig printing.

    Examples
    --------
    >>> import unxt as u

    From an integer:

    >>> u.Q(1, "m")
    Quantity(Array(1, dtype=int32...), unit='m')

    From a float:

    >>> u.Q(1.0, "m")
    Quantity(Array(1., dtype=float32...), unit='m')

    From a list:

    >>> u.Q([1, 2, 3], "m")
    Quantity(Array([1, 2, 3], dtype=int32), unit='m')

    From a tuple:

    >>> u.Q((1, 2, 3), "m")
    Quantity(Array([1, 2, 3], dtype=int32), unit='m')

    From a `numpy.ndarray`:

    >>> import numpy as np
    >>> u.Q(np.array([1, 2, 3]), "m")
    Quantity(Array([1, 2, 3], dtype=int32), unit='m')

    From a `jax.Array`:

    >>> import jax.numpy as jnp
    >>> u.Q(jnp.array([1, 2, 3]), "m")
    Quantity(Array([1, 2, 3], dtype=int32), unit='m')

    The unit can also be given as a units object:

    >>> u.Q(1, u.unit("m"))
    Quantity(Array(1, dtype=int32...), unit='m')

    In the previous examples, the dimension parameter was inferred from the
    values. It can also be given explicitly:

    >>> u.Q["length"](1, "m")
    Quantity(Array(1, dtype=int32...), unit='m')

    This can be used for runtime checking of the input dimension!

    >>> try:
    ...     u.Q["length"](1, "s")
    ... except Exception as e:
    ...     print(e)
    Physical type mismatch.

    The dimension can also be given as a dimension object:

    >>> dims = u.dimension("length")
    >>> dims
    PhysicalType('length')
    >>> u.Q[dims](1.0, "m")
    Quantity(Array(1., dtype=float32...), unit='m')

    Or as a unit:

    >>> u.Q[u.unit("m")](1.0, "m")
    Quantity(Array(1., dtype=float32...), unit='m')

    Some tricky cases are when the physical type is unknown:

    >>> unit = u.unit("m2 / (kg s2)")
    >>> u.dimension_of(unit)
    PhysicalType('unknown')

    The dimension can be given as a string in all cases, but is necessary when
    the physical type is unknown:

    >>> print(u.Q["m2 kg-1 s-2"](1.0, unit))  # to show the [dim]
    Quantity['m2 kg-1 s-2'](1., unit='m2 / (kg s2)')

    """

    short_name: ClassVar[str] = "Q"
    """Short name for compact printing."""

    value: Shaped[Array, "*shape"] | StaticValue = eqx.field(
        converter=convert_to_quantity_value
    )
    """The value of the `AbstractQuantity`."""

    unit: AbstractUnit = eqx.field(static=True, converter=parse_unit)
    """The unit associated with this value."""

    # TODO: consider moving up to `AbstractQuantity`
    @override
    def __eq__(self, other: object, /) -> object:  # type: ignore[override]
        """Element-wise equality, with a scalar-bool fast path for StaticValue.

        When both operands carry a `StaticValue`, structural (scalar bool)
        equality is returned so that this quantity can be used safely as a
        ``static_argnames`` argument in `jax.jit`.  Units are accounted for by
        converting `other` to `self`'s units before comparing.  In all other
        cases the element-wise `NumpyEqMixin` behaviour is preserved.

        Examples
        --------
        >>> import numpy as np
        >>> import unxt as u

        Normal array quantities return element-wise boolean arrays:

        >>> q1 = u.Q([1, 2, 3], "m")
        >>> q2 = u.Q([1, 0, 3], "m")
        >>> q1 == q2
        Quantity(Array([ True, False,  True], dtype=bool), unit='')

        When both quantities carry a `StaticValue`, a scalar `bool` is returned,
        which is required for use as ``static_argnames`` in `jax.jit`:

        >>> sv1 = u.quantity.StaticValue(np.array([1.0, 2.0]))
        >>> sv2 = u.quantity.StaticValue(np.array([1.0, 2.0]))
        >>> u.Q(sv1, "m") == u.Q(sv2, "m")
        True

        >>> sv3 = u.quantity.StaticValue(np.array([3.0, 4.0]))
        >>> u.Q(sv1, "m") == u.Q(sv3, "m")
        False

        Unit conversion is applied before comparing, so equivalent quantities in
        different units compare equal:

        >>> sv_km = u.quantity.StaticValue(np.array([0.001, 0.002]))
        >>> u.Q(sv1, "m") == u.Q(sv_km, "km")
        True

        Quantities with incompatible dimensions are never equal:

        >>> sv_s = u.quantity.StaticValue(np.array([1.0, 2.0]))
        >>> u.Q(sv1, "m") == u.Q(sv_s, "s")
        False

        """
        if (
            isinstance(self.value, StaticValue)
            and isinstance(other, AbstractQuantity)
            and isinstance(other.value, StaticValue)
        ):
            if not is_unit_convertible(other.unit, self.unit):
                return False
            converted = uconvert_value(self.unit, other.unit, other.value.array)
            return bool(np.array_equal(self.value.array, converted))

        return quax_blocks.NumpyEqMixin.__eq__(self, other)

    # TODO: consider moving up to `AbstractQuantity`
    @override
    def __hash__(self) -> int:
        """Return hash when value is a StaticValue; otherwise unhashable.

        JAX arrays cannot be hashed, but a `StaticValue` can. This allows a
        ``Quantity`` backed by a `StaticValue` to be used as a static argument
        in `jax.jit` via ``static_argnames``.

        Examples
        --------
        >>> import numpy as np
        >>> import unxt as u

        A ``Quantity`` backed by a `StaticValue` is hashable:

        >>> sv = u.quantity.StaticValue(np.array([1.0, 2.0]))
        >>> isinstance(hash(u.Q(sv, "m")), int)
        True

        A normal ``Quantity`` (JAX array value) is not:

        >>> try:
        ...     hash(u.Q([1.0, 2.0], "m"))
        ... except TypeError as e:
        ...     print(e)
        unhashable type: 'jaxlib...ArrayImpl'

        """
        if isinstance(self.value, StaticValue):
            return hash((self.value, self.unit))
        return super().__hash__()


Q = Quantity  # convenience alias
