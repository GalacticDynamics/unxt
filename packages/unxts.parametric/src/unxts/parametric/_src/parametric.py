# pylint: disable=import-error, no-member, unsubscriptable-object
#    b/c it doesn't understand dataclass fields

__all__ = ("PQ", "ParametricQuantity")

from typing import ClassVar, final
from typing_extensions import override

import equinox as eqx
from jaxtyping import Array, Shaped
from plum import add_promotion_rule, parametric

from .base_parametric import AbstractParametricQuantity
from unxt.quantity import (
    Quantity,
    StaticValue,
    convert_to_quantity_value,
)
from unxt.units import AbstractUnit, parse_unit


@final
@parametric
class ParametricQuantity(AbstractParametricQuantity):
    """Arrays with associated units, parametrized by dimension.

    This class is parametrized by the dimensions of the units, which enables
    runtime dimension checking and dimension-specific dispatch. For the
    non-parametric default quantity, see `unxt.Quantity`.

    Attributes
    ----------
    short_name : str
        Short name 'PQ' used for compact wadler-lindig printing.

    Examples
    --------
    >>> import unxt as u
    >>> import unxts.parametric as up

    From an integer:

    >>> up.PQ(1, "m")
    ParametricQuantity(Array(1, dtype=int32...), unit='m')

    From a float:

    >>> up.PQ(1.0, "m")
    ParametricQuantity(Array(1., dtype=float32...), unit='m')

    From a list:

    >>> up.PQ([1, 2, 3], "m")
    ParametricQuantity(Array([1, 2, 3], dtype=int32), unit='m')

    From a tuple:

    >>> up.PQ((1, 2, 3), "m")
    ParametricQuantity(Array([1, 2, 3], dtype=int32), unit='m')

    From a `numpy.ndarray`:

    >>> import numpy as np
    >>> up.PQ(np.array([1, 2, 3]), "m")
    ParametricQuantity(Array([1, 2, 3], dtype=int32), unit='m')

    From a `jax.Array`:

    >>> import jax.numpy as jnp
    >>> up.PQ(jnp.array([1, 2, 3]), "m")
    ParametricQuantity(Array([1, 2, 3], dtype=int32), unit='m')

    The unit can also be given as a units object:

    >>> up.PQ(1, u.unit("m"))
    ParametricQuantity(Array(1, dtype=int32...), unit='m')

    In the previous examples, the dimension parameter was inferred from the
    values. It can also be given explicitly:

    >>> up.PQ["length"](1, "m")
    ParametricQuantity(Array(1, dtype=int32...), unit='m')

    This can be used for runtime checking of the input dimension!

    >>> try:
    ...     up.PQ["length"](1, "s")
    ... except Exception as e:
    ...     print(e)
    Physical type mismatch.

    The dimension can also be given as a dimension object:

    >>> dims = u.dimension("length")
    >>> dims
    PhysicalType('length')
    >>> up.PQ[dims](1.0, "m")
    ParametricQuantity(Array(1., dtype=float32...), unit='m')

    Or as a unit:

    >>> up.PQ[u.unit("m")](1.0, "m")
    ParametricQuantity(Array(1., dtype=float32...), unit='m')

    Some tricky cases are when the physical type is unknown:

    >>> unit = u.unit("m2 / (kg s2)")
    >>> u.dimension_of(unit)
    PhysicalType('unknown')

    The dimension can be given as a string in all cases, but is necessary when
    the physical type is unknown:

    >>> print(up.PQ["m2 kg-1 s-2"](1.0, unit))  # to show the [dim]
    ParametricQuantity['m2 kg-1 s-2'](1., unit='m2 / (kg s2)')

    """

    short_name: ClassVar[str] = "PQ"
    """Short name for compact printing."""

    value: Shaped[Array, "*shape"] | StaticValue = eqx.field(
        converter=convert_to_quantity_value
    )
    """The value of the `AbstractQuantity`."""

    unit: AbstractUnit = eqx.field(static=True, converter=parse_unit)
    """The unit associated with this value."""

    @override
    def __hash__(self) -> int:
        """Return hash when value is a StaticValue; otherwise unhashable.

        JAX arrays cannot be hashed, but a `StaticValue` can. This allows a
        ``ParametricQuantity`` backed by a `StaticValue` to be used as a static argument
        in `jax.jit` via ``static_argnames``.

        Examples
        --------
        >>> import numpy as np
        >>> import unxt as u
        >>> import unxts.parametric as up

        A ``ParametricQuantity`` backed by a `StaticValue` is hashable:

        >>> sv = u.quantity.StaticValue(np.array([1.0, 2.0]))
        >>> isinstance(hash(up.PQ(sv, "m")), int)
        True

        A normal ``ParametricQuantity`` (JAX array value) is not:

        >>> try:
        ...     hash(up.PQ([1.0, 2.0], "m"))
        ... except TypeError as e:
        ...     print(e)
        unhashable type: 'jaxlib...ArrayImpl'

        """
        if isinstance(self.value, StaticValue):
            return hash((self.value, self.unit))
        return super().__hash__()


PQ = ParametricQuantity
"""Convenience alias for `ParametricQuantity`."""

add_promotion_rule(Quantity, ParametricQuantity, ParametricQuantity)
