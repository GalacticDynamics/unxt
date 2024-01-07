# pylint: disable=import-error, no-member, unsubscriptable-object
#    b/c it doesn't understand dataclass fields

__all__ = ["Quantity", "can_convert"]

import operator
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import array_api_jax_compat
import equinox as eqx
import jax
import jax.core
from astropy.units import Unit, UnitConversionError
from jaxtyping import ArrayLike
from quax import ArrayValue, quaxify
from typing_extensions import Self

if TYPE_CHECKING:
    from array_api import ArrayAPINamespace


class Quantity(ArrayValue):  # type: ignore[misc]
    """Represents an array, with each axis bound to a name."""

    value: jax.Array = eqx.field(converter=jax.numpy.asarray)
    unit: Unit = eqx.field(static=True, converter=Unit)

    # ===============================================================
    # Quax

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the array."""
        return self.value.shape

    def materialise(self) -> None:
        msg = "Refusing to materialise `Quantity`."
        raise RuntimeError(msg)

    def aval(self) -> jax.core.ShapedArray:
        return jax.core.get_aval(self.value)

    def enable_materialise(self, _: bool = True) -> Self:  # noqa: FBT001, FBT002
        return type(self)(self.value, self.unit)

    # ===============================================================
    # Array API

    def __array_namespace__(self, *, api_version: Any = None) -> "ArrayAPINamespace":
        return array_api_jax_compat

    # ===============================================================
    # Quantity

    def to(self, units: Unit) -> "Quantity":
        return type(self)(self.value * self.unit.to(units), units)

    def to_value(self, units: Unit) -> ArrayLike:
        if units == self.unit:
            return self.value
        return self.value * self.unit.to(units)

    def __getitem__(self, key: Any) -> "Quantity":
        return replace(self, value=self.value[key])

    # __add__
    # __radd__
    # __sub__
    # __rsub__
    # __mul__
    # __rmul__
    # __matmul__
    # __rmatmul__
    __pow__ = quaxify(operator.pow)
    __truediv__ = quaxify(operator.truediv)

    # Boolean
    __and__ = quaxify(operator.__and__)
    __gt__ = quaxify(operator.__gt__)
    __ge__ = quaxify(operator.__ge__)
    __lt__ = quaxify(operator.__lt__)
    __le__ = quaxify(operator.__le__)
    __eq__ = quaxify(operator.__eq__)
    __ne__ = quaxify(operator.__ne__)
    __neg__ = quaxify(operator.__neg__)


def can_convert(from_: Unit, to: Unit) -> bool:
    try:
        from_.to(to)
    except UnitConversionError:
        return False
    return True
