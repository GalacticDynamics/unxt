# pylint: disable=import-error, no-member, unsubscriptable-object
#    b/c it doesn't understand dataclass fields

__all__ = ["Quantity", "can_convert"]

import operator
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any, TypeVar, final

import array_api_jax_compat
import equinox as eqx
import jax
import jax.core
import jax.numpy as jnp
from array_api_jax_compat._dispatch import dispatcher
from astropy.units import (
    CompositeUnit,
    Quantity as AstropyQuantity,
    Unit,
    UnitConversionError,
)
from jaxtyping import Array, ArrayLike, Shaped
from quax import ArrayValue, quaxify
from typing_extensions import Self

if TYPE_CHECKING:
    from array_api import ArrayAPINamespace


T = TypeVar("T")


def _flip_binop(binop: Callable[[Any, Any], Any]) -> Callable[[Any, Any], Any]:
    def _binop(x: Any, y: Any) -> Any:
        return binop(y, x)

    return _binop


def bool_op(op: Callable[[Any, Any], Any]) -> Callable[[Any, Any], Any]:
    def _op(self: "Quantity", other: Any) -> Shaped[Array, "..."]:
        if not isinstance(other, Quantity):
            return NotImplemented

        try:
            other_value = other.to_value(self.unit)
        except UnitConversionError:
            return jnp.full(self.value.shape, fill_value=False, dtype=bool)

        return op(self.value, other_value)

    return _op


@final
class Quantity(ArrayValue):  # type: ignore[misc]
    """Represents an array, with each axis bound to a name."""

    value: jax.Array = eqx.field(converter=jax.numpy.asarray)
    unit: Unit = eqx.field(static=True, converter=Unit)

    # ---------------------------------------------------------------
    # Constructors

    @classmethod
    @dispatcher
    def constructor(
        cls: "type[Quantity]", value: ArrayLike, unit: Any, /
    ) -> "Quantity":
        # Dispatch on both arguments.
        # Construct using the standard `__init__` method.
        return cls(value, unit)

    @classmethod  # type: ignore[no-redef]
    @dispatcher
    def constructor(
        cls: "type[Quantity]", value: ArrayLike, *, unit: Any
    ) -> "Quantity":
        # Dispatch on the `value` only. Dispatch to the full constructor.
        return cls.constructor(value, unit)

    @classmethod  # type: ignore[no-redef]
    @dispatcher
    def constructor(
        cls: "type[Quantity]", *, value: ArrayLike, unit: Any
    ) -> "Quantity":
        # Dispatched on no argument. Dispatch to the full constructor.
        return cls.constructor(value, unit)

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
    # Quantity

    def to(self, units: Unit) -> "Quantity":
        return replace(self, value=self.value * self.unit.to(units), unit=units)

    def to_value(self, units: Unit) -> ArrayLike:
        if units == self.unit:
            return self.value
        return self.value * self.unit.to(units)  # TODO: allow affine transforms

    def decompose(self, bases: Sequence[Unit]) -> "Quantity":
        """Decompose the quantity into the given bases."""
        du = self.unit.decompose(bases)  # decomposed units
        base_units = CompositeUnit(scale=1, bases=du.bases, powers=du.powers)
        return replace(self, value=self.value * du.scale, unit=base_units)

    # ===============================================================
    # Array API

    def __array_namespace__(self, *, api_version: Any = None) -> "ArrayAPINamespace":
        return array_api_jax_compat

    def __getitem__(self, key: Any) -> "Quantity":
        return replace(self, value=self.value[key])

    __add__ = quaxify(operator.add)
    __radd__ = quaxify(_flip_binop(operator.add))
    __sub__ = quaxify(operator.sub)
    __rsub__ = quaxify(_flip_binop(operator.sub))
    __mul__ = quaxify(operator.mul)
    __rmul__ = quaxify(_flip_binop(operator.mul))
    __matmul__ = quaxify(operator.matmul)
    __rmatmul__ = quaxify(_flip_binop(operator.matmul))
    __pow__ = quaxify(operator.pow)
    __rpow__ = quaxify(_flip_binop(operator.pow))
    __truediv__ = quaxify(operator.truediv)
    __rtruediv__ = quaxify(_flip_binop(operator.truediv))

    # ---------------------------------
    # Boolean operations

    __lt__ = bool_op(jnp.less)
    __le__ = bool_op(jnp.less_equal)
    __eq__ = bool_op(jnp.equal)
    __ge__ = bool_op(jnp.greater_equal)
    __gt__ = bool_op(jnp.greater)
    __ne__ = bool_op(jnp.not_equal)

    def __neg__(self) -> "Quantity":
        return replace(self, value=-self.value)

    # ===============================================================
    # I/O

    def as_type(self, format: type[T], /) -> T:
        """Convert to a type."""
        if format is AstropyQuantity:
            return AstropyQuantity(self.value, self.unit)

        msg = f"Unknown format {format}."
        raise TypeError(msg)


# -----------------------------------------------
# Register additional constructors


@Quantity.constructor._f.register  # noqa: SLF001
def constructor(cls: type[Quantity], value: Quantity, unit: Any, /) -> Quantity:
    """Construct a `Quantity` from another `Quantity`.

    The `value` is converted to the new `unit`.
    """
    return value.to(unit)


@Quantity.constructor._f.register  # type: ignore[no-redef] # noqa: SLF001
def constructor(cls: type[Quantity], value: AstropyQuantity, unit: Any, /) -> Quantity:
    """Construct a `Quantity` from another `Quantity`.

    The `value` is converted to the new `unit`.
    """
    return Quantity(value.to_value(unit), unit)


@Quantity.constructor._f.register  # type: ignore[no-redef] # noqa: SLF001
def constructor(cls: type[Quantity], value: AstropyQuantity) -> Quantity:
    """Construct a `Quantity` from another `Quantity`.

    The `value` is converted to the new `unit`.
    """
    return Quantity(value.value, value.unit)


# ===============================================================


def can_convert(from_: Unit, to: Unit) -> bool:
    """Check if a unit can be converted to another unit.

    Parameters
    ----------
    from_ : Unit
        The unit to convert from.
    to : Unit
        The unit to convert to.

    Returns
    -------
    bool
        Whether the conversion is possible.

    """
    try:
        from_.to(to)
    except UnitConversionError:
        return False
    return True
