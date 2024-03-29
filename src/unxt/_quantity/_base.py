# pylint: disable=import-error, no-member, unsubscriptable-object
#    b/c it doesn't understand dataclass fields

__all__ = ["AbstractQuantity", "can_convert_unit"]

from collections.abc import Callable, Sequence
from dataclasses import fields, replace
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

import equinox as eqx
import jax
import jax.core
import jax.numpy as jnp
from astropy.units import (
    CompositeUnit,
    Quantity as AstropyQuantity,
    Unit,
    UnitConversionError,
)
from jax.numpy import dtype as DType  # noqa: N812
from jaxtyping import Array, ArrayLike, Shaped
from plum import add_promotion_rule
from quax import ArrayValue
from typing_extensions import Self

import quaxed.array_api as xp
import quaxed.operator as qoperator
from quaxed.array_api._dispatch import dispatcher

if TYPE_CHECKING:
    from array_api import ArrayAPINamespace


FMT = TypeVar("FMT")


def _flip_binop(binop: Callable[[Any, Any], Any]) -> Callable[[Any, Any], Any]:
    def _binop(x: Any, y: Any) -> Any:
        return binop(y, x)

    return _binop


def bool_op(op: Callable[[Any, Any], Any]) -> Callable[[Any, Any], Any]:
    # TODO: support operations on dimensionless quantities with array.
    def _op(self: "AbstractQuantity", other: Any) -> Shaped[Array, "..."]:
        if not isinstance(other, AbstractQuantity):
            return NotImplemented

        try:
            other_value = other.to_value(self.unit)
        except UnitConversionError:
            return jnp.full(self.value.shape, fill_value=False, dtype=bool)

        return op(self.value, other_value)

    return _op


##############################################################################


class AbstractQuantity(ArrayValue):  # type: ignore[misc]
    """Represents an array, with each axis bound to a name.

    Parameters
    ----------
    value : array-like
        The array of values. Anything that can be converted to an array by
        `jax.numpy.asarray`.
    unit : Unit-like
        The unit of the array. Anything that can be converted to a unit by
        `astropy.units.Unit`.

    Examples
    --------
    >>> from unxt import Quantity

    From an integer:

    >>> Quantity(1, "m")
    Quantity['length'](Array(1, dtype=int32, ...), unit='m')

    From a float:

    >>> Quantity(1.0, "m")
    Quantity['length'](Array(1., dtype=float32, ...), unit='m')

    From a list:

    >>> Quantity([1, 2, 3], "m")
    Quantity['length'](Array([1, 2, 3], dtype=int32), unit='m')

    From a tuple:

    >>> Quantity((1, 2, 3), "m")
    Quantity['length'](Array([1, 2, 3], dtype=int32), unit='m')

    From a :class:`numpy.ndarray`:

    >>> import numpy as np
    >>> Quantity(np.array([1, 2, 3]), "m")
    Quantity['length'](Array([1, 2, 3], dtype=int32), unit='m')

    From a :class:`jax.Array`:

    >>> import jax.numpy as jnp
    >>> Quantity(jnp.array([1, 2, 3]), "m")
    Quantity['length'](Array([1, 2, 3], dtype=int32), unit='m')

    The unit can also be given as a :class:`astropy.units.Unit`:

    >>> import astropy.units as u
    >>> Quantity(1, u.m)
    Quantity['length'](Array(1, dtype=int32, ...), unit='m')

    """

    value: Shaped[Array, "*shape"] = eqx.field(converter=jax.numpy.asarray)
    unit: Unit = eqx.field(static=True, converter=Unit)

    # ---------------------------------------------------------------
    # Plum stuff

    __faithful__: ClassVar[bool] = True
    """Tells :mod:`plum` that this type can be cached more efficiently."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.value!r}, unit={self.unit.to_string()!r})"

    # ---------------------------------------------------------------
    # Constructors

    @classmethod
    @dispatcher
    def constructor(
        cls: "type[AbstractQuantity]",
        value: ArrayLike,
        unit: Any,
        /,
        *,
        dtype: Any = None,
    ) -> "AbstractQuantity":
        # Dispatch on both arguments.
        # Construct using the standard `__init__` method.
        return cls(xp.asarray(value, dtype=dtype), unit)

    @classmethod  # type: ignore[no-redef]
    @dispatcher
    def constructor(
        cls: "type[AbstractQuantity]", value: ArrayLike, *, unit: Any, dtype: Any = None
    ) -> "AbstractQuantity":
        # Dispatch on the `value` only. Dispatch to the full constructor.
        return cls.constructor(value, unit, dtype=dtype)

    @classmethod  # type: ignore[no-redef]
    @dispatcher
    def constructor(
        cls: "type[AbstractQuantity]", *, value: ArrayLike, unit: Any, dtype: Any = None
    ) -> "AbstractQuantity":
        # Dispatched on no argument. Dispatch to the full constructor.
        return cls.constructor(value, unit, dtype=dtype)

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
        return replace(self, value=self.value, unit=self.unit)

    # ===============================================================
    # Quantity

    def to(self, units: Unit) -> "AbstractQuantity":
        return replace(self, value=self.value * self.unit.to(units), unit=units)

    def to_value(self, units: Unit) -> ArrayLike:
        if units == self.unit:
            return self.value
        return self.value * self.unit.to(units)  # TODO: allow affine transforms

    def decompose(self, bases: Sequence[Unit]) -> "AbstractQuantity":
        """Decompose the quantity into the given bases."""
        du = self.unit.decompose(bases)  # decomposed units
        base_units = CompositeUnit(scale=1, bases=du.bases, powers=du.powers)
        return replace(self, value=self.value * du.scale, unit=base_units)

    # ===============================================================
    # Array API

    def __array_namespace__(self, *, api_version: Any = None) -> "ArrayAPINamespace":
        return xp

    def __getitem__(self, key: Any) -> "AbstractQuantity":
        return replace(self, value=self.value[key])

    def __len__(self) -> int:
        return len(self.value)

    __abs__ = qoperator.abs

    __add__ = qoperator.add
    __radd__ = _flip_binop(qoperator.add)

    __sub__ = qoperator.sub
    __rsub__ = _flip_binop(qoperator.sub)

    __mul__ = qoperator.mul
    __rmul__ = _flip_binop(qoperator.mul)

    __matmul__ = qoperator.matmul
    __rmatmul__ = _flip_binop(qoperator.matmul)

    __pow__ = qoperator.pow
    __rpow__ = _flip_binop(qoperator.pow)

    __truediv__ = qoperator.truediv
    __rtruediv__ = _flip_binop(qoperator.truediv)

    @property
    def dtype(self) -> DType:
        """Data type of the array."""
        return self.value.dtype

    @property
    def device(self) -> jax.Device:
        """Device where the array is located."""
        return self.value.devices().pop()

    @property
    def mT(self) -> "AbstractQuantity":  # noqa: N802
        """Transpose of the array."""
        return replace(self, value=xp.matrix_transpose(self.value))

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self.value.ndim

    @property
    def size(self) -> int:
        """Total number of elements."""
        return self.value.size

    @property
    def T(self) -> "AbstractQuantity":  # noqa: N802
        """Transpose of the array."""
        return replace(self, value=self.value.T)

    def to_device(self, device: None | jax.Device = None) -> "AbstractQuantity":
        """Move the array to a new device."""
        return replace(self, value=self.value.to_device(device))

    # ---------------------------------
    # Boolean operations

    __lt__ = bool_op(jnp.less)
    __le__ = bool_op(jnp.less_equal)
    __eq__ = bool_op(jnp.equal)
    __ge__ = bool_op(jnp.greater_equal)
    __gt__ = bool_op(jnp.greater)
    __ne__ = bool_op(jnp.not_equal)

    def __neg__(self) -> "AbstractQuantity":
        return replace(self, value=-self.value)  # pylint: disable=E1130

    # ---------------------------------
    # Misc

    def flatten(self) -> "AbstractQuantity":
        return replace(self, value=self.value.flatten())

    def reshape(self, *args: Any, order: str = "C") -> "AbstractQuantity":
        __tracebackhide__ = True  # pylint: disable=unused-variable
        return replace(self, value=self.value.reshape(*args, order=order))

    def __mod__(self, other: Any) -> "Self":
        """Take the mod.

        Examples
        --------
        >>> from unxt import Quantity

        >>> q = Quantity(480, "deg")
        >>> q % Quantity(360, "deg")
        Quantity['angle'](Array(120, dtype=int32, ...), unit='deg')

        """
        if not isinstance(other, AbstractQuantity):
            return NotImplemented

        if not can_convert_unit(other.unit, self.unit):
            raise UnitConversionError

        # TODO: figure out how to defer to quaxed (e.g. quaxed.operator.mod)
        return replace(self, value=self.value % other.to_value(self.unit))

    # ===============================================================
    # Python stuff

    def __hash__(self) -> int:
        """Hash the object as the tuple of its field values.

        This raises a `TypeError` if the object is unhashable,
        which JAX arrays are.

        Examples
        --------
        >>> from unxt import Quantity
        >>> q1 = Quantity(1, "m")
        >>> try: hash(q1)
        ... except TypeError as e: print(e)
        unhashable type: 'ArrayImpl'

        """
        return hash(tuple(getattr(self, f.name) for f in fields(self)))

    # ===============================================================
    # I/O

    def convert_to(self, format: type[FMT], /) -> FMT:
        """Convert to a type."""
        if format is AstropyQuantity:
            return AstropyQuantity(self.value, self.unit)

        msg = f"Unknown format {format}."
        raise TypeError(msg)


# -----------------------------------------------
# Register additional constructors


@AbstractQuantity.constructor._f.register  # noqa: SLF001
def constructor(
    cls: type[AbstractQuantity],
    value: AbstractQuantity,
    unit: Any,
    /,
    *,
    dtype: Any = None,
) -> AbstractQuantity:
    """Construct a `Quantity` from another `Quantity`.

    The `value` is converted to the new `unit`.
    """
    value = xp.asarray(value.to(unit), dtype=dtype)
    return cls(value.value, unit)


@AbstractQuantity.constructor._f.register  # type: ignore[no-redef] # noqa: SLF001
def constructor(
    cls: type[AbstractQuantity],
    value: AbstractQuantity,
    /,
    *,
    unit: Any | None = None,
    dtype: Any = None,
) -> AbstractQuantity:
    """Construct a `Quantity` from another `Quantity`, with no unit change."""
    unit = value.unit if unit is None else unit
    value = xp.asarray(value.to(unit), dtype=dtype)
    return cls(value.value, unit)


@AbstractQuantity.constructor._f.register  # type: ignore[no-redef] # noqa: SLF001
def constructor(
    cls: type[AbstractQuantity], value: AstropyQuantity, *, dtype: Any = None
) -> AbstractQuantity:
    """Construct a `Quantity` from another `Quantity`.

    The `value` is converted to the new `unit`.
    """
    return cls(xp.asarray(value.value, dtype=dtype), value.unit)


@AbstractQuantity.constructor._f.register  # type: ignore[no-redef] # noqa: SLF001
def constructor(
    cls: type[AbstractQuantity],
    value: AstropyQuantity,
    unit: Any,
    /,
    *,
    dtype: Any = None,
) -> AbstractQuantity:
    """Construct a `Quantity` from another `Quantity`.

    The `value` is converted to the new `unit`.
    """
    return cls(xp.asarray(value.to_value(unit), dtype=dtype), unit)


# -----------------------------------------------
# Promotion rules

add_promotion_rule(AbstractQuantity, AbstractQuantity, AbstractQuantity)


# ===============================================================


def can_convert_unit(from_: AbstractQuantity | Unit, to: Unit) -> bool:
    """Check if a unit can be converted to another unit.

    Parameters
    ----------
    from_ : :clas:`unxt.AbstractQuantity` | Unit
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
