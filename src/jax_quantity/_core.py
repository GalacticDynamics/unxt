# pylint: disable=import-error, no-member, unsubscriptable-object
#    b/c it doesn't understand dataclass fields

__all__ = ["Quantity", "can_convert_unit"]

import operator
from collections.abc import Callable, Sequence
from dataclasses import fields, replace
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar, final

import equinox as eqx
import jax
import jax.core
import jax.numpy as jnp
from astropy.units import (
    CompositeUnit,
    PhysicalType,
    Quantity as AstropyQuantity,
    Unit,
    UnitBase,
    UnitConversionError,
    get_physical_type,
)
from jax.numpy import dtype as DType  # noqa: N812
from jaxtyping import Array, ArrayLike, Shaped
from plum import conversion_method, parametric
from quax import ArrayValue, quaxify
from typing_extensions import Self

import quaxed.array_api as xp
from quaxed.array_api._dispatch import dispatcher

if TYPE_CHECKING:
    from array_api import ArrayAPINamespace


FMT = TypeVar("FMT")


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


##############################################################################


@final
@parametric
class Quantity(ArrayValue):  # type: ignore[misc]
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
    >>> from jax_quantity import Quantity

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

    In the previous examples, the dimensions parameter was inferred from the
    values. It can also be given explicitly:

    >>> Quantity["length"](1, "m")
    Quantity['length'](Array(1, dtype=int32, ...), unit='m')

    This can be used for runtime checking of the input dimensions!

    >>> try: Quantity["length"](1, "s")
    ... except Exception as e: print(e)
    Physical type mismatch.

    The dimensions can also be given as a :class:`astropy.units.PhysicalType`:

    >>> dimensions = u.km.physical_type
    >>> dimensions
    PhysicalType('length')
    >>> Quantity[dimensions](1.0, "m")
    Quantity['length'](Array(1., dtype=float32, ...), unit='m')

    Or as a unit:

    >>> Quantity[u.m](1.0, "m")
    Quantity['length'](Array(1., dtype=float32, ...), unit='m')

    Some tricky cases are when the physical type is unknown:

    >>> unit = u.m ** 2 / (u.kg * u.s ** 2)
    >>> unit.physical_type
    PhysicalType('unknown')

    The dimensions can be given as a string in all cases, but is necessary when
    the physical type is unknown:

    >>> Quantity['m2 kg-1 s-2'](1.0, unit)
    Quantity['m2 kg-1 s-2'](Array(1., dtype=float32, ...), unit='m2 / (kg s2)')

    """

    value: Shaped[Array, "*shape"] = eqx.field(converter=jax.numpy.asarray)
    unit: Unit = eqx.field(static=True, converter=Unit)

    def __check_init__(self) -> None:
        """Check whether the arguments are valid."""
        expected_dimensions = self._type_parameter._physical_type_id  # noqa: SLF001
        got_dimensions = self.unit.physical_type._physical_type_id  # noqa: SLF001
        if got_dimensions != expected_dimensions:
            msg = "Physical type mismatch."  # TODO: better error message
            raise ValueError(msg)

    # ---------------------------------------------------------------
    # Plum stuff

    __faithful__: ClassVar[bool] = True
    """Tells :mod:`plum` that this type can be cached more efficiently."""

    @classmethod
    @dispatcher
    def __init_type_parameter__(cls, dimensions: PhysicalType) -> tuple[PhysicalType]:
        """Check whether the type parameters are valid."""
        return (dimensions,)

    @classmethod  # type: ignore[no-redef]
    @dispatcher
    def __init_type_parameter__(cls, dimensions: str) -> tuple[PhysicalType]:
        """Check whether the type parameters are valid."""
        try:
            dims = get_physical_type(dimensions)
        except ValueError:
            dims = PhysicalType(Unit(dimensions), dimensions)
        return (dims,)

    @classmethod  # type: ignore[no-redef]
    @dispatcher
    def __init_type_parameter__(cls, unit: UnitBase) -> tuple[PhysicalType]:
        """Infer the type parameter from the arguments."""
        if unit.physical_type != "unknown":
            return (unit.physical_type,)
        return (PhysicalType(unit, unit.to_string(fraction=False)),)

    @classmethod
    def __infer_type_parameter__(
        cls, value: ArrayLike, unit: Any
    ) -> tuple[PhysicalType]:
        """Infer the type parameter from the arguments."""
        return (Unit(unit).physical_type,)

    @classmethod
    @dispatcher  # type: ignore[misc]
    def __le_type_parameter__(
        cls,
        left: tuple[PhysicalType],
        right: tuple[PhysicalType],
    ) -> bool:
        """Define an order on type parameters.

        That is, check whether `left <= right` or not.
        """
        (dim_left,) = left
        (dim_right,) = right
        return dim_left == dim_right

    def __repr__(self) -> str:
        # fmt: off
        if self._type_parameter == "unknown":
            dim = " ".join(
                f"{unit}{power}" if power != 1 else unit
                for unit, power in self._type_parameter._physical_type_id  # noqa: SLF001
            )
        else:
            dim = self._type_parameter._name_string_as_ordered_set().split("'")[1]  # noqa: SLF001
        return f"Quantity[{dim!r}]({self.value!r}, unit={self.unit.to_string()!r})"
        # fmt: on

    # ---------------------------------------------------------------
    # Constructors

    @classmethod
    @dispatcher
    def constructor(
        cls: "type[Quantity]", value: ArrayLike, unit: Any, /, *, dtype: Any = None
    ) -> "Quantity":
        # Dispatch on both arguments.
        # Construct using the standard `__init__` method.
        return cls(xp.asarray(value, dtype=dtype), unit)

    @classmethod  # type: ignore[no-redef]
    @dispatcher
    def constructor(
        cls: "type[Quantity]", value: ArrayLike, *, unit: Any, dtype: Any = None
    ) -> "Quantity":
        # Dispatch on the `value` only. Dispatch to the full constructor.
        return cls.constructor(value, unit, dtype=dtype)

    @classmethod  # type: ignore[no-redef]
    @dispatcher
    def constructor(
        cls: "type[Quantity]", *, value: ArrayLike, unit: Any, dtype: Any = None
    ) -> "Quantity":
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
        return xp

    def __getitem__(self, key: Any) -> "Quantity":
        return replace(self, value=self.value[key])

    def __len__(self) -> int:
        return len(self.value)

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

    @property
    def dtype(self) -> DType:
        """Data type of the array."""
        return self.value.dtype

    @property
    def device(self) -> jax.Device:
        """Device where the array is located."""
        return self.value.devices().pop()

    @property
    def mT(self) -> "Quantity":  # noqa: N802
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
    def T(self) -> "Quantity":  # noqa: N802
        """Transpose of the array."""
        return replace(self, value=self.value.T)

    def to_device(self, device: None | jax.Device = None) -> "Quantity":
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

    def __neg__(self) -> "Quantity":
        return replace(self, value=-self.value)  # pylint: disable=E1130

    # ---------------------------------
    # Misc

    def flatten(self) -> "Quantity":
        return replace(self, value=self.value.flatten())

    def reshape(self, *args: Any, order: str = "C") -> "Quantity":
        __tracebackhide__ = True  # pylint: disable=unused-variable
        return replace(self, value=self.value.reshape(*args, order=order))

    # ===============================================================
    # Python stuff

    def __hash__(self) -> int:
        """Hash the object as the tuple of its field values.

        This raises a `TypeError` if the object is unhashable,
        which JAX arrays are.

        Examples
        --------
        >>> from jax_quantity import Quantity
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


@Quantity.constructor._f.register  # noqa: SLF001
def constructor(
    cls: type[Quantity], value: Quantity, unit: Any, /, *, dtype: Any = None
) -> Quantity:
    """Construct a `Quantity` from another `Quantity`.

    The `value` is converted to the new `unit`.
    """
    return xp.asarray(value.to(unit), dtype=dtype)


@Quantity.constructor._f.register  # type: ignore[no-redef] # noqa: SLF001
def constructor(
    cls: type[Quantity],
    value: Quantity,
    /,
    *,
    unit: Any | None = None,
    dtype: Any = None,
) -> Quantity:
    """Construct a `Quantity` from another `Quantity`, with no unit change."""
    unit = value.unit if unit is None else unit
    return xp.asarray(value.to(unit), dtype=dtype)


@Quantity.constructor._f.register  # type: ignore[no-redef] # noqa: SLF001
def constructor(
    cls: type[Quantity], value: AstropyQuantity, *, dtype: Any = None
) -> Quantity:
    """Construct a `Quantity` from another `Quantity`.

    The `value` is converted to the new `unit`.
    """
    return Quantity(xp.asarray(value.value, dtype=dtype), value.unit)


@Quantity.constructor._f.register  # type: ignore[no-redef] # noqa: SLF001
def constructor(
    cls: type[Quantity], value: AstropyQuantity, unit: Any, /, *, dtype: Any = None
) -> Quantity:
    """Construct a `Quantity` from another `Quantity`.

    The `value` is converted to the new `unit`.
    """
    return Quantity(xp.asarray(value.to_value(unit), dtype=dtype), unit)


# ===============================================================


def can_convert_unit(from_: Quantity | Unit, to: Unit) -> bool:
    """Check if a unit can be converted to another unit.

    Parameters
    ----------
    from_ : Quantity | Unit
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


# ===============================================================
# Compat


@conversion_method(type_from=Quantity, type_to=AstropyQuantity)  # type: ignore[misc]
def convert_quantity_to_astropyquantity(obj: Quantity, /) -> AstropyQuantity:
    """`Quantity` -> `astropy.Quantity`."""
    return AstropyQuantity(obj.value, obj.unit)
