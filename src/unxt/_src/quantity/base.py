# pylint: disable=import-error, no-member, unsubscriptable-object
#    b/c it doesn't understand dataclass fields

__all__ = ["AbstractQuantity"]

from collections.abc import Callable, Mapping
from functools import partial
from types import ModuleType
from typing import TYPE_CHECKING, Any, ClassVar, NoReturn, TypeAlias, TypeVar

import equinox as eqx
import jax
import jax.core
from astropy.units import UnitConversionError
from jax._src.numpy.array_methods import _IndexUpdateHelper, _IndexUpdateRef
from jaxtyping import Array, ArrayLike, Shaped
from numpy import bool_ as np_bool, dtype as DType, number as np_number  # noqa: N812
from plum import add_promotion_rule, dispatch
from quax import ArrayValue

import quaxed.numpy as jnp
import quaxed.operator as qoperator
from dataclassish import fields, replace

from .api import is_unit_convertible, uconvert, ustrip
from .mixins import AstropyQuantityCompatMixin, IPythonReprMixin, NumPyCompatMixin
from unxt._src.units.core import AbstractUnits, unit as parse_unit

if TYPE_CHECKING:
    from typing_extensions import Self


FMT = TypeVar("FMT")
ArrayLikeScalar: TypeAlias = np_bool | np_number | bool | int | float | complex
ArrayLikeSequence: TypeAlias = list[ArrayLikeScalar] | tuple[ArrayLikeScalar, ...]


def _flip_binop(binop: Callable[[Any, Any], Any]) -> Callable[[Any, Any], Any]:
    def _binop(x: Any, y: Any) -> Any:
        return binop(y, x)

    return _binop


def bool_op(op: Callable[[Any, Any], Any]) -> Callable[[Any, Any], Any]:
    def _op(self: "AbstractQuantity", other: Any) -> Shaped[Array, "..."]:
        return op(self, other)

    return _op


##############################################################################


class AbstractQuantity(
    AstropyQuantityCompatMixin,
    NumPyCompatMixin,
    IPythonReprMixin,
    ArrayValue,  # type: ignore[misc]
):
    """Represents an array, with each axis bound to a name.

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

    From a `numpy.ndarray`:

    >>> import numpy as np
    >>> Quantity(np.array([1, 2, 3]), "m")
    Quantity['length'](Array([1, 2, 3], dtype=int32), unit='m')

    From a `jax.Array`:

    >>> import jax.numpy as jnp
    >>> Quantity(jnp.array([1, 2, 3]), "m")
    Quantity['length'](Array([1, 2, 3], dtype=int32), unit='m')

    The unit can also be given as a `astropy.units.Unit`:

    >>> import astropy.units as apyu
    >>> Quantity(1, apyu.m)
    Quantity['length'](Array(1, dtype=int32, ...), unit='m')

    """

    value: Shaped[Array, "*shape"] = eqx.field(converter=jax.numpy.asarray)
    """The value of the `AbstractQuantity`."""

    unit: AbstractUnits = eqx.field(static=True, converter=parse_unit)
    """The unit associated with this value."""

    # ---------------------------------------------------------------
    # Constructors

    @classmethod
    @dispatch
    def from_(
        cls: "type[AbstractQuantity]",
        value: ArrayLike | ArrayLikeSequence,
        unit: Any,
        /,
        *,
        dtype: Any = None,
    ) -> "AbstractQuantity":
        """Construct a `unxt.Quantity` from an array-like value and a unit.

        :param value: The array-like value.
        :param unit: The unit of the value.
        :param dtype: The data type of the array (keyword-only).

        Examples
        --------
        For this example we'll use the `Quantity` class. The same applies to
        any subclass of `AbstractQuantity`.

        >>> import jax.numpy as jnp
        >>> from unxt import Quantity

        >>> x = jnp.array([1.0, 2, 3])
        >>> Quantity.from_(x, "m")
        Quantity['length'](Array([1., 2., 3.], dtype=float32), unit='m')

        >>> Quantity.from_([1.0, 2, 3], "m")
        Quantity['length'](Array([1., 2., 3.], dtype=float32), unit='m')

        >>> Quantity.from_((1.0, 2, 3), "m")
        Quantity['length'](Array([1., 2., 3.], dtype=float32), unit='m')

        """
        # Dispatch on both arguments.
        # Construct using the standard `__init__` method.
        return cls(jnp.asarray(value, dtype=dtype), unit)

    @classmethod  # type: ignore[no-redef]
    @dispatch
    def from_(
        cls: "type[AbstractQuantity]",
        value: ArrayLike | ArrayLikeSequence,
        /,
        *,
        unit: Any,
        dtype: Any = None,
    ) -> "AbstractQuantity":
        """Make a `unxt.AbstractQuantity` from an array-like value and a unit kwarg.

        Examples
        --------
        For this example we'll use the `unxt.Quantity` class. The same applies
        to any subclass of `unxt.AbstractQuantity`.

        >>> from unxt import Quantity
        >>> Quantity.from_([1.0, 2, 3], unit="m")
        Quantity['length'](Array([1., 2., 3.], dtype=float32), unit='m')

        """
        # Dispatch on the `value` only. Dispatch to the full constructor.
        return cls.from_(value, unit, dtype=dtype)

    @classmethod  # type: ignore[no-redef]
    @dispatch
    def from_(
        cls: "type[AbstractQuantity]", *, value: Any, unit: Any, dtype: Any = None
    ) -> "AbstractQuantity":
        """Construct a `AbstractQuantity` from value and unit kwargs.

        Examples
        --------
        For this example we'll use the `Quantity` class. The same applies to
        any subclass of `AbstractQuantity`.

        >>> from unxt import Quantity
        >>> Quantity.from_(value=[1.0, 2, 3], unit="m")
        Quantity['length'](Array([1., 2., 3.], dtype=float32), unit='m')

        """
        # Dispatched on no argument. Dispatch to the full constructor.
        return cls.from_(value, unit, dtype=dtype)

    @classmethod  # type: ignore[no-redef]
    @dispatch
    def from_(
        cls: "type[AbstractQuantity]", mapping: Mapping[str, Any]
    ) -> "AbstractQuantity":
        """Construct a `Quantity` from a Mapping.

        Parameters
        ----------
        mapping : Mapping[str, Any]
            Mapping of the fields of the `Quantity`, e.g. 'value' and 'unit'.

        Examples
        --------
        For this example we'll use the `Quantity` class. The same applies to
        any subclass of `AbstractQuantity`.

        >>> import jax.numpy as jnp
        >>> from unxt import Quantity

        >>> x = jnp.array([1.0, 2, 3])
        >>> q = Quantity.from_({"value": x, "unit": "m"})
        >>> q
        Quantity['length'](Array([1., 2., 3.], dtype=float32), unit='m')

        >>> Quantity.from_({"value": q, "unit": "km"})
        Quantity['length'](Array([0.001, 0.002, 0.003], dtype=float32), unit='km')

        """
        # Dispatch on both arguments.
        # Construct using the standard `__init__` method.
        return cls.from_(**mapping)

    # See below for additional constructors.

    # ===============================================================
    # Quantity API

    def uconvert(self, u: Any, /) -> "AbstractQuantity":
        """Convert the quantity to the given units.

        See Also
        --------
        unxt.uconvert : convert a quantity to a new unit.

        Examples
        --------
        >>> from unxt import Quantity

        >>> q = Quantity(1, "m")
        >>> q.uconvert("cm")
        Quantity['length'](Array(100., dtype=float32, ...), unit='cm')

        """
        return uconvert(u, self)

    def ustrip(self, u: Any, /) -> Array:
        """Return the value in the given units.

        Examples
        --------
        >>> from unxt import Quantity

        >>> q = Quantity(1, "m")
        >>> q.ustrip("cm")
        Array(100., dtype=float32, weak_type=True)

        """
        return ustrip(u, self)

    # ===============================================================
    # Quax API

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the array."""
        return self.value.shape

    def materialise(self) -> NoReturn:
        msg = "Refusing to materialise `Quantity`."
        raise RuntimeError(msg)

    def aval(self) -> jax.core.ShapedArray:
        return jax.core.get_aval(self.value)

    def enable_materialise(self, _: bool = True) -> "Self":  # noqa: FBT001, FBT002
        return replace(self, value=self.value, unit=self.unit)

    # ===============================================================
    # Plum API

    #: This tells `plum` that this type can be efficiently cached.
    __faithful__: ClassVar[bool] = True

    # ===============================================================
    # Array API

    def __array_namespace__(self, *, api_version: Any = None) -> ModuleType:
        """Return the namespace for the array API.

        Here we return the `quaxed.numpy` module, which is a drop-in replacement
        for `jax.numpy`, but allows for array-ish objects to be used in place of
        `jax` arrays. See `quax` for more information.

        """
        return jnp  # quaxed.numpy

    # ---------------------------------------------------------------
    # attributes

    @property
    def dtype(self) -> DType:
        """Data type of the array.

        Examples
        --------
        >>> from unxt import Quantity
        >>> Quantity(1, "m").dtype
        dtype('int32')

        """
        return self.value.dtype

    @property
    def device(self) -> jax.Device:
        """Device where the array is located.

        Examples
        --------
        >>> from unxt import Quantity
        >>> Quantity(1, "m").device
        CpuDevice(id=0)

        """
        return self.value.devices().pop()

    @property
    def mT(self) -> "AbstractQuantity":  # noqa: N802
        """Transpose of the array."""
        return replace(self, value=jnp.matrix_transpose(self.value))

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

    # ---------------------------------------------------------------
    # arithmetic operators

    def __pos__(self) -> "AbstractQuantity":
        """Return the value of the array.

        Examples
        --------
        >>> import unxt as u
        >>> q = u.Quantity(1, "m")
        >>> +q
        Quantity['length'](Array(1, dtype=int32, ...), unit='m')

        """
        return replace(self, value=+self.value)  # pylint: disable=E1130

    def __neg__(self) -> "AbstractQuantity":
        """Negate the value of the array.

        Examples
        --------
        >>> import unxt as u
        >>> q = u.Quantity(1, "m")
        >>> -q
        Quantity['length'](Array(-1, dtype=int32, ...), unit='m')

        """
        return replace(self, value=-self.value)  # pylint: disable=E1130

    __add__ = qoperator.add
    __radd__ = _flip_binop(qoperator.add)

    __sub__ = qoperator.sub
    __rsub__ = _flip_binop(qoperator.sub)

    __mul__ = qoperator.mul
    __rmul__ = _flip_binop(qoperator.mul)

    __truediv__ = qoperator.truediv
    __rtruediv__ = _flip_binop(qoperator.truediv)

    __floordiv__ = qoperator.floordiv
    __rfloordiv__ = _flip_binop(qoperator.floordiv)

    @dispatch  # type: ignore[misc]
    def __mod__(self: "AbstractQuantity", other: Any) -> "AbstractQuantity":
        """Take the modulus.

        Examples
        --------
        >>> from unxt import Quantity

        >>> q = Quantity(480, "deg")
        >>> q % Quantity(360, "deg")
        Quantity['angle'](Array(120, dtype=int32, ...), unit='deg')

        """
        if not is_unit_convertible(other.unit, self.unit):
            raise UnitConversionError

        # TODO: figure out how to defer to quaxed (e.g. quaxed.operator.mod)
        return replace(self, value=self.value % ustrip(self.unit, other))

    def __rmod__(self, other: Any) -> Any:
        """Take the modulus.

        Examples
        --------
        >>> from unxt import Quantity

        >>> q = Quantity(480, "deg")
        >>> q.__rmod__(Quantity(360, "deg"))
        Quantity['angle'](Array(120, dtype=int32, ...), unit='deg')

        """
        return self % other

    __pow__ = qoperator.pow
    __rpow__ = _flip_binop(qoperator.pow)

    # ---------------------------------------------------------------
    # array operators

    __matmul__ = qoperator.matmul
    __rmatmul__ = _flip_binop(qoperator.matmul)

    # ---------------------------------------------------------------
    # bitwise operators
    # TODO: handle edge cases, e.g. boolean Quantity, not in Astropy

    # __invert__ = qoperator.invert
    # __and__ = qoperator.and_
    # __rand__ = _flip_binop(qoperator.and_)
    # __or__ = qoperator.or_
    # __ror__ = _flip_binop(qoperator.or_)
    # __xor__ = qoperator.xor
    # __rxor__ = _flip_binop(qoperator.xor)
    # __lshift__ = qoperator.lshift
    # __rlshift__ = _flip_binop(qoperator.lshift)
    # __rshift__ = qoperator.rshift
    # __rrshift__ = _flip_binop(qoperator.rshift)

    # ---------------------------------------------------------------
    # comparison operators

    __lt__ = bool_op(jnp.less)
    __le__ = bool_op(jnp.less_equal)
    __eq__ = bool_op(jnp.equal)
    __ge__ = bool_op(jnp.greater_equal)
    __gt__ = bool_op(jnp.greater)
    __ne__ = bool_op(jnp.not_equal)

    # ---------------------------------------------------------------
    # methods

    __abs__ = qoperator.abs

    def __bool__(self) -> bool:
        """Convert a zero-dimensional array to a Python bool object.

        Examples
        --------
        >>> import unxt as u
        >>> q = u.Quantity([0, 1], "m")

        >>> bool(q[0])
        False

        >>> bool(q[1])
        True

        """
        return bool(self.value)

    def __complex__(self) -> complex:
        """Convert the array to a Python complex object.

        Examples
        --------
        >>> import unxt as u
        >>> q = u.Quantity(1, "")
        >>> complex(q)
        (1+0j)

        """
        return complex(ustrip("", self))

    def __dlpack__(self, *args: Any, **kwargs: Any) -> Any:
        """Export the array for consumption as a DLPack capsule."""
        raise NotImplementedError

    def __dlpack_device__(self, *args: Any, **kwargs: Any) -> Any:
        """Return device type and device ID in DLPack format."""
        raise NotImplementedError

    def __float__(self) -> float:
        """Convert the array to a Python float object.

        Examples
        --------
        >>> import unxt as u
        >>> q = u.Quantity(1, "")
        >>> float(q)
        1.0

        """
        return float(ustrip("", self))

    def __getitem__(self, key: Any) -> "AbstractQuantity":
        """Get an item from the array.

        This is a simple wrapper around the `__getitem__` method of the array,
        calling `replace` to only update the value.

        """
        return replace(self, value=self.value[key])

    def __index__(self) -> int:
        """Convert a zero-dimensional integer array to a Python int object.

        Examples
        --------
        >>> import unxt as u
        >>> q = u.Quantity(1, "")
        >>> q.__index__()
        1

        """
        return ustrip("", self).__index__()

    def __int__(self) -> int:
        """Convert a zero-dimensional array to a Python int object.

        Examples
        --------
        >>> import unxt as u
        >>> q = u.Quantity(1, "")
        >>> int(q)
        1

        """
        return int(ustrip("", self))

    def __setitem__(self, key: Any, value: Any) -> None:
        """Set an item in the array.

        This is a simple wrapper around the `__setitem__` method of the array.

        Examples
        --------
        >>> import unxt as u
        >>> q = u.Quantity([1, 2, 3], "m")
        >>> try:
        ...     q[0] = 2
        ... except Exception as e:
        ...     print(e)
        '<class 'jaxlib...ArrayImpl'>' object does not support item assignment...

        """
        self.value[key] = value

    def to_device(self, device: None | jax.Device = None) -> "AbstractQuantity":
        """Move the array to a new device."""
        return replace(self, value=self.value.to_device(device))

    # ===============================================================
    # JAX API

    def __iter__(self) -> Any:
        """Iterate over the Quantity's value.

        Examples
        --------
        >>> from unxt import Quantity
        >>> q = Quantity([1, 2, 3], "m")
        >>> [x for x in q]
        [Quantity['length'](Array(1, dtype=int32), unit='m'),
         Quantity['length'](Array(2, dtype=int32), unit='m'),
         Quantity['length'](Array(3, dtype=int32), unit='m')]

        """
        yield from (self[i] for i in range(len(self.value)))

    def __len__(self) -> int:
        """Return the length of the array.

        Examples
        --------
        >>> from unxt import Quantity

        Length of an unsized array:

        >>> try:
        ...     len(Quantity(1, "m"))
        ... except TypeError as e:
        ...     print(e)
        len() of unsized object

        Length of a sized array:

        >>> len(Quantity([1, 2, 3], "m"))
        3

        """
        return len(self.value)

    def argmax(self, *args: Any, **kwargs: Any) -> Array:
        """Return the indices of the maximum value.

        Examples
        --------
        >>> from unxt import Quantity
        >>> q = Quantity([1, 2, 3], "m")
        >>> q.argmax()
        Array(2, dtype=int32)

        """
        return self.value.argmax(*args, **kwargs)

    def argmin(self, *args: Any, **kwargs: Any) -> Array:
        """Return the indices of the minimum value.

        Examples
        --------
        >>> from unxt import Quantity
        >>> q = Quantity([1, 2, 3], "m")
        >>> q.argmin()
        Array(0, dtype=int32)

        """
        return self.value.argmin(*args, **kwargs)

    @partial(property, doc=jax.Array.at.__doc__)
    def at(self) -> "_QuantityIndexUpdateHelper":
        return _QuantityIndexUpdateHelper(self)

    def block_until_ready(self) -> "AbstractQuantity":
        """Block until the array is ready.

        Examples
        --------
        >>> from unxt import Quantity
        >>> q = Quantity(1, "m")
        >>> q.block_until_ready() is q
        True

        """
        _ = self.value.block_until_ready()
        return self

    def flatten(self) -> "AbstractQuantity":
        """Return a flattened version of the array.

        Examples
        --------
        >>> from unxt import Quantity
        >>> q = Quantity([[1, 2], [3, 4]], "m")
        >>> q.flatten()
        Quantity['length'](Array([1, 2, 3, 4], dtype=int32), unit='m')

        """
        return replace(self, value=self.value.flatten())

    def ravel(self) -> "AbstractQuantity":
        """Return a flattened version of the array.

        Examples
        --------
        >>> from unxt import Quantity
        >>> q = Quantity([[1, 2], [3, 4]], "m")
        >>> q.ravel()
        Quantity['length'](Array([1, 2, 3, 4], dtype=int32), unit='m')

        """
        return replace(self, value=self.value.ravel())

    def reshape(self, *args: Any, order: str = "C") -> "AbstractQuantity":
        """Return a reshaped version of the array.

        Examples
        --------
        >>> import unxt as u
        >>> q = u.Quantity([1, 2, 3, 4], "m")
        >>> q.reshape(2, 2)
        Quantity['length'](Array([[1, 2],
                                  [3, 4]], dtype=int32), unit='m')

        """
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
        >>> from unxt import Quantity
        >>> q1 = Quantity(1, "m")
        >>> try:
        ...     hash(q1)
        ... except TypeError as e:
        ...     print(e)
        unhashable type: ...

        """
        return hash(tuple(getattr(self, f.name) for f in fields(self)))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.value!r}, unit={self.unit.to_string()!r})"


# -----------------------------------------------
# Register additional constructors


@AbstractQuantity.from_._f.register  # noqa: SLF001
def from_(
    cls: type[AbstractQuantity],
    value: AbstractQuantity,
    unit: Any,
    /,
    *,
    dtype: Any = None,
) -> AbstractQuantity:
    """Construct a `Quantity` from another `Quantity`.

    The `value` is converted to the new `unit`.

    Examples
    --------
    >>> from unxt import Quantity

    >>> q = Quantity(1, "m")
    >>> Quantity.from_(q, "cm")
    Quantity['length'](Array(100., dtype=float32, ...), unit='cm')

    """
    value = jnp.asarray(uconvert(unit, value), dtype=dtype)
    return cls(value.value, unit)


@AbstractQuantity.from_._f.register  # type: ignore[no-redef]  # noqa: SLF001
def from_(
    cls: type[AbstractQuantity],
    value: AbstractQuantity,
    unit: None,
    /,
    *,
    dtype: Any = None,
) -> AbstractQuantity:
    """Construct a `Quantity` from another `Quantity`.

    The `value` is converted to the new `unit`.

    Examples
    --------
    >>> from unxt import Quantity

    >>> q = Quantity(1, "m")
    >>> Quantity.from_(q, None)
    Quantity['length'](Array(1, dtype=int32, ...), unit='m')

    """
    value = jnp.asarray(value, dtype=dtype)
    return cls(value.value, value.unit)


@AbstractQuantity.from_._f.register  # type: ignore[no-redef] # noqa: SLF001
def from_(
    cls: type[AbstractQuantity],
    value: AbstractQuantity,
    /,
    *,
    unit: Any | None = None,
    dtype: Any = None,
) -> AbstractQuantity:
    """Construct a `Quantity` from another `Quantity`, with no unit change."""
    unit = value.unit if unit is None else unit
    value = jnp.asarray(uconvert(unit, value), dtype=dtype)
    return cls(value.value, unit)


# -----------------------------------------------
# Promotion rules

add_promotion_rule(AbstractQuantity, AbstractQuantity, AbstractQuantity)

# ===============================================================
# Support for ``at``.


# `_QuantityIndexUpdateHelper` is defined up here because it is used in the
# runtime-checkable type annotation in `AbstractQuantity.at`.
# `_QuantityIndexUpdateRef` is defined after `AbstractQuantity` because it
# references `AbstractQuantity` in its runtime-checkable type annotations.
class _QuantityIndexUpdateHelper(_IndexUpdateHelper):  # type: ignore[misc]
    def __getitem__(self, index: Any) -> "_IndexUpdateRef":
        return _QuantityIndexUpdateRef(self.array, index)

    def __repr__(self) -> str:
        """Return a string representation of the object.

        Examples
        --------
        >>> from unxt import Quantity
        >>> q = Quantity([1, 2, 3, 4], "m")
        >>> q.at
        _QuantityIndexUpdateHelper(Quantity['length'](Array([1, 2, 3, 4], dtype=int32), unit='m'))

        """  # noqa: E501
        return f"_QuantityIndexUpdateHelper({self.array!r})"


class _QuantityIndexUpdateRef(_IndexUpdateRef):  # type: ignore[misc]
    # This is a subclass of `_IndexUpdateRef` that is used to implement the `at`
    # attribute of `AbstractQuantity`. See also `_QuantityIndexUpdateHelper`.

    def __repr__(self) -> str:
        return super().__repr__().replace("_IndexUpdateRef", "_QuantityIndexUpdateRef")

    def get(
        self,
        *,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        mode: jax.lax.GatherScatterMode | None = None,
        fill_value: AbstractQuantity | None = None,
    ) -> AbstractQuantity:
        # TODO: by quaxified super
        value = self.array.value.at[self.index].get(
            indices_are_sorted=indices_are_sorted,
            unique_indices=unique_indices,
            mode=mode,
            fill_value=fill_value
            if fill_value is None
            else ustrip(self.array.unit, fill_value),
        )
        return replace(self.array, value=value)

    def set(
        self,
        values: AbstractQuantity,
        *,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        mode: None = None,
    ) -> AbstractQuantity:
        # TODO: by quaxified super
        value = self.array.value.at[self.index].set(
            ustrip(self.array.unit, values),
            indices_are_sorted=indices_are_sorted,
            unique_indices=unique_indices,
            mode=mode,
        )
        return replace(self.array, value=value)

    def apply(
        self,
        func: Any,
        *,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        mode: jax.lax.GatherScatterMode | None = None,
    ) -> AbstractQuantity:
        raise NotImplementedError  # TODO: by quaxified super

    def add(
        self,
        values: AbstractQuantity,
        *,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        mode: jax.lax.GatherScatterMode | None = None,
    ) -> AbstractQuantity:
        # TODO: by quaxified super
        value = self.array.value.at[self.index].add(
            ustrip(self.array.unit, values),
            indices_are_sorted=indices_are_sorted,
            unique_indices=unique_indices,
            mode=mode,
        )
        return replace(self.array, value=value)

    def multiply(
        self,
        values: ArrayLike,
        *,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        mode: jax.lax.GatherScatterMode | None = None,
    ) -> AbstractQuantity:
        values = eqx.error_if(
            values, isinstance(values, AbstractQuantity), "values cannot be a Quantity"
        )  # TODO: can permit dimensionless quantities.

        # TODO: by quaxified super
        value = self.array.value.at[self.index].multiply(
            values,
            indices_are_sorted=indices_are_sorted,
            unique_indices=unique_indices,
            mode=mode,
        )
        return replace(self.array, value=value)

    mul = multiply

    def divide(
        self,
        values: ArrayLike,
        *,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        mode: jax.lax.GatherScatterMode | None = None,
    ) -> AbstractQuantity:
        values = eqx.error_if(
            values, isinstance(values, AbstractQuantity), "values cannot be a Quantity"
        )  # TODO: can permit dimensionless quantities.

        # TODO: by quaxified super
        value = self.array.value.at[self.index].divide(
            values,
            indices_are_sorted=indices_are_sorted,
            unique_indices=unique_indices,
            mode=mode,
        )
        return replace(self.array, value=value)

    def power(
        self,
        values: ArrayLike,
        *,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        mode: jax.lax.GatherScatterMode | None = None,
    ) -> AbstractQuantity:
        raise NotImplementedError

    def min(
        self,
        values: AbstractQuantity,
        *,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        mode: jax.lax.GatherScatterMode | None = None,
    ) -> AbstractQuantity:
        # TODO: by quaxified super
        value = self.array.value.at[self.index].min(
            ustrip(self.array.unit, values),
            indices_are_sorted=indices_are_sorted,
            unique_indices=unique_indices,
            mode=mode,
        )
        return replace(self.array, value=value)

    def max(
        self,
        values: AbstractQuantity,
        *,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        mode: jax.lax.GatherScatterMode | None = None,
    ) -> AbstractQuantity:
        # TODO: by quaxified super
        value = self.array.value.at[self.index].max(
            ustrip(self.array.unit, values),
            indices_are_sorted=indices_are_sorted,
            unique_indices=unique_indices,
            mode=mode,
        )
        return replace(self.array, value=value)
