# pylint: disable=import-error, no-member, unsubscriptable-object
#    b/c it doesn't understand dataclass fields

__all__ = ["AbstractQuantity", "is_any_quantity"]

from collections.abc import Mapping
from functools import partial
from types import ModuleType
from typing import TYPE_CHECKING, Any, ClassVar, NoReturn, TypeAlias, TypeGuard
from typing_extensions import override

import equinox as eqx
import jax
import jax.core
import quax_blocks
from astropy.units import UnitConversionError
from jax._src.numpy.array_methods import _IndexUpdateHelper, _IndexUpdateRef
from jaxtyping import Array, ArrayLike, Bool, ScalarLike, Shaped
from plum import add_promotion_rule, dispatch
from quax import ArrayValue

import quaxed.numpy as jnp
from dataclassish import replace

from .api import is_unit_convertible, uconvert, ustrip
from .mixins import AstropyQuantityCompatMixin, IPythonReprMixin, NumPyCompatMixin
from unxt._src.units import AstropyUnits
from unxt.units import unit_of

if TYPE_CHECKING:
    from typing import Self


ArrayLikeSequence: TypeAlias = list[ScalarLike] | tuple[ScalarLike, ...]


class AbstractQuantity(
    AstropyQuantityCompatMixin,
    NumPyCompatMixin,
    IPythonReprMixin,
    ArrayValue,
    quax_blocks.NumpyBinaryOpsMixin[Any, "AbstractQuantity"],
    quax_blocks.NumpyComparisonMixin[Any, Bool[Array, "*shape"]],  # TODO: shape hint
    quax_blocks.NumpyUnaryMixin["AbstractQuantity"],
    quax_blocks.NumpyRoundMixin["AbstractQuantity"],
    quax_blocks.NumpyTruncMixin["AbstractQuantity"],
    quax_blocks.NumpyFloorMixin["AbstractQuantity"],
    quax_blocks.NumpyCeilMixin["AbstractQuantity"],
    quax_blocks.LaxLenMixin,
    quax_blocks.LaxLengthHintMixin,
):
    """Represents an array, with each axis bound to a name.

    Examples
    --------
    >>> import unxt as u

    From an integer:

    >>> u.Quantity(1, "m")
    Quantity['length'](Array(1, dtype=int32, ...), unit='m')

    From a float:

    >>> u.Quantity(1.0, "m")
    Quantity['length'](Array(1., dtype=float32, ...), unit='m')

    From a list:

    >>> u.Quantity([1, 2, 3], "m")
    Quantity['length'](Array([1, 2, 3], dtype=int32), unit='m')

    From a tuple:

    >>> u.Quantity((1, 2, 3), "m")
    Quantity['length'](Array([1, 2, 3], dtype=int32), unit='m')

    From a `numpy.ndarray`:

    >>> import numpy as np
    >>> u.Quantity(np.array([1, 2, 3]), "m")
    Quantity['length'](Array([1, 2, 3], dtype=int32), unit='m')

    From a `jax.Array`:

    >>> import jax.numpy as jnp
    >>> u.Quantity(jnp.array([1, 2, 3]), "m")
    Quantity['length'](Array([1, 2, 3], dtype=int32), unit='m')

    The unit can also be given as a `astropy.units.Unit`:

    >>> import astropy.units as apyu
    >>> u.Quantity(1, apyu.m)
    Quantity['length'](Array(1, dtype=int32, ...), unit='m')

    """

    value: eqx.AbstractVar[Shaped[Array, "*shape"]]
    """The value of the `AbstractQuantity`."""

    unit: eqx.AbstractVar[AstropyUnits]
    """The unit associated with this value."""

    # ---------------------------------------------------------------
    # Constructors

    @classmethod
    @dispatch.abstract
    def from_(
        cls: "type[AbstractQuantity]",
        *args: Any,
        **kwargs: Any,
    ) -> "AbstractQuantity":
        raise NotImplementedError  # pragma: no cover

    # See below for additional constructors

    # ===============================================================
    # Quantity API

    def uconvert(self, u: Any, /) -> "AbstractQuantity":
        """Convert the quantity to the given units.

        See Also
        --------
        `unxt.uconvert` : convert a quantity to a new unit.

        Examples
        --------
        >>> import unxt as u

        >>> q = u.Quantity(1, "m")
        >>> q.uconvert("cm")
        Quantity['length'](Array(100., dtype=float32, ...), unit='cm')

        """
        return uconvert(u, self)

    def ustrip(self, u: Any, /) -> Array:
        """Return the value in the given units.

        See Also
        --------
        `unxt.ustrip` : strip the units from a quantity.

        Examples
        --------
        >>> import unxt as u

        >>> q = u.Quantity(1, "m")
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
        msg = f"Refusing to materialise `{type(self).__name__}`."
        raise RuntimeError(msg)

    def aval(self) -> jax.core.ShapedArray:
        return jax.core.get_aval(self.value)  # type: ignore[no-untyped-call]

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
    def dtype(self) -> jax.numpy.dtype:
        """Data type of the array.

        Examples
        --------
        >>> import unxt as u
        >>> u.Quantity(1, "m").dtype
        dtype('int32')

        """
        return self.value.dtype

    @property
    def device(self) -> jax.Device:
        """Device where the array is located.

        Examples
        --------
        >>> import unxt as u
        >>> u.Quantity(1, "m").device
        CpuDevice(id=0)

        """
        return self.value.devices().pop()

    @property
    def mT(self) -> "AbstractQuantity":  # noqa: N802
        """Matrix transpose of the array.

        Examples
        --------
        >>> import unxt as u
        >>> q = u.Quantity([[0, 1], [1, 2]], "m")
        >>> q.mT
        Quantity['length'](Array([[0, 1],
                                  [1, 2]], dtype=int32), unit='m')

        """
        return replace(self, value=jnp.matrix_transpose(self.value))

    @property
    def ndim(self) -> int:
        """Number of dimensions.

        Examples
        --------
        >>> import unxt as u
        >>> q = u.Quantity([[1]], "m")
        >>> q.ndim
        2

        """
        return self.value.ndim

    @property
    def size(self) -> int:
        """Total number of elements.

        Examples
        --------
        >>> import unxt as u
        >>> q = u.Quantity([1, 2, 3], "m")
        >>> q.size
        3

        """
        return self.value.size

    @property
    def T(self) -> "AbstractQuantity":  # noqa: N802
        """Transpose of the array.

        Examples
        --------
        >>> import unxt as u
        >>> q = u.Quantity([[0, 1], [1, 2]], "m")
        >>> q.T
        Quantity['length'](Array([[0, 1],
                                  [1, 2]], dtype=int32), unit='m')

        """
        return replace(self, value=self.value.T)

    # ---------------------------------------------------------------
    # arithmetic operators

    @dispatch
    def __mod__(self: "AbstractQuantity", other: Any) -> "AbstractQuantity":
        """Take the modulus.

        Examples
        --------
        >>> import unxt as u

        >>> q = u.Quantity(480, "deg")
        >>> q % u.Quantity(360, "deg")
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
        >>> import unxt as u

        >>> q = u.Quantity(480, "deg")
        >>> q.__rmod__(u.Quantity(360, "deg"))
        Quantity['angle'](Array(120, dtype=int32, ...), unit='deg')

        """
        return self % other

    # required to override mixin methods
    __eq__ = quax_blocks.NumpyEqMixin.__eq__

    # ---------------------------------------------------------------
    # methods

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
        ...     print("jax arrays do not support in-place item assignment")
        jax arrays do not support in-place item assignment

        """
        self.value[key] = value

    def to_device(self, device: None | jax.Device = None) -> "AbstractQuantity":
        """Move the array to a new device.

        Examples
        --------
        >>> import unxt as u
        >>> q = u.Quantity(1, "m")
        >>> q.to_device(None)
        Quantity['length'](Array(1, dtype=int32, weak_type=True), unit='m')

        """
        return replace(self, value=self.value.to_device(device))

    # ===============================================================
    # JAX API

    def __iter__(self) -> Any:
        """Iterate over the Quantity's value.

        Examples
        --------
        >>> import unxt as u
        >>> q = u.Quantity([1, 2, 3], "m")
        >>> [x for x in q]
        [Quantity['length'](Array(1, dtype=int32), unit='m'),
         Quantity['length'](Array(2, dtype=int32), unit='m'),
         Quantity['length'](Array(3, dtype=int32), unit='m')]

        """
        yield from (self[i] for i in range(len(self.value)))

    def argmax(self, *args: Any, **kwargs: Any) -> Array:
        """Return the indices of the maximum value.

        Examples
        --------
        >>> import unxt as u
        >>> q = u.Quantity([1, 2, 3], "m")
        >>> q.argmax()
        Array(2, dtype=int32)

        """
        return self.value.argmax(*args, **kwargs)

    def argmin(self, *args: Any, **kwargs: Any) -> Array:
        """Return the indices of the minimum value.

        Examples
        --------
        >>> import unxt as u
        >>> q = u.Quantity([1, 2, 3], "m")
        >>> q.argmin()
        Array(0, dtype=int32)

        """
        return self.value.argmin(*args, **kwargs)

    def astype(self, *args: Any, **kwargs: Any) -> "AbstractQuantity":
        """Copy the array and cast to a specified dtype.

        Examples
        --------
        >>> import unxt as u
        >>> q = u.Quantity([1, 2, 3], "m")
        >>> q.dtype
        dtype('int32')

        >>> q.astype(float)
        Quantity['length'](Array([1., 2., 3.], dtype=float32), unit='m')

        """
        return replace(self, value=self.value.astype(*args, **kwargs))

    @partial(property, doc=jax.Array.at.__doc__)
    def at(self) -> "_QuantityIndexUpdateHelper":
        return _QuantityIndexUpdateHelper(self)  # type: ignore[no-untyped-call]

    def block_until_ready(self) -> "AbstractQuantity":
        """Block until the array is ready.

        Examples
        --------
        >>> import unxt as u
        >>> q = u.Quantity(1, "m")
        >>> q.block_until_ready() is q
        True

        """
        _ = self.value.block_until_ready()
        return self

    def devices(self) -> set[jax.Device]:
        """Return the devices where the array is located.

        Examples
        --------
        >>> import unxt as u
        >>> q = u.Quantity(1, "m")
        >>> q.devices()
        {CpuDevice(id=0)}

        """
        return self.value.devices()

    def flatten(self) -> "AbstractQuantity":
        """Return a flattened version of the array.

        Examples
        --------
        >>> import unxt as u
        >>> q = u.Quantity([[1, 2], [3, 4]], "m")
        >>> q.flatten()
        Quantity['length'](Array([1, 2, 3, 4], dtype=int32), unit='m')

        """
        return replace(self, value=self.value.flatten())

    def max(self, *args: Any, **kwargs: Any) -> "AbstractQuantity":
        """Return the maximum value.

        Examples
        --------
        >>> import unxt as u
        >>> q = u.Quantity([1, 2, 3], "m")
        >>> q.max()
        Quantity['length'](Array(3, dtype=int32), unit='m')

        """
        return replace(self, value=self.value.max(*args, **kwargs))

    def mean(self, *args: Any, **kwargs: Any) -> "AbstractQuantity":
        """Return the mean value.

        Examples
        --------
        >>> import unxt as u
        >>> q = u.Quantity([1, 2, 3], "m")
        >>> q.mean()
        Quantity['length'](Array(2., dtype=float32), unit='m')

        """
        return replace(self, value=self.value.mean(*args, **kwargs))

    def min(self, *args: Any, **kwargs: Any) -> "AbstractQuantity":
        """Return the minimum value.

        Examples
        --------
        >>> import unxt as u
        >>> q = u.Quantity([1, 2, 3], "m")
        >>> q.min()
        Quantity['length'](Array(1, dtype=int32), unit='m')

        """
        return replace(self, value=self.value.min(*args, **kwargs))

    def ravel(self) -> "AbstractQuantity":
        """Return a flattened version of the array.

        Examples
        --------
        >>> import unxt as u
        >>> q = u.Quantity([[1, 2], [3, 4]], "m")
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

    def round(self, *args: Any, **kwargs: Any) -> "AbstractQuantity":
        """Round the array to the given number of decimals.

        Examples
        --------
        >>> import unxt as u
        >>> q = u.Quantity([1.1, 2.2, 3.3], "m")
        >>> q.round(0)
        Quantity['length'](Array([1., 2., 3.], dtype=float32), unit='m')

        """
        return replace(self, value=self.value.round(*args, **kwargs))

    @property
    def sharding(self) -> Any:
        """Return the sharding configuration of the array.

        Examples
        --------
        >>> import unxt as u
        >>> q = u.Quantity([1, 2, 3], "m")
        >>> q.sharding
        SingleDeviceSharding(device=..., memory_kind=...)

        """
        return self.value.sharding

    def squeeze(self, *args: Any, **kwargs: Any) -> "AbstractQuantity":
        """Return the array with all single-dimensional entries removed.

        Examples
        --------
        >>> import unxt as u
        >>> q = u.Quantity([[[1], [2], [3]]], "m")
        >>> q.squeeze()
        Quantity['length'](Array([1, 2, 3], dtype=int32), unit='m')

        """
        return replace(self, value=self.value.squeeze(*args, **kwargs))

    # ===============================================================
    # Python stuff

    def __hash__(self) -> int:
        """Return the hash of the quantity.

        This is the hash of the value and the unit, however since the value is
        generally not hashable this will generally raise an exception.
        Defining the `__hash__` method is required for the `AbstractQuantity` to
        be considered immutable, e.g. by `dataclasses.dataclass`.

        Examples
        --------
        >>> import unxt as u
        >>> q = u.Quantity(1, "m")
        >>> try:
        ...     hash(q)
        ... except TypeError as e:
        ...     print(e)
        unhashable type: 'jaxlib.xla_extension.ArrayImpl'

        """
        return hash((self.value, self.unit))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.value!r}, unit={self.unit.to_string()!r})"


# -----------------------------------------------
# Register additional constructors


@AbstractQuantity.from_.dispatch
def from_(
    cls: type[AbstractQuantity],
    value: ArrayLike | ArrayLikeSequence,
    unit: Any,
    /,
    *,
    dtype: Any = None,
) -> AbstractQuantity:
    """Construct a `unxt.Quantity` from an array-like value and a unit.

    :param value: The array-like value.
    :param unit: The unit of the value.
    :param dtype: The data type of the array (keyword-only).

    Examples
    --------
    For this example we'll use the `Quantity` class. The same applies to
    any subclass of `AbstractQuantity`.

    >>> import jax.numpy as jnp
    >>> import unxt as u

    >>> x = jnp.array([1.0, 2, 3])
    >>> u.Quantity.from_(x, "m")
    Quantity['length'](Array([1., 2., 3.], dtype=float32), unit='m')

    >>> u.Quantity.from_([1.0, 2, 3], "m")
    Quantity['length'](Array([1., 2., 3.], dtype=float32), unit='m')

    >>> u.Quantity.from_((1.0, 2, 3), "m")
    Quantity['length'](Array([1., 2., 3.], dtype=float32), unit='m')

    """
    # Dispatch on both arguments.
    # Construct using the standard `__init__` method.
    return cls(jnp.asarray(value, dtype=dtype), unit)


@AbstractQuantity.from_.dispatch
def from_(
    cls: type[AbstractQuantity],
    value: ArrayLike | ArrayLikeSequence,
    /,
    *,
    unit: Any,
    dtype: Any = None,
) -> AbstractQuantity:
    """Make a `unxt.AbstractQuantity` from an array-like value and a unit kwarg.

    Examples
    --------
    For this example we'll use the `unxt.Quantity` class. The same applies
    to any subclass of `unxt.AbstractQuantity`.

    >>> import unxt as u
    >>> u.Quantity.from_([1.0, 2, 3], unit="m")
    Quantity['length'](Array([1., 2., 3.], dtype=float32), unit='m')

    """
    # Dispatch on the `value` only. Dispatch to the full constructor.
    return cls.from_(value, unit, dtype=dtype)


@AbstractQuantity.from_.dispatch
def from_(
    cls: type[AbstractQuantity], *, value: Any, unit: Any, dtype: Any = None
) -> AbstractQuantity:
    """Construct a `AbstractQuantity` from value and unit kwargs.

    Examples
    --------
    For this example we'll use the `Quantity` class. The same applies to
    any subclass of `AbstractQuantity`.

    >>> import unxt as u
    >>> u.Quantity.from_(value=[1.0, 2, 3], unit="m")
    Quantity['length'](Array([1., 2., 3.], dtype=float32), unit='m')

    """
    # Dispatched on no argument. Dispatch to the full constructor.
    return cls.from_(value, unit, dtype=dtype)


@AbstractQuantity.from_.dispatch
def from_(cls: type[AbstractQuantity], mapping: Mapping[str, Any]) -> AbstractQuantity:
    """Construct a `Quantity` from a Mapping.

    Examples
    --------
    For this example we'll use the `Quantity` class. The same applies to
    any subclass of `AbstractQuantity`.

    >>> import jax.numpy as jnp
    >>> import unxt as u

    >>> x = jnp.array([1.0, 2, 3])
    >>> q = u.Quantity.from_({"value": x, "unit": "m"})
    >>> q
    Quantity['length'](Array([1., 2., 3.], dtype=float32), unit='m')

    >>> u.Quantity.from_({"value": q, "unit": "km"})
    Quantity['length'](Array([0.001, 0.002, 0.003], dtype=float32), unit='km')

    """
    # Dispatch on both arguments.
    # Construct using the standard `__init__` method.
    return cls.from_(**mapping)


@AbstractQuantity.from_.dispatch
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
    >>> import unxt as u

    >>> q = u.Quantity(1, "m")
    >>> u.Quantity.from_(q, "cm")
    Quantity['length'](Array(100., dtype=float32, ...), unit='cm')

    """
    value = jnp.asarray(uconvert(unit, value), dtype=dtype)
    return cls(ustrip(unit, value), unit)


@AbstractQuantity.from_.dispatch
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
    >>> import unxt as u

    >>> q = u.Quantity(1, "m")
    >>> u.Quantity.from_(q, None)
    Quantity['length'](Array(1, dtype=int32, ...), unit='m')

    """
    value = jnp.asarray(value, dtype=dtype)
    unit = unit_of(value)
    return cls(ustrip(unit, value), unit)


@AbstractQuantity.from_.dispatch
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
    return cls(ustrip(unit, value), unit)


# -----------------------------------------------
# Promotion rules

add_promotion_rule(AbstractQuantity, AbstractQuantity, AbstractQuantity)


# ===============================================================
# Support for ``at``.


# `_QuantityIndexUpdateHelper` is defined up here because it is used in the
# runtime-checkable type annotation in `AbstractQuantity.at`.
# `_QuantityIndexUpdateRef` is defined after `AbstractQuantity` because it
# references `AbstractQuantity` in its runtime-checkable type annotations.
class _QuantityIndexUpdateHelper(_IndexUpdateHelper):
    def __getitem__(self, index: Any) -> "_IndexUpdateRef":
        return _QuantityIndexUpdateRef(self.array, index)  # type: ignore[no-untyped-call]

    def __repr__(self) -> str:
        """Return a string representation of the object.

        Examples
        --------
        >>> import unxt as u
        >>> q = u.Quantity([1, 2, 3, 4], "m")
        >>> q.at
        _QuantityIndexUpdateHelper(Quantity['length'](Array([1, 2, 3, 4], dtype=int32), unit='m'))

        """  # noqa: E501
        return f"_QuantityIndexUpdateHelper({self.array!r})"


class _QuantityIndexUpdateRef(_IndexUpdateRef):
    # This is a subclass of `_IndexUpdateRef` that is used to implement the `at`
    # attribute of `AbstractQuantity`. See also `_QuantityIndexUpdateHelper`.

    def __repr__(self) -> str:
        return super().__repr__().replace("_IndexUpdateRef", "_QuantityIndexUpdateRef")

    @override
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


def is_any_quantity(obj: Any, /) -> TypeGuard[AbstractQuantity]:
    """Check if an object is an instance of `unxt.quantity.AbstractQuantity`.

    Examples
    --------
    >>> import unxt as u

    >>> q = u.Quantity(1, "m")
    >>> is_any_quantity(q)
    True

    """
    return isinstance(obj, AbstractQuantity)
