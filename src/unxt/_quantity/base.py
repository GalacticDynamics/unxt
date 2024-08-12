# pylint: disable=import-error, no-member, unsubscriptable-object
#    b/c it doesn't understand dataclass fields

__all__ = ["AbstractQuantity", "can_convert_unit"]

from collections.abc import Callable, Mapping, Sequence
from dataclasses import fields, replace
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar, NoReturn, TypeAlias, TypeGuard, TypeVar
from typing_extensions import Self

import equinox as eqx
import jax
import jax.core
from astropy.units import CompositeUnit, UnitConversionError
from jax._src.numpy.array_methods import _IndexUpdateHelper, _IndexUpdateRef
from jaxtyping import Array, ArrayLike, Shaped
from numpy import bool_ as np_bool, dtype as DType, number as np_number  # noqa: N812
from plum import add_promotion_rule
from quax import ArrayValue

import quaxed.array_api as xp
import quaxed.numpy as jnp
import quaxed.operator as qoperator
from quaxed.array_api._dispatch import dispatcher

from unxt._units import Unit

if TYPE_CHECKING:
    from array_api import ArrayAPINamespace


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


def _is_tracing(x: Any) -> TypeGuard[jax.core.Tracer]:
    return isinstance(x, jax.core.Tracer)


# `_QuantityIndexUpdateHelper` is defined up here because it is used in the
# runtime-checkable type annotation in `AbstractQuantity.at`.
# `_QuantityIndexUpdateRef` is defined after `AbstractQuantity` because it
# references `AbstractQuantity` in its runtime-checkable type annotations.
class _QuantityIndexUpdateHelper(_IndexUpdateHelper):  # type: ignore[misc]
    def __getitem__(self, index: Any) -> "_IndexUpdateRef":
        return _QuantityIndexUpdateRef(self.array, index)

    def __repr__(self) -> str:
        return f"_QuantityIndexUpdateHelper({self.array!r})"


##############################################################################


class AbstractQuantity(ArrayValue):  # type: ignore[misc]
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

    >>> import astropy.units as u
    >>> Quantity(1, u.m)
    Quantity['length'](Array(1, dtype=int32, ...), unit='m')

    """

    value: Shaped[Array, "*shape"] = eqx.field(converter=jax.numpy.asarray)
    """The value of the Quantity."""

    unit: Unit = eqx.field(static=True, converter=Unit)
    """The unit associated with this value."""

    # ---------------------------------------------------------------
    # Plum stuff

    __faithful__: ClassVar[bool] = True
    """Tells `plum` that this type can be cached more efficiently."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.value!r}, unit={self.unit.to_string()!r})"

    # ---------------------------------------------------------------
    # Constructors

    @classmethod
    @dispatcher
    def constructor(
        cls: "type[AbstractQuantity]",
        value: ArrayLike | ArrayLikeSequence,
        unit: Any,
        /,
        *,
        dtype: Any = None,
    ) -> "AbstractQuantity":
        """Construct a `Quantity` from an array-like value and a unit.

        Parameters
        ----------
        value : ArrayLike | list[...] | tuple[...]
            The array-like value.
        unit : Any
            The unit of the value.

        dtype : Any, optional
            The data type of the array.

        Returns
        -------
        AbstractQuantity

        Examples
        --------
        For this example we'll use the `Quantity` class. The same applies to
        any subclass of `AbstractQuantity`.

        >>> import jax.numpy as jnp
        >>> from unxt import Quantity

        >>> x = jnp.array([1.0, 2, 3])
        >>> Quantity.constructor(x, "m")
        Quantity['length'](Array([1., 2., 3.], dtype=float32), unit='m')

        >>> Quantity.constructor([1.0, 2, 3], "m")
        Quantity['length'](Array([1., 2., 3.], dtype=float32), unit='m')

        >>> Quantity.constructor((1.0, 2, 3), "m")
        Quantity['length'](Array([1., 2., 3.], dtype=float32), unit='m')

        """
        # Dispatch on both arguments.
        # Construct using the standard `__init__` method.
        return cls(xp.asarray(value, dtype=dtype), unit)

    @classmethod  # type: ignore[no-redef]
    @dispatcher
    def constructor(
        cls: "type[AbstractQuantity]",
        value: ArrayLike | ArrayLikeSequence,
        /,
        *,
        unit: Any,
        dtype: Any = None,
    ) -> "AbstractQuantity":
        """Construct a `Quantity` from an array-like value and a unit kwarg.

        This is a convenience method for constructing a `Quantity` when the
        unit is given as a keyword argument. It is equivalent to calling
        ``Quantity.constructor(value, unit)`` as positional arguments.
        """
        # Dispatch on the `value` only. Dispatch to the full constructor.
        return cls.constructor(value, unit, dtype=dtype)

    @classmethod  # type: ignore[no-redef]
    @dispatcher
    def constructor(
        cls: "type[AbstractQuantity]", *, value: Any, unit: Any, dtype: Any = None
    ) -> "AbstractQuantity":
        """Construct a `Quantity` from value and unit kwargs.

        This is a convenience method for constructing a `Quantity` when both the
        value and the unit are given as a keyword arguments. It is equivalent to
        calling ``Quantity.constructor(value, unit)`` as positional arguments.
        """
        # Dispatched on no argument. Dispatch to the full constructor.
        return cls.constructor(value, unit, dtype=dtype)

    @classmethod  # type: ignore[no-redef]
    @dispatcher
    def constructor(
        cls: "type[AbstractQuantity]", mapping: Mapping[str, Any]
    ) -> "AbstractQuantity":
        """Construct a `Quantity` from a Mapping.

        Parameters
        ----------
        mapping : Mapping[str, Any]
            Mapping of the fields of the `Quantity`, e.g. 'value' and 'unit'.

        Returns
        -------
        AbstractQuantity

        Examples
        --------
        For this example we'll use the `Quantity` class. The same applies to
        any subclass of `AbstractQuantity`.

        >>> import jax.numpy as jnp
        >>> from unxt import Quantity

        >>> x = jnp.array([1.0, 2, 3])
        >>> q = Quantity.constructor({"value": x, "unit": "m"})
        >>> q
        Quantity['length'](Array([1., 2., 3.], dtype=float32), unit='m')

        >>> Quantity.constructor({"value": q, "unit": "km"})
        Quantity['length'](Array([0.001, 0.002, 0.003], dtype=float32), unit='km')

        """
        # Dispatch on both arguments.
        # Construct using the standard `__init__` method.
        return cls.constructor(**mapping)

    # See below for additional constructors.

    # ===============================================================
    # Quax

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the array."""
        return self.value.shape

    def materialise(self) -> NoReturn:
        msg = "Refusing to materialise `Quantity`."
        raise RuntimeError(msg)

    def aval(self) -> jax.core.ShapedArray:
        return jax.core.get_aval(self.value)

    def enable_materialise(self, _: bool = True) -> Self:  # noqa: FBT001, FBT002
        return replace(self, value=self.value, unit=self.unit)

    # ===============================================================
    # Quantity API

    def to_units(self, units: Unit) -> "AbstractQuantity":
        """Convert the quantity to the given units.

        Parameters
        ----------
        units : Unit
            The units to convert to.

        Returns
        -------
        AbstractQuantity

        Examples
        --------
        >>> from unxt import Quantity

        >>> q = Quantity(1, "m")
        >>> q.to_units("cm")
        Quantity['length'](Array(100., dtype=float32, ...), unit='cm')

        """
        return replace(self, value=self.to_units_value(units), unit=units)

    def to_units_value(self, units: Unit) -> Array:
        """Return the value in the given units.

        Parameters
        ----------
        units : Unit
            The units to convert to.

        Returns
        -------
        ArrayLike

        Examples
        --------
        >>> from unxt import Quantity

        >>> q = Quantity(1, "m")
        >>> q.to_units_value("cm")
        Array(100., dtype=float32, weak_type=True)

        """
        # Hot-path: if no unit conversion is necessary
        if self.unit == units:
            return self.value

        # Hot-path: if in tracing mode
        # TODO: jaxpr units so we can understand them at trace time.
        if _is_tracing(self.value) and not can_convert_unit(self.unit, units):
            return self.value

        return self.value * self.unit.to(units)

    def decompose(self, bases: Sequence[Unit]) -> "AbstractQuantity":
        """Decompose the quantity into the given bases."""
        du = self.unit.decompose(bases)  # decomposed units
        base_units = CompositeUnit(scale=1, bases=du.bases, powers=du.powers)
        return replace(self, value=self.value * du.scale, unit=base_units)

    # ===============================================================
    # Astropy Quantity API

    def to(self, units: Unit) -> "AbstractQuantity":
        """Convert the quantity to the given units.

        See :meth:`AbstractQuantity.to_units`.
        """
        return self.to_units(units)

    def to_value(self, units: Unit) -> ArrayLike:
        """Return the value in the given units.

        See :meth:`AbstractQuantity.to_units_value`.
        """
        return self.to_units_value(units)

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

    @dispatcher  # type: ignore[misc]
    def __mod__(self: "AbstractQuantity", other: Any) -> "AbstractQuantity":
        """Take the mod.

        Examples
        --------
        >>> from unxt import Quantity

        >>> q = Quantity(480, "deg")
        >>> q % Quantity(360, "deg")
        Quantity['angle'](Array(120, dtype=int32, ...), unit='deg')

        """
        if not can_convert_unit(other.unit, self.unit):
            raise UnitConversionError

        # TODO: figure out how to defer to quaxed (e.g. quaxed.operator.mod)
        return replace(self, value=self.value % other.to_units_value(self.unit))

    # ===============================================================
    # JAX API

    @partial(property, doc=jax.Array.at.__doc__)
    def at(self) -> _QuantityIndexUpdateHelper:
        return _QuantityIndexUpdateHelper(self)

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
        unhashable type: ...

        """
        return hash(tuple(getattr(self, f.name) for f in fields(self)))


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

    Examples
    --------
    >>> from unxt import Quantity

    >>> q = Quantity(1, "m")
    >>> Quantity.constructor(q, "cm")
    Quantity['length'](Array(100., dtype=float32, ...), unit='cm')

    """
    value = xp.asarray(value.to_units(unit), dtype=dtype)
    return cls(value.value, unit)


@AbstractQuantity.constructor._f.register  # type: ignore[no-redef]  # noqa: SLF001
def constructor(
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
    >>> Quantity.constructor(q, None)
    Quantity['length'](Array(1, dtype=int32, ...), unit='m')

    """
    value = xp.asarray(value, dtype=dtype)
    return cls(value.value, value.unit)


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
    value = xp.asarray(value.to_units(unit), dtype=dtype)
    return cls(value.value, unit)


# -----------------------------------------------
# Promotion rules

add_promotion_rule(AbstractQuantity, AbstractQuantity, AbstractQuantity)


# ===============================================================


def can_convert_unit(from_unit: AbstractQuantity | Unit, to_unit: Unit, /) -> bool:
    """Check if a unit can be converted to another unit.

    Parameters
    ----------
    from_unit : :clas:`unxt.AbstractQuantity` | Unit
        The unit to convert from.
    to_unit : Unit
        The unit to convert to.

    Returns
    -------
    bool
        Whether the conversion is possible.

    """
    try:
        from_unit.to(to_unit)
    except UnitConversionError:
        return False
    return True


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
            else fill_value.to_units_value(self.array.unit),
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
            values.to_units_value(self.array.unit),
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
            values.to_units_value(self.array.unit),
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
            values.to_units_value(self.array.unit),
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
            values.to_units_value(self.array.unit),
            indices_are_sorted=indices_are_sorted,
            unique_indices=unique_indices,
            mode=mode,
        )
        return replace(self.array, value=value)
