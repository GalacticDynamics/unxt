# pylint: disable=import-error, no-member, unsubscriptable-object
#    b/c it doesn't understand dataclass fields

__all__ = ["AbstractParametricQuantity"]

from collections.abc import Callable
from functools import partial
from typing import Any

import equinox as eqx
from astropy.units import PhysicalType, Unit
from jaxtyping import Array, Shaped
from plum import dispatch, parametric, type_nonparametric, type_unparametrized

from dataclassish import field_items, fields

from .base import AbstractQuantity
from unxt._src.units import AstropyUnits
from unxt.dims import AbstractDimension, dimension, dimension_of
from unxt.units import unit as parse_unit


@parametric
class AbstractParametricQuantity(AbstractQuantity):
    """Arrays with associated units.

    This class is parametrized by the dimensions of the units.

    """

    value: eqx.AbstractVar[Shaped[Array, "*shape"]]
    """The value of the `AbstractQuantity`."""

    unit: eqx.AbstractVar[Unit]
    """The unit associated with this value."""

    def __post_init__(self) -> None:
        """Check whether the arguments are valid."""
        self._type_parameter: AbstractDimension

    def __check_init__(self) -> None:
        """Check whether the arguments are valid."""
        expected_dimensions = self._type_parameter
        got_dimensions = dimension_of(self.unit)
        if got_dimensions != expected_dimensions:
            msg = "Physical type mismatch."  # TODO: better error message
            raise ValueError(msg)

    # ---------------------------------------------------------------
    # Plum stuff

    @classmethod
    @dispatch
    def __init_type_parameter__(
        cls, dims: AbstractDimension, /
    ) -> tuple[AbstractDimension]:
        """Check whether the type parameters are valid."""
        return (dims,)

    @classmethod
    @dispatch
    def __init_type_parameter__(cls, dims: str, /) -> tuple[AbstractDimension]:
        """Check whether the type parameters are valid."""
        dims_: AbstractDimension
        try:
            dims_ = dimension(dims)
        except ValueError:
            dims_ = dimension_of(parse_unit(dims))
        return (dims_,)

    @classmethod
    @dispatch
    def __init_type_parameter__(cls, unit: AstropyUnits, /) -> tuple[AbstractDimension]:
        """Infer the type parameter from the arguments."""
        dims = dimension_of(unit)
        if dims != "unknown":
            return (dims,)
        return (PhysicalType(unit, unit.to_string(fraction=False)),)

    @classmethod
    def __infer_type_parameter__(
        cls, value: Any, unit: Any, **kwargs: Any
    ) -> tuple[AbstractDimension]:
        """Infer the type parameter from the arguments."""
        return (dimension_of(parse_unit(unit)),)

    @classmethod
    @dispatch
    def __le_type_parameter__(
        cls, left: tuple[AbstractDimension], right: tuple[AbstractDimension]
    ) -> bool:
        """Define an order on type parameters.

        That is, check whether `left <= right` or not.
        """
        (dim_left,) = left
        (dim_right,) = right
        return dim_left == dim_right

    # ---------------------------------------------------------------
    # misc

    def __getnewargs_ex__(self) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Return args, kwargs for ``__new__``.

        In protocols 2+, this is used to determine the values (args,
        and kwargs) passed to ``__new__``. We implement ``__getnewargs_ex__``
        instead of ``__getnewargs__`` because the latter does not support
        keyword-only arguments.

        Examples
        --------
        >>> import copy as pycopy
        >>> import unxt as u

        >>> x = u.Quantity([1, 2, 3], "m")
        >>> pycopy.copy(x)
        Quantity['length'](Array([1, 2, 3], dtype=int32), unit='m')

        """
        return (), field_items(self)

    # TODO: see if pickling can be accomplished without reduce.
    # https://docs.python.org/3.12/library/pickle.html
    def __reduce__(
        self,
    ) -> tuple[Callable[..., "AbstractParametricQuantity"], tuple[Any, ...]]:
        r"""Return the reduced value: a constructor and arguments.

        The ``__reduce__`` protocol has limited support for keyword-only
        argument. The only built-in means to pass kwargs is to the
        ``__setstate__`` method or through a callable with signature ``(obj,
        state)``. Neither of these methods allow for the kwargs to be passed to
        the constructor, only after the object is partially initialized. This
        does not work well for JAX, Equinox, or particularly run-time
        typechecking. To get around this, instead of the class type as the
        customary first element of reduced value we will instead return a
        `functools.partial`-wrapping of the class type with the kwargs bundled
        into the partial object. This is allowed by the ``__reduce__`` protocol,
        which says the first returned item must be any "callable object that
        will be called to create the initial version of the object." In
        conjunction with the standard args, the `functools.partial`-wrapped
        class type will properly construct any parametric subclass.

        Returns
        -------
        functools.partial[type]
            The `plum.type_unparametrized` form of this class object, which is
            the parametric class without the specialization to the specific
            parameter. The specialized subtype cannot be pickled since it is
            dynamically produced. The type is wrapped into a `functools.partial`
            with the keyword-only arguments to the constructor.
        tuple[Any, ...]
            The arguments to this class. Note: the keyword-only arguments are
            bundled with the type.

        Examples
        --------
        >>> import pickle
        >>> import unxt as u

        >>> x = u.Quantity([1, 2, 3], "m")
        >>> pickle.dumps(x)
        b'...'

        """
        args, kwargs = [], {}
        for f in fields(self):
            v = getattr(self, f.name)
            if f.kw_only:
                kwargs[f.name] = v
            else:
                args.append(v)

        return partial(type_unparametrized(self), **kwargs), tuple(args)

    def __repr__(self) -> str:
        # --- class name ---
        base_cls_name = type_nonparametric(self).__name__
        if self._type_parameter == "unknown":
            ptid = self._type_parameter._unit._physical_type_id  # noqa: SLF001
            dim = " ".join(
                f"{unit}{power}" if power != 1 else unit for unit, power in ptid
            )
        else:
            dim = self._type_parameter._name_string_as_ordered_set().split("'")[1]  # noqa: SLF001
        cls_name = f"{base_cls_name}[{dim!r}]"

        # --- args, kwargs ---

        fs = dict(field_items(self))
        del fs["value"]
        del fs["unit"]

        base_fields = f"{self.value!r}, unit={self.unit.to_string()!r}"
        extra_fields = ", ".join(f"{k}={v}" for k, v in fs.items())
        all_fields = base_fields + (", " + extra_fields if fs else "")

        # ------
        return f"{cls_name}({all_fields})"
