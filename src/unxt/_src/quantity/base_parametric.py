# pylint: disable=import-error, no-member, unsubscriptable-object
#    b/c it doesn't understand dataclass fields

__all__ = ("AbstractParametricQuantity",)

from collections.abc import Callable
from functools import partial
from typing import Any

import equinox as eqx
import wadler_lindig as wl
from astropy.units import PhysicalType, Unit
from jaxtyping import Array, Shaped
from plum import dispatch, parametric, type_unparametrized

from dataclassish import field_items, fields

from .base import AbstractQuantity
from unxt._src.dimensions import name_of
from unxt.dims import AbstractDimension, dimension, dimension_of
from unxt.units import AbstractUnit, unit as parse_unit


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
        if got_dimensions != expected_dimensions:  # pylint: disable=unreachable
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
    def __init_type_parameter__(cls, unit: AbstractUnit, /) -> tuple[AbstractDimension]:
        """Infer the type parameter from the arguments."""
        dims = dimension_of(unit)
        if dims != "unknown":  # pylint: disable=unreachable
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
        Quantity(Array([1, 2, 3], dtype=int32), unit='m')

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

    def __pdoc__(
        self,
        *,
        include_params: bool = False,
        named_unit: bool = True,
        use_short_name: bool = False,
        **kwargs: Any,
    ) -> wl.AbstractDoc:
        """Return the Wadler-Lindig representation of this class.

        This is used for the `__repr__` and `__str__` methods or when using the
        `wadler_lindig` library.

        Parameters
        ----------
        include_params
            If `True`, the type parameter is included in the representation. If
            `False`, the type parameter is not included in the representation.
            For example, ``Quantity['length'][i32]`` versus ``Quantity[i32]``.
        named_unit
            If `True`, the unit is included in the representation as a named
            argument. If `False`, the unit is included as a positional argument.
            For example, ``Quantity(<array>, unit='m')`` versus
            ``Quantity(<array>, 'm')``.
        use_short_name
            If `True` and the class has a ``short_name`` class variable, use the
            short name instead of the full class name. For example, ``Q(...)``
            instead of ``Quantity(...)``.
        kwargs
            Additional keyword arguments ``wadler_lindig.pdoc`` method for
            formatting the value, stringified unit, and other fields.

        Examples
        --------
        >>> import unxt as u
        >>> import wadler_lindig as wl

        >>> q = u.Quantity([1, 2, 3], "m")

        The default pretty printing:

        >>> wl.pprint(q)
        Quantity(i32[3], unit='m')

        The type parameter can be included in the representation:

        >>> wl.pprint(q, include_params=True)
        Quantity['length'](i32[3], unit='m')

        The `str` method uses this as well:

        >>> print(q)
        Quantity['length']([1, 2, 3], unit='m')

        Arrays can be printed in full:

        >>> wl.pprint(q, short_arrays=False)
        Quantity(Array([1, 2, 3], dtype=int32), unit='m')

        The `repr` method uses this setting:

        >>> print(repr(q))
        Quantity(Array([1, 2, 3], dtype=int32), unit='m')

        The units can be turned from a named argument to a positional argument
        by setting `named_unit=False`:

        >>> wl.pprint(q, named_unit=False)
        Quantity(i32[3], 'm')

        The class can be printed with its short name:

        >>> wl.pprint(q, use_short_name=True)
        Q(i32[3], unit='m')

        Short name with type parameter:

        >>> wl.pprint(q, use_short_name=True, include_params=True)
        Q['length'](i32[3], unit='m')

        """
        pdoc = super().__pdoc__(
            include_params=include_params,
            named_unit=named_unit,
            use_short_name=use_short_name,
            **kwargs,
        )

        # Type Parameter
        if not include_params:
            param = wl.TextDoc("")
        else:
            param = wl.TextDoc(f"[{name_of(self._type_parameter)!r}]")

        return wl.ConcatDoc(pdoc.children[0], param, *pdoc.children[1:])

    def __repr__(self) -> str:
        return wl.pformat(
            self, include_params=False, named_unit=True, short_arrays=False, indent=4
        )

    def __str__(self) -> str:
        return wl.pformat(
            self,
            include_params=True,
            named_unit=True,
            short_arrays="compact",
            indent=4,
        )
