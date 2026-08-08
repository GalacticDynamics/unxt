"""Orthogonal mixin classes for quantity classes."""

__all__: tuple[str, ...] = ()

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, cast

import equinox as eqx
import numpy as np
from astropy.units import (
    CompositeUnit,
    UnitConversionError,
    dimensionless_unscaled as one,
)
from jax.typing import ArrayLike
from jaxtyping import Array

from dataclassish import replace

import unxt_api as uapi
from .register_ufuncs import apply_ufunc
from unxt._src import fmt
from unxt.units import AbstractUnit, unit as parse_unit

if TYPE_CHECKING:
    import unxt.quantity


class AstropyQuantityCompatMixin:
    """Mixin for compatibility with `astropy.units.Quantity`."""

    value: eqx.AbstractVar[ArrayLike]
    unit: eqx.AbstractVar[AbstractUnit]
    uconvert: Callable[[Any], "unxt.quantity.AbstractQuantity"]
    ustrip: Callable[[Any], ArrayLike]

    def to(self, u: Any, /) -> "unxt.quantity.AbstractQuantity":
        """Convert the quantity to the given units.

        See `unxt.quantity.AbstractQuantity.uconvert`.

        Examples
        --------
        >>> from unxt import Quantity

        >>> q = Quantity(1, "m")
        >>> q.to("cm")
        Quantity(Array(100., dtype=float32, ...), unit='cm')

        """
        return uapi.uconvert(u, self)  # redirect to the standard method

    def to_value(self, u: Any, /) -> ArrayLike:
        """Return the value in the given units.

        See `unxt.AbstractQuantity.ustrip`.

        Examples
        --------
        >>> from unxt import Quantity

        >>> q = Quantity(1, "m")
        >>> q.to_value("cm")
        Array(100., dtype=float32, weak_type=True)

        """
        return uapi.ustrip(u, self)  # redirect to the standard method

    def decompose(
        self, bases: Sequence[AbstractUnit | str], /
    ) -> "unxt.quantity.AbstractQuantity":
        """Decompose the quantity into the given bases.

        Examples
        --------
        >>> from unxt import Quantity

        >>> q = Quantity(1, "m")
        >>> q.decompose(["cm", "s"])
        Quantity(Array(100., dtype=float32, ...), unit='cm')

        """
        bases_ = [parse_unit(b) for b in bases]
        du = self.unit.decompose(bases_)  # decomposed units
        base_units = CompositeUnit(scale=1, bases=du.bases, powers=du.powers)
        return replace(self, value=self.value * du.scale, unit=base_units)


#####################################################################


SUPPORTED_IPYTHON_REPR_FORMATS: dict[str, str] = {
    "text/plain": "__repr__",
    "text/html": "_repr_html_",
    "text/latex": "_repr_latex_",
}


class IPythonReprMixin:
    """Mixin class for IPython representation of a quantity."""

    value: Array
    unit: AbstractUnit

    def _repr_mimebundle_(
        self,
        *,
        include: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
    ) -> dict[str, str]:
        r"""Return a MIME bundle representation of the quantity.

        :param include: The set of keys to include in the MIME bundle. If not
            provided, all supported formats are included.
        :param exclude: The set of keys to exclude in the MIME bundle. If not
            provided, all supported formats are included. 'include' has
            precedence over 'exclude'.

        Examples
        --------
        >>> from unxt import Quantity

        >>> q = Quantity([1.0, 2, 3, 4], "m")
        >>> q._repr_mimebundle_()
        {'text/plain':
         "Quantity(Array([1., 2., 3., 4.], dtype=float32), unit='m')",
         'text/html': '<span>[1., 2., 3., 4.]</span> * <span>m</span>',
         'text/latex': '$[1.,~2.,~3.,~4.] \\; \\mathrm{m}$'}

        >>> q._repr_mimebundle_(include=["text/plain"])
        {'text/plain':
         "Quantity(Array([1., 2., 3., 4.], dtype=float32), unit='m')"}

        >>> q._repr_mimebundle_(exclude=["text/html", "text/latex"])
        {'text/plain':
         "Quantity(Array([1., 2., 3., 4.], dtype=float32), unit='m')"}

        """
        # Determine the set of keys to include in the MIME bundle
        keys: Sequence[str]
        if include is None and exclude is None:
            keys = tuple(SUPPORTED_IPYTHON_REPR_FORMATS)
        elif include is not None:
            keys = [key for key in include if key in SUPPORTED_IPYTHON_REPR_FORMATS]
        else:
            keys = [
                k
                for k in SUPPORTED_IPYTHON_REPR_FORMATS
                if k not in cast("str", exclude)
            ]

        # Create the MIME bundle
        return {
            key: getattr(self, SUPPORTED_IPYTHON_REPR_FORMATS[key])() for key in keys
        }

    def _repr_markup_(self, markup: str, /) -> str:
        """Render through the `unxt._fmt` engine in the given markup."""
        return fmt.parts_to_markup(fmt.pparts(self, markup=markup), markup=markup)

    def _repr_html_(self) -> str:
        """Return an HTML representation of the quantity.

        Examples
        --------
        >>> from unxt import Quantity

        >>> q = Quantity([1.0, 2, 3, 4], "m")
        >>> q._repr_html_()
        '<span>[1., 2., 3., 4.]</span> * <span>m</span>'

        """
        return self._repr_markup_("html")

    def _repr_latex_(self) -> str:
        r"""Return a LaTeX representation of the quantity.

        Examples
        --------
        >>> from unxt import Quantity

        >>> q = Quantity([1.0, 2, 3, 4], "m")
        >>> q._repr_latex_()
        '$[1.,~2.,~3.,~4.] \\; \\mathrm{m}$'

        """
        return self._repr_markup_("latex")

    # TODO: implement:
    # - _repr_markdown_
    # - _repr_json_


#####################################################################


class NumPyCompatMixin:
    """Mixin for compatibility with numpy arrays."""

    unit: AbstractUnit

    __array_namespace__: Callable[[], Any]

    def __array__(self, *args: object, **kw: object) -> np.ndarray:
        """Return a bare array -- but only where that loses nothing.

        A *dimensionful* quantity has no unambiguous array form. ``np.asarray``
        on ``Quantity(1.5, "km")`` would give ``1.5`` while the same length in
        metres gives ``1500``, so the consumer reads a number whose meaning
        depends on a unit it never saw.

        This matters because ``__array__`` is reached *implicitly*: ``np.asarray``
        has always used it, and ``jax.numpy.asarray`` does too as of jax 0.10 --
        where it previously raised. Returning the value here therefore turns a
        loud failure into a wrong number, which is the one outcome a unit library
        must not produce. Use `unxt.ustrip` to say which unit you meant.

        Dimensionless quantities convert, because there is nothing to lose.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import numpy as np

        >>> np.array(Quantity(1.01, ""))
        array(1.01, dtype=float32)

        A dimensionful one refuses, and names the way to be explicit:

        >>> try:
        ...     np.array(Quantity(1.01, "m"))
        ... except Exception as e:
        ...     print(type(e).__name__)
        UnitConversionError

        """
        if self.unit != one:
            msg = (
                f"cannot convert Quantity in {self.unit!r} to a bare array: the "
                f"result would be a number whose meaning depends on a unit the "
                f"caller never sees. Use `unxt.ustrip(<unit>, q)` to choose one, "
                f"or `unxt.uconvert('', q)` if it really is dimensionless."
            )
            raise UnitConversionError(msg)
        return np.asarray(uapi.ustrip(self.unit, self), *args, **kw)

    # TODO: why doesn't `__array_namespace__` supersede this?
    def __array_function__(
        self,
        func: Callable[..., Any],
        types: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """Dispatch to the corresponding jax.numpy function.

        Examples
        --------
        >>> import numpy as np
        >>> import unxt as u

        >>> q = u.Q([1.0, 2, 3, 4], "m")
        >>> np.sum(q)
        Quantity(Array(10., dtype=float32), unit='m')

        >>> np.stack([q, q])
        Quantity(Array([[1., 2., 3., 4.],
                        [1., 2., 3., 4.]], dtype=float32), unit='m')

        """
        xp = self.__array_namespace__()
        xfunc = getattr(xp, func.__name__)
        return xfunc(*args, **kwargs)

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs: Any,
        **kwargs: Any,
    ) -> Any:
        """Dispatch a NumPy ufunc to a unit-aware handler.

        Built-in ufuncs delegate to `quaxed.numpy`, which propagates units via
        quax. Custom ufuncs may be registered with
        `unxt.quantity.register_ufunc`; unhandled ufuncs or methods return
        ``NotImplemented`` so NumPy raises a loud ``TypeError`` rather than
        silently dropping units.

        Examples
        --------
        >>> import numpy as np
        >>> import unxt as u

        >>> np.multiply(u.Q(5.0, "m"), u.Q(3.0, "m"))
        Quantity(Array(15., dtype=float32...), unit='m2')

        >>> np.sqrt(u.Q(4.0, "m2"))
        Quantity(Array(2., dtype=float32...), unit='m')

        """
        return apply_ufunc(ufunc, method, inputs, kwargs)
