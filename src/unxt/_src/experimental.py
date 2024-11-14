r"""Experimental features.

.. warning::

    These features may be removed or changed in the future without notice.

On some occasions JAX's automatic differentiation functions do not work well
with quantities. This is checked by enabling runtime type-checking (see the
docs), which will raise an error if a quantity's units do not match the expected
input / output units of a function. In these cases, you can use the functions in
this module to provide the units to the automatic differentiation functions.
Instead of directly propagating the units through the automatic differentiation
functions, the units are stripped and re-applied, while also being provided
within the function being AD'd.

To import this experimental module

>>> from unxt import experimental

"""
# pylint: disable=import-error

__all__ = ["grad", "hessian", "jacfwd"]

from collections.abc import Callable
from functools import partial
from typing import Any, ParamSpec, TypeVar

import equinox as eqx
import jax
from jaxtyping import ArrayLike
from plum.parametric import type_unparametrized

from .quantity.core import Quantity
from .typing_ext import Unit
from unxt._src.quantity.api import ustrip
from unxt._src.units.core import unit

P = ParamSpec("P")
R = TypeVar("R", bound=Quantity)


def unit_or_none(obj: Any) -> Unit | None:
    return obj if obj is None else unit(obj)


def grad(
    fun: Callable[P, R], argnums: int = 0, *, units: tuple[Unit | str, ...]
) -> Callable[P, R]:
    """Gradient of a function with units.

    In general, if you can use `quax.quaxify(jax.grad(func))` (or the syntactic
    sugar `quax.grad(func)`), that's the better option! The difference from
    those functions is how this units are supported. `quaxify` will directly
    propagate the units through the automatic differentiation functions. But
    sometimes that doesn't work and we need to strip the units and re-apply
    them. This function does that, using the ``units`` kwarg.

    See Also
    --------
    jax.grad : The original JAX gradient function.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt
    >>> from unxt import Quantity

    >>> def square_volume(x: Quantity["length"]) -> Quantity["volume"]:
    ...     return x**3

    >>> grad_square_volume = unxt.experimental.grad(square_volume, units=("m",))
    >>> grad_square_volume(Quantity(2.0, "m"))
    Quantity['area'](Array(12., dtype=float32, weak_type=True), unit='m2')

    """
    theunits: tuple[Unit | None, ...] = tuple(map(unit_or_none, units))

    # Gradient of function, stripping and adding units
    @partial(jax.grad, argnums=argnums)
    def gradfun_mag(*args: P.args) -> ArrayLike:
        args_ = (
            (a if unit is None else Quantity(a, unit))
            for a, unit in zip(args, theunits, strict=True)
        )
        return fun(*args_).value  # type: ignore[call-arg]

    def gradfun(*args: P.args, **kw: P.kwargs) -> R:
        # Get the value of the args. They are turned back into Quantity
        # inside the function we are taking the grad of.
        args_ = tuple(
            (a if unit is None else ustrip(unit, a))
            for a, unit in zip(args, theunits, strict=True)
        )
        # Call the grad, returning a Quantity
        value = fun(*args)  # type: ignore[call-arg]
        grad_value = gradfun_mag(*args_)
        # Adjust the Quantity by the units of the derivative
        # TODO: get Quantity[unit] / unit2 -> Quantity[unit/unit2] working
        return type_unparametrized(value)(grad_value, value.unit / theunits[argnums])

    return gradfun


def jacfwd(
    fun: Callable[P, R], argnums: int = 0, *, units: tuple[Unit | str, ...]
) -> Callable[P, R]:
    """Jacobian of ``fun`` evaluated column-by-column using forward-mode AD.

    In general, if you can use `quax.quaxify(jax.jacfwd(func))` (or the
    syntactic sugar `quax.jacfwd(func)`), that's the better option! The
    difference from those functions is how this units are supported. `quaxify`
    will directly propagate the units through the automatic differentiation
    functions. But sometimes that doesn't work and we need to strip the units
    and re-apply them. This function does that, using the ``units`` kwarg.

    See Also
    --------
    jax.jacfwd : The original JAX jacfwd function.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt
    >>> from unxt import Quantity

    >>> def square_volume(x: Quantity["length"]) -> Quantity["volume"]:
    ...     return x**3

    >>> jacfwd_square_volume = unxt.experimental.jacfwd(square_volume, units=("m",))
    >>> jacfwd_square_volume(Quantity(2.0, "m"))
    Quantity['area'](Array(12., dtype=float32, weak_type=True), unit='m2')

    """
    argnums = eqx.error_if(
        argnums,
        not isinstance(argnums, int),
        "only int argnums are currently supported",
    )

    theunits: tuple[Unit | None, ...] = tuple(map(unit_or_none, units))

    @partial(jax.jacfwd, argnums=argnums)
    def jacfun_mag(*args: P.args) -> R:
        args_ = (
            (a if unit is None else Quantity(a, unit))
            for a, unit in zip(args, theunits, strict=True)
        )
        return fun(*args_)  # type: ignore[call-arg]

    def jacfun(*args: P.args, **kw: P.kwargs) -> R:
        # Get the value of the args. They are turned back into Quantity
        # inside the function we are taking the Jacobian of.
        args_ = tuple(
            (a if unit is None else ustrip(unit, a))
            for a, unit in zip(args, theunits, strict=True)
        )
        # Call the Jacobian, returning a Quantity
        value = jacfun_mag(*args_)
        # Adjust the Quantity by the units of the derivative
        # TODO: check the unit correction
        # TODO: get Quantity[unit] / unit2 -> Quantity[unit/unit2] working
        return type_unparametrized(value)(value.value, value.unit / theunits[argnums])

    return jacfun


def hessian(
    fun: Callable[P, R], argnums: int = 0, *, units: tuple[Unit | str, ...]
) -> Callable[P, R]:
    """Hessian.

    In general, if you can use `quax.quaxify(jax.hessian(func))` (or the
    syntactic sugar `quax.hessian(func)`), that's the better option! The
    difference from those functions is how this units are supported. `quaxify`
    will directly propagate the units through the automatic differentiation
    functions. But sometimes that doesn't work and we need to strip the units
    and re-apply them. This function does that, using the ``units`` kwarg.

    See Also
    --------
    jax.hessian : The original JAX hessian function.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt
    >>> from unxt import Quantity

    >>> def square_volume(x: Quantity["length"]) -> Quantity["volume"]:
    ...     return x**3

    >>> hessian_square_volume = unxt.experimental.hessian(square_volume, units=("m",))
    >>> hessian_square_volume(Quantity(2.0, "m"))
    Quantity['length'](Array(12., dtype=float32, weak_type=True), unit='m')

    """
    theunits: tuple[Unit, ...] = tuple(map(unit_or_none, units))

    @partial(jax.hessian)
    def hessfun_mag(*args: P.args) -> R:
        args_ = (
            (a if unit is None else Quantity(a, unit))
            for a, unit in zip(args, theunits, strict=True)
        )
        return fun(*args_)  # type: ignore[call-arg]

    def hessfun(*args: P.args, **kw: P.kwargs) -> R:
        # Get the value of the args. They are turned back into Quantity
        # inside the function we are taking the hessian of.
        args_ = tuple(
            (a if unit is None else ustrip(unit, a))
            for a, unit in zip(args, units, strict=True)
        )
        # Call the hessian, returning a Quantity
        value = hessfun_mag(*args_)
        # Adjust the Quantity by the units of the derivative
        # TODO: check the unit correction
        # TODO: get Quantity[unit] / unit2 -> Quantity[unit/unit2] working
        return type_unparametrized(value)(
            value.value, value.unit / theunits[argnums] ** 2
        )

    return hessfun
