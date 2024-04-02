# pylint: disable=import-error

"""unxt: Quantities in JAX.

THIS MODULE IS NOT GUARANTEED TO HAVE A STABLE API!

Copyright (c) 2023 Galactic Dynamics. All rights reserved.
"""

__all__ = ["grad", "hessian", "jacfwd"]

from collections.abc import Callable
from functools import partial
from typing import ParamSpec, TypeVar

import equinox as eqx
import jax
from astropy.units import UnitBase as Unit
from jaxtyping import ArrayLike

from ._quantity.core import Quantity
from ._quantity.utils import type_unparametrized

P = ParamSpec("P")
R = TypeVar("R", bound=Quantity)


def grad(
    fun: Callable[P, R], argnums: int = 0, *, units: tuple[Unit, ...]
) -> Callable[P, R]:
    @partial(jax.grad, argnums=argnums)
    def gradfun_mag(*args: P.args) -> ArrayLike:
        args_ = (
            (a if unit is None else Quantity(a, unit))
            for a, unit in zip(args, units, strict=True)
        )
        return fun(*args_).value

    def gradfun(*args: P.args, **kw: P.kwargs) -> R:
        # Get the value of the args. They are turned back into Quantity
        # inside the function we are taking the grad of.
        args_ = tuple(
            (a if unit is None else a.to_units_value(unit))  # type: ignore[attr-defined]
            for a, unit in zip(args, units, strict=True)
        )
        # Call the grad, returning a Quantity
        value = fun(*args)
        grad_value = gradfun_mag(*args_)
        # Adjust the Quantity by the units of the derivative
        # TODO: get Quantity[unit] / unit2 -> Quantity[unit/unit2] working
        return type_unparametrized(value)(grad_value, value.unit / units[argnums])

    return gradfun


def jacfwd(
    fun: Callable[P, R], argnums: int = 0, *, units: tuple[Unit, ...]
) -> Callable[P, R]:
    """Jacobian of ``fun`` evaluated column-by-column using forward-mode AD.

    In general, if you can use ``quaxed.jacfwd``, that's the better option!  The
    difference from ``quaxed.jacfwd`` is how this function supports units.
    ``quaxed.jacfwd`` does `quax.quaxify(jax.jacfwd)`, which will 'strip' the
    units when passing through. But sometimes that doesn't work and we need the
    units

    """
    argnums = eqx.error_if(
        argnums,
        not isinstance(argnums, int),
        "only int argnums are currently supported",
    )

    @partial(jax.jacfwd, argnums=argnums)
    def jacfun_mag(*args: P.args) -> R:
        args_ = (
            (a if unit is None else Quantity(a, unit))
            for a, unit in zip(args, units, strict=True)
        )
        return fun(*args_)

    def jacfun(*args: P.args, **kw: P.kwargs) -> R:
        # Get the value of the args. They are turned back into Quantity
        # inside the function we are taking the Jacobian of.
        args_ = tuple(
            (a if unit is None else a.to_units_value(unit))  # type: ignore[attr-defined]
            for a, unit in zip(args, units, strict=True)
        )
        # Call the Jacobian, returning a Quantity
        value = jacfun_mag(*args_)
        # Adjust the Quantity by the units of the derivative
        # TODO: check the unit correction
        # TODO: get Quantity[unit] / unit2 -> Quantity[unit/unit2] working
        return type_unparametrized(value)(value.value, value.unit / units[argnums])

    return jacfun


def hessian(fun: Callable[P, R], *, units: tuple[Unit, ...]) -> Callable[P, R]:
    """Hessian.

    In general, if you can use ``quaxed.jacfwd``, that's the better option!  The
    difference from ``quaxed.jacfwd`` is how this function supports units.
    ``quaxed.jacfwd`` does `quax.quaxify(jax.jacfwd)`, which will 'strip' the
    units when passing through. But sometimes that doesn't work and we need the
    units

    """

    @partial(jax.hessian)
    def hessfun_mag(*args: P.args) -> R:
        args_ = (
            (a if unit is None else Quantity(a, unit))
            for a, unit in zip(args, units, strict=True)
        )
        return fun(*args_)

    def hessfun(*args: P.args, **kw: P.kwargs) -> R:
        # Get the value of the args. They are turned back into Quantity
        # inside the function we are taking the hessian of.
        args_ = tuple(
            (a if unit is None else a.to_units_value(unit))  # type: ignore[attr-defined]
            for a, unit in zip(args, units, strict=True)
        )
        # Call the hessian, returning a Quantity
        value = hessfun_mag(*args_)
        # Adjust the Quantity by the units of the derivative
        # TODO: check the unit correction
        # TODO: get Quantity[unit] / unit2 -> Quantity[unit/unit2] working
        return type_unparametrized(value)(value.value, value.unit / units[0] ** 2)

    return hessfun
