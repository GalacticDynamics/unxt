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

__all__ = ("grad", "hessian", "jacfwd", "where")

import functools as ft
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import equinox as eqx
import jax
from jaxtyping import ArrayLike
from plum import type_unparametrized

from .quantity import AbstractQuantity, Quantity
from .units import AbstractUnit
from unxt_api import unit, unit_of, ustrip


def unit_or_none(obj: Any) -> AbstractUnit | None:
    return obj if obj is None else unit(obj)


def where[R: AbstractQuantity](condition: ArrayLike, x: R, y: AbstractQuantity, /) -> R:
    """Unit-checked ``where``: both branches must be quantities (experimental).

    A strict alternative to :func:`jax.numpy.where`. Both ``x`` and ``y`` must be
    quantities (a dimensionless ``Quantity`` is fine). ``y`` is converted to
    ``x``'s unit -- raising if they are not convertible -- and the result is
    returned in ``x``'s unit and concrete type (``x`` typed as ``R`` so a checker
    sees, e.g., ``Angle`` in -> ``Angle`` out).

    Unlike ``jnp.where``, this will **not** silently reinterpret a raw array as
    being in the quantity's unit (see the "``jnp.where`` adopts the quantity's
    unit for a raw-array branch" sharp bit): a raw-array branch is rejected, so
    wrap it as ``unxt.Quantity(arr, unit)`` first. ``jnp.where`` cannot itself be
    made strict, because JAX lowers masking ops (``triu``/``tril``/``trace``,
    ``where(mask, q, 0)``) to the same primitive and relies on the raw zero-fill
    adopting the unit.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> from unxt import experimental

    >>> cond = jnp.asarray([True, False])

    ``y`` is converted to ``x``'s unit; the result is in ``x``'s unit:

    >>> experimental.where(cond, u.Q([1.0, 2.0], "m"), u.Q([0.003, 0.004], "km"))
    Quantity(Array([1., 4.], dtype=float32), unit='m')

    Incompatible units raise:

    >>> try:
    ...     experimental.where(cond, u.Q([1.0, 2.0], "m"), u.Q([1.0, 2.0], "s"))
    ... except Exception as e:
    ...     print(type(e).__name__)
    UnitConversionError

    A raw-array branch is rejected rather than silently adopting the unit:

    >>> try:
    ...     experimental.where(cond, u.Q([1.0, 2.0], "m"), jnp.asarray([3.0, 4.0]))
    ... except TypeError as e:
    ...     print(e)
    unxt.experimental.where requires both branches to be Quantities; ...

    """
    if not isinstance(x, AbstractQuantity) or not isinstance(y, AbstractQuantity):
        msg = (
            "unxt.experimental.where requires both branches to be Quantities; "
            "wrap a raw array as unxt.Quantity(arr, unit)."
        )
        raise TypeError(msg)
    # Strip ``x`` in its own unit (no conversion); convert only ``y`` to
    # ``x``'s unit (raises if not convertible). Converting ``x`` to its own unit
    # would be needless work and can promote integer dtypes. Result in x's unit.
    xv = ustrip(x)
    yv = ustrip(x.unit, y)
    return replace(x, value=jax.numpy.where(condition, xv, yv))


def grad[*Args, R: AbstractQuantity](
    fun: Callable[[*Args], R],
    argnums: int = 0,
    *,
    units: tuple[AbstractUnit | str | None, ...],
) -> Callable[[*Args], R]:
    """Gradient of a function with units.

    In general, if you can use ``quax.quaxify(jax.grad(func))`` (or the
    syntactic sugar ``quaxed.grad(func)``), that's the better option! The
    difference from those functions is how units are handled. ``quaxify``
    will directly propagate the units through the automatic differentiation
    functions. But sometimes that doesn't work and we need to strip the units
    and re-apply them. This function does that, using the ``units`` kwarg.

    See Also
    --------
    jax.grad : The original JAX gradient function.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u

    >>> def cube_volume(x: u.Q["length"]) -> u.Q["volume"]:
    ...     return x**3

    >>> grad_cube_volume = u.experimental.grad(cube_volume, units=("m",))
    >>> grad_cube_volume(u.Q(2.0, "m"))
    Quantity(Array(12., dtype=float32...), unit='m2')

    Inputs are converted to ``units`` first, so a convertible input gives the
    same result (``200 cm`` is ``2 m``):

    >>> grad_cube_volume(u.Q(200.0, "cm"))
    Quantity(Array(12., dtype=float32...), unit='m2')

    A ``None`` entry in ``units`` marks an argument as a plain (unitless) value
    rather than a `Quantity`, so functions can mix the two:

    >>> def scaled_area(distance, factor):
    ...     return distance**2 * factor

    >>> g = u.experimental.grad(scaled_area, argnums=0, units=("m", None))
    >>> g(u.Q(3.0, "m"), 2.0)
    Quantity(Array(12., dtype=float32...), unit='m')

    """
    theunits: tuple[AbstractUnit | None, ...] = tuple(map(unit_or_none, units))

    # Gradient of function, stripping and adding units
    @ft.partial(jax.grad, argnums=argnums)
    def gradfun_mag(*args: Any) -> ArrayLike:
        args_ = (
            (a if unit is None else Quantity(a, unit))
            for a, unit in zip(args, theunits, strict=True)
        )
        return ustrip(fun(*args_))  # type: ignore[arg-type]

    def gradfun(*args: *Args) -> R:
        # Get the value of the args. They are turned back into Quantity
        # inside the function we are taking the grad of.
        args_ = tuple(  # type: ignore[var-annotated]
            (a if unit is None else ustrip(unit, a))
            for a, unit in zip(args, theunits, strict=True)  # type: ignore[arg-type]
        )
        # Evaluate the value on the same args normalized to ``units`` that the
        # gradient is computed from, so its unit is consistent — an input given
        # in a convertible unit (e.g. cm for ``units=("m",)``) yields the same
        # result as the normalized unit.
        qargs = tuple(  # type: ignore[var-annotated]
            (a if unit is None else Quantity(ustrip(unit, a), unit))
            for a, unit in zip(args, theunits, strict=True)  # type: ignore[arg-type]
        )
        value = fun(*qargs)
        grad_value = gradfun_mag(*args_)
        # Adjust the Quantity by the units of the derivative. A dimensionless
        # differentiation argument (unit ``None``) contributes no unit.
        # TODO: get Quantity[unit] / unit2 ->
        # Quantity[unit/unit2] working
        du = theunits[argnums]
        new_unit = unit_of(value) if du is None else unit_of(value) / du
        return type_unparametrized(value)(grad_value, new_unit)

    return gradfun


def jacfwd[*Args, R: AbstractQuantity](
    fun: Callable[[*Args], R],
    argnums: int = 0,
    *,
    units: tuple[AbstractUnit | str | None, ...],
) -> Callable[[*Args], R]:
    """Jacobian of ``fun`` evaluated column-by-column using forward-mode AD.

    In general, if you can use `quax.quaxify(jax.jacfwd(func))` (or the
    syntactic sugar `quax.jacfwd(func)`), that's the better option! The
    difference from those functions is how units are handled. `quaxify`
    will directly propagate the units through the automatic differentiation
    functions. But sometimes that doesn't work and we need to strip the units
    and re-apply them. This function does that, using the ``units`` kwarg.

    See Also
    --------
    jax.jacfwd : The original JAX jacfwd function.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u

    >>> def cubbe_volume(x: u.Q["length"]) -> u.Q["volume"]:
    ...     return x**3

    >>> jacfwd_cubbe_volume = u.experimental.jacfwd(cubbe_volume, units=("m",))
    >>> jacfwd_cubbe_volume(u.Q(2.0, "m"))
    Quantity(Array(12., dtype=float32...), unit='m2')

    """
    argnums = eqx.error_if(
        argnums,
        not isinstance(argnums, int),
        "only int argnums are currently supported",
    )

    theunits: tuple[AbstractUnit | None, ...] = tuple(map(unit_or_none, units))

    @ft.partial(jax.jacfwd, argnums=argnums)
    def jacfun_mag(*args: Any) -> R:
        args_ = tuple(
            (a if unit is None else Quantity(a, unit))
            for a, unit in zip(args, theunits, strict=True)
        )
        return fun(*args_)  # type: ignore[arg-type]

    def jacfun(*args: *Args) -> R:
        # Get the value of the args. They are turned back into Quantity
        # inside the function we are taking the Jacobian of.
        args_ = tuple(  # type: ignore[var-annotated]
            (a if unit is None else ustrip(unit, a))
            for a, unit in zip(args, theunits, strict=True)  # type: ignore[arg-type]
        )
        # Call the Jacobian, returning a Quantity
        value = jacfun_mag(*args_)
        # Adjust the Quantity by the units of the derivative. A dimensionless
        # differentiation argument (unit ``None``) contributes no unit.
        # TODO: check the unit correction
        # TODO: get Quantity[unit] / unit2 ->
        # Quantity[unit/unit2] working
        du = theunits[argnums]
        new_unit = unit_of(value) if du is None else unit_of(value) / du
        return type_unparametrized(value)(ustrip(value), new_unit)

    return jacfun


def hessian[*Args, R: AbstractQuantity](
    fun: Callable[[*Args], R],
    argnums: int = 0,
    *,
    units: tuple[AbstractUnit | str | None, ...],
) -> Callable[[*Args], R]:
    """Hessian.

    In general, if you can use `quax.quaxify(jax.hessian(func))` (or the
    syntactic sugar `quax.hessian(func)`), that's the better option! The
    difference from those functions is how units are handled. `quaxify`
    will directly propagate the units through the automatic differentiation
    functions. But sometimes that doesn't work and we need to strip the units
    and re-apply them. This function does that, using the ``units`` kwarg.

    See Also
    --------
    jax.hessian : The original JAX hessian function.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u

    >>> def cubbe_volume(x: u.Q["length"]) -> u.Q["volume"]:
    ...     return x**3

    >>> hessian_cubbe_volume = u.experimental.hessian(cubbe_volume, units=("m",))
    >>> hessian_cubbe_volume(u.Q(2.0, "m"))
    Quantity(Array(12., dtype=float32...), unit='m')

    ``argnums`` selects the argument to differentiate with respect to. For
    ``f(x, y) = x y**2``, the second derivative w.r.t. ``y`` is ``2 x``:

    >>> def f(x, y):
    ...     return x * y**2

    >>> hess_y = u.experimental.hessian(f, argnums=1, units=("m", "s"))
    >>> hess_y(u.Q(3.0, "m"), u.Q(4.0, "s"))
    Quantity(Array(6., dtype=float32...), unit='m')

    """
    theunits: tuple[AbstractUnit | None, ...] = tuple(map(unit_or_none, units))

    @ft.partial(jax.hessian, argnums=argnums)
    def hessfun_mag(*args: Any) -> R:
        args_ = tuple(
            (a if unit is None else Quantity(a, unit))
            for a, unit in zip(args, theunits, strict=True)
        )
        return fun(*args_)  # type: ignore[arg-type]

    def hessfun(*args: *Args) -> R:
        # Get the value of the args. They are turned back into Quantity
        # inside the function we are taking the hessian of.
        args_ = tuple(  # type: ignore[var-annotated]
            (a if unit is None else ustrip(unit, a))
            for a, unit in zip(args, theunits, strict=True)  # type: ignore[arg-type]
        )
        # Call the hessian, returning a Quantity
        value = hessfun_mag(*args_)
        # Adjust the Quantity by the units of the derivative. A dimensionless
        # differentiation argument (unit ``None``) contributes no unit.
        # TODO: check the unit correction
        # TODO: get Quantity[unit] / unit2 ->
        # Quantity[unit/unit2] working
        du = theunits[argnums]
        new_unit = unit_of(value) if du is None else unit_of(value) / du**2
        return type_unparametrized(value)(ustrip(value), new_unit)

    return hessfun
