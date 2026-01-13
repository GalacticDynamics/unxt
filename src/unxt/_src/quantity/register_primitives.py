"""Register jax primitives support for Quantity."""
# pylint: disable=import-error, too-many-lines

__all__: tuple[str, ...] = ()

from collections.abc import Sequence
from dataclasses import replace
from math import prod
from typing import Any, Literal, TypeAlias, TypeVar, overload

import equinox as eqx
import jax.tree as jt
import numpy as np
from astropy.units import (  # pylint: disable=no-name-in-module
    UnitConversionError,
    dimensionless_unscaled as one,
    radian,
)
from jax import lax, numpy as jnp
from jax._src.lax import linalg as lax_linalg
from jax.extend.core.primitives import add_jaxvals_p
from jaxtyping import Array, ArrayLike, DTypeLike, Int
from plum import convert, promote
from plum.parametric import type_unparametrized as type_np
from quax import register

from quaxed import lax as qlax

from .base import AbstractQuantity as ABCQ  # noqa: N814
from .base_angle import AbstractAngle
from .base_parametric import AbstractParametricQuantity as ABCPQ  # noqa: N814
from .flag import AllowValue
from .quantity import Q, Quantity
from .static_quantity import StaticQuantity
from .value import StaticValue
from unxt._src.utils import promote_dtypes, promote_dtypes_if_needed
from unxt_api import is_unit_convertible, uconvert, unit, unit_of, ustrip

T = TypeVar("T")

Axes: TypeAlias = tuple[int, ...]


def _to_val_rad_or_one(q: ABCQ) -> ArrayLike:
    return ustrip(radian if is_unit_convertible(q.unit, radian) else one, q)


################################################################################
# Registering Primitives

# ==============================================================================


@register(lax.abs_p)
def abs_p(x: ABCQ, /) -> ABCQ:
    """Absolute value of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> q = u.Q(-1, "m")
    >>> jnp.abs(q)
    Quantity(Array(1, dtype=int32, ...), unit='m')
    >>> abs(q)
    Quantity(Array(1, dtype=int32, ...), unit='m')

    >>> q = u.quantity.BareQuantity(-1, "m")
    >>> jnp.abs(q)
    BareQuantity(Array(1, dtype=int32, ...), unit='m')
    >>> abs(q)
    BareQuantity(Array(1, dtype=int32, ...), unit='m')

    """
    return replace(x, value=qlax.abs(ustrip(x)))


# ==============================================================================


@register(lax.acos_p)
def acos_p_aq(x: ABCQ, /) -> ABCQ:
    """Inverse cosine of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as xp
    >>> import unxt as u

    >>> q = u.quantity.BareQuantity(-1, "")
    >>> jnp.acos(q).round(4)
    BareQuantity(Array(3.1416, dtype=float32, weak_type=True), unit='rad')

    >>> q = u.Q(-1, "")
    >>> jnp.acos(q).round(4)
    Quantity(Array(3.1416, dtype=float32, ...), unit='rad')

    """
    x_ = ustrip(one, x)
    return type_np(x)(value=qlax.acos(x_), unit=radian)


# ==============================================================================


@register(lax.acosh_p)
def acosh_p_aq(x: ABCQ, /) -> ABCQ:
    """Inverse hyperbolic cosine of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> q = u.quantity.BareQuantity(2.0, "")
    >>> jnp.acosh(q)
    BareQuantity(Array(1.316958, dtype=float32, ...), unit='rad')

    >>> q = u.Q(2.0, "")
    >>> jnp.acosh(q)
    Quantity(Array(1.316958, dtype=float32, ...), unit='rad')

    """
    x_ = ustrip(one, x)
    return type_np(x)(value=qlax.acosh(x_), unit=radian)


# ==============================================================================
# Addition


@register(lax.add_p)
def add_p_aqaq(x: ABCQ, y: ABCQ, /) -> ABCQ:
    """Add two quantities.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> q1 = u.quantity.BareQuantity(1, "km")
    >>> q2 = u.quantity.BareQuantity(500.0, "m")
    >>> jnp.add(q1, q2)
    BareQuantity(Array(1.5, dtype=float32, ...), unit='km')
    >>> q1 + q2
    BareQuantity(Array(1.5, dtype=float32, ...), unit='km')

    >>> q1 = u.Q(1, "km")
    >>> q2 = u.Q(500.0, "m")
    >>> jnp.add(q1, q2)
    Quantity(Array(1.5, dtype=float32, ...), unit='km')
    >>> q1 + q2
    Quantity(Array(1.5, dtype=float32, ...), unit='km')

    >>> q1 = u.quantity.BareQuantity(1, "km")
    >>> q2 = u.Q(500.0, "m")
    >>> jnp.add(q1, q2)
    Quantity(Array(1.5, dtype=float32, weak_type=True), unit='km')
    >>> q1 + q2
    Quantity(Array(1.5, dtype=float32, weak_type=True), unit='km')

    """
    x, y = promote(x, y)

    # Strip the units to compare the values.
    xv = ustrip(x)
    yv = ustrip(x.unit, y)  # this can change the dtype
    xv, yv = promote_dtypes_if_needed((x.dtype, y.dtype), xv, yv)

    return replace(x, value=xv + yv)


@register(lax.add_p)
def add_p_vaq(x: ArrayLike, y: ABCQ, /) -> ABCQ:
    """Add a value and a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> x = jnp.asarray(500)

    `unxt.BareQuantity`:

    >>> y = u.quantity.BareQuantity(1.0, "km")

    >>> try:
    ...     jnp.add(x, y)
    ... except Exception as e:
    ...     print(e)
    'km' (length) and '' (dimensionless) are not convertible

    >>> try:
    ...     x + y
    ... except Exception as e:
    ...     print(e)
    'km' (length) and '' (dimensionless) are not convertible

    >>> y = u.quantity.BareQuantity(100.0, "")
    >>> jnp.add(x, y)
    BareQuantity(Array(600., dtype=float32, ...), unit='')

    >>> x + y
    BareQuantity(Array(600., dtype=float32, ...), unit='')

    >>> q2 = u.quantity.BareQuantity(1.0, "km")
    >>> q3 = u.quantity.BareQuantity(1_000.0, "m")
    >>> jnp.add(x, q2 / q3)
    BareQuantity(Array(501., dtype=float32, weak_type=True), unit='')

    `unxt.Quantity`:

    >>> x = jnp.asarray(500.0)
    >>> q2 = u.Q(1.0, "km")
    >>> try:
    ...     x + q2
    ... except Exception as e:
    ...     print(e)
    'km' (length) and '' (dimensionless) are not convertible

    >>> q2 = u.Q(100.0, "")
    >>> jnp.add(x, q2)
    Quantity(Array(600., dtype=float32, ...), unit='')

    >>> x + q2
    Quantity(Array(600., dtype=float32, ...), unit='')

    >>> q2 = u.Q(1.0, "km")
    >>> q3 = u.Q(1_000.0, "m")
    >>> jnp.add(x, q2 / q3)
    Quantity(Array(501., dtype=float32, weak_type=True), unit='')

    """
    y = uconvert(one, y)
    return replace(y, value=qlax.add(x, ustrip(y)))


@register(lax.add_p)
def add_p_aqv(x: ABCQ, y: ArrayLike, /) -> ABCQ:
    """Add a quantity and a value.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> y = jnp.asarray(500)

    `unxt.BareQuantity`:

    >>> q1 = u.quantity.BareQuantity(1.0, "km")

    >>> try:
    ...     jnp.add(q1, y)
    ... except Exception as e:
    ...     print(e)
    'km' (length) and '' (dimensionless) are not convertible

    >>> try:
    ...     q1 + y
    ... except Exception as e:
    ...     print(e)
    'km' (length) and '' (dimensionless) are not convertible

    >>> q1 = u.quantity.BareQuantity(100.0, "")
    >>> jnp.add(q1, y)
    BareQuantity(Array(600., dtype=float32, ...), unit='')

    >>> q1 + y
    BareQuantity(Array(600., dtype=float32, ...), unit='')

    >>> q2 = u.quantity.BareQuantity(1.0, "km")
    >>> q3 = u.quantity.BareQuantity(1_000.0, "m")
    >>> jnp.add(q2 / q3, y)
    BareQuantity(Array(501., dtype=float32, weak_type=True), unit='')

    `unxt.Quantity`:

    >>> q1 = u.Q(1.0, "km")

    >>> try:
    ...     jnp.add(q1, y)
    ... except Exception as e:
    ...     print(e)
    'km' (length) and '' (dimensionless) are not convertible

    >>> try:
    ...     q1 + y
    ... except Exception as e:
    ...     print(e)
    'km' (length) and '' (dimensionless) are not convertible

    >>> q1 = u.Q(100.0, "")
    >>> jnp.add(q1, y)
    Quantity(Array(600., dtype=float32, ...), unit='')

    >>> q1 + y
    Quantity(Array(600., dtype=float32, ...), unit='')

    >>> q2 = u.Q(1.0, "km")
    >>> q3 = u.Q(1_000.0, "m")
    >>> jnp.add(q2 / q3, y)
    Quantity(Array(501., dtype=float32, weak_type=True), unit='')

    """
    x = uconvert(one, x)
    return replace(x, value=qlax.add(ustrip(x), y))


# ==============================================================================


@register(add_jaxvals_p)
def add_jaxvals_p_qq(x: ABCPQ, y: ABCPQ, /) -> ABCPQ:
    """Add two quantities using the ``jax.interpreters.ad.add_jaxvals_p``.

    Examples
    --------
    >>> import jax
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> q1 = u.Q(1, "km")
    >>> q2 = u.Q(500.0, "m")

    >>> @jax.jit
    ... def f(x, y):
    ...     return add_jaxvals_p_qq(x, y)

    >>> f(q1, q2)
    Quantity(Array(1.5, dtype=float32), unit='km')

    """
    xv, yv = ustrip(x), ustrip(x.unit, y)
    xv, yv = promote_dtypes(xv, yv)
    return replace(x, value=add_jaxvals_p.bind(xv, yv))  # type: ignore[no-untyped-call]


# ==============================================================================


@register(lax.and_p)
def and_p_aq(x1: ABCQ, x2: ABCQ, /) -> ArrayLike:
    """Bitwise AND of two quantities.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> x1 = u.quantity.BareQuantity(1, "")
    >>> x2 = u.quantity.BareQuantity(2, "")
    >>> jnp.bitwise_and(x1, x2)
    Array(0, dtype=int32, ...)

    >>> x1 = u.Q(1, "")
    >>> x2 = u.Q(2, "")
    >>> jnp.bitwise_and(x1, x2)
    Array(0, dtype=int32, ...)

    """
    return lax.and_p.bind(ustrip(one, x1), ustrip(one, x2))


# ==============================================================================


@register(lax.approx_top_k_p)
def approx_top_k_p(x: ABCQ, /, **kwargs: Any) -> ABCQ:
    """Approximate top-k of a quantity.

    Examples
    --------
    >>> import quaxed.lax as qlax
    >>> import unxt as u

    >>> x = u.quantity.BareQuantity([1.0, 2, 3], "m")
    >>> qlax.approx_max_k(x, k=2)
    [BareQuantity(Array([3., 2.], dtype=float32), unit='m'),
     BareQuantity(Array([2., 1.], dtype=float32), unit='m')]

    """
    return replace(x, value=lax.approx_top_k_p.bind(ustrip(x), **kwargs))  # type: ignore[no-untyped-call]


# ==============================================================================


@register(lax.argmax_p)
def argmax_p(
    operand: ABCQ, /, *, axes: int | tuple[int, ...], index_dtype: DTypeLike
) -> Array:
    """Argmax of a Quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> x = u.Q([1, 2, 3], "m")
    >>> jnp.argmax(x)
    Array(2, dtype=int32)

    >>> x = u.quantity.BareQuantity([1, 2, 3], "m")
    >>> jnp.argmax(x)
    Array(2, dtype=int32)

    """
    return lax.argmax_p.bind(ustrip(operand), axes=axes, index_dtype=index_dtype)


# ==============================================================================


@register(lax.argmin_p)
def argmin_p(
    operand: ABCQ, *, axes: int | tuple[int, ...], index_dtype: DTypeLike
) -> Array:
    """Argmin of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> x = u.Q([1, 2, 3], "m")
    >>> jnp.argmin(x)
    Array(0, dtype=int32)

    >>> x = u.quantity.BareQuantity([1, 2, 3], "m")
    >>> jnp.argmin(x)
    Array(0, dtype=int32)

    """
    return lax.argmin_p.bind(ustrip(operand), axes=axes, index_dtype=index_dtype)


# ==============================================================================


@register(lax.asin_p)
def asin_p_aq(x: ABCQ) -> ABCQ:
    """Inverse sine of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> q = u.quantity.BareQuantity(1, "")
    >>> jnp.asin(q)
    BareQuantity(Array(1.5707964, dtype=float32, ...), unit='rad')

    """
    return type_np(x)(lax.asin(ustrip(one, x)), unit=radian)


@register(lax.asin_p)
def asin_p_q(x: ABCPQ["dimensionless"]) -> ABCPQ["angle"]:
    """Inverse sine of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> q = u.Q(1, "")
    >>> jnp.asin(q)
    Quantity(Array(1.5707964, dtype=float32, ...), unit='rad')

    """
    return type_np(x)(lax.asin(ustrip(one, x)), unit=radian)


# ==============================================================================


@register(lax.asinh_p)
def asinh_p_aq(x: ABCQ) -> ABCQ:
    """Inverse hyperbolic sine of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> q = u.quantity.BareQuantity(2, "")
    >>> jnp.asinh(q)
    BareQuantity(Array(1.4436355, dtype=float32, ...), unit='rad')

    """
    return type_np(x)(lax.asinh(ustrip(one, x)), unit=radian)


@register(lax.asinh_p)
def asinh_p_q(x: ABCPQ["dimensionless"]) -> ABCPQ["angle"]:
    """Inverse hyperbolic sine of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> q = u.Q(2, "")
    >>> jnp.asinh(q)
    Quantity(Array(1.4436355, dtype=float32, ...), unit='rad')

    """
    return type_np(x)(lax.asinh(ustrip(one, x)), unit=radian)


# ==============================================================================


@register(lax.atan2_p)
def atan2_p_aqaq(x: ABCQ, y: ABCQ) -> ABCQ:
    """Arctangent2 of two abstract quantities.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> q1 = u.quantity.BareQuantity(1, "m")
    >>> q2 = u.quantity.BareQuantity(3.0, "m")
    >>> jnp.atan2(q1, q2)
    BareQuantity(Array(0.32175055, dtype=float32, ...), unit='rad')

    """
    x, y = promote(x, y)  # e.g. Distance -> Quantity
    yv = ustrip(x.unit, y)
    return type_np(x)(lax.atan2(ustrip(x), yv), unit=radian)


@register(lax.atan2_p)
def atan2_p_qq(x: ABCPQ, y: ABCPQ) -> ABCPQ["radian"]:
    """Arctangent2 of two quantities.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> q1 = u.Q(1, "m")
    >>> q2 = u.Q(3.0, "m")
    >>> jnp.atan2(q1, q2)
    Quantity(Array(0.32175055, dtype=float32, ...), unit='rad')

    """
    x, y = promote(x, y)  # e.g. Distance -> Quantity
    yv = ustrip(x.unit, y)
    return type_np(x)(lax.atan2(ustrip(x), yv), unit=radian)


# ---------------------------


@register(lax.atan2_p)
def atan2_p_vaq(x: ArrayLike, y: ABCQ) -> ABCQ:
    """Arctangent2 of a value and a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> x1 = jnp.asarray(1.0)
    >>> q2 = u.quantity.BareQuantity(3, "")
    >>> jnp.atan2(x1, q2)
    BareQuantity(Array(0.32175055, dtype=float32, ...), unit='rad')

    """
    yv = ustrip(one, y)
    return type_np(y)(lax.atan2(x, yv), unit=radian)


@register(lax.atan2_p)
def atan2_p_vq(x: ArrayLike, y: ABCPQ["dimensionless"]) -> ABCPQ["angle"]:
    """Arctangent2 of a value and a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> x1 = jnp.asarray(1.0)
    >>> q2 = u.Q(3, "")
    >>> jnp.atan2(x1, q2)
    Quantity(Array(0.32175055, dtype=float32, ...), unit='rad')

    """
    yv = ustrip(one, y)
    return Quantity(lax.atan2(x, yv), unit=radian)


# ---------------------------


@register(lax.atan2_p)
def atan2_p_aqv(x: ABCQ, y: ArrayLike) -> ABCQ:
    """Arctangent2 of a quantity and a value.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> q1 = u.quantity.BareQuantity(1.0, "")
    >>> x2 = jnp.asarray(3)
    >>> jnp.atan2(q1, x2)
    BareQuantity(Array(0.32175055, dtype=float32, ...), unit='rad')

    """
    xv = ustrip(one, x)
    return type_np(x)(lax.atan2(xv, y), unit=radian)


@register(lax.atan2_p)
def atan2_p_qv(x: ABCPQ["dimensionless"], y: ArrayLike) -> ABCPQ["angle"]:
    """Arctangent2 of a quantity and a value.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> q1 = u.Q(1.0, "")
    >>> x2 = jnp.asarray(3)
    >>> jnp.atan2(q1, x2)
    Quantity(Array(0.32175055, dtype=float32, ...), unit='rad')

    """
    xv = ustrip(one, x)
    return type_np(x)(lax.atan2(xv, y), unit=radian)


# ==============================================================================


@register(lax.atan_p)
def atan_p_aq(x: ABCQ) -> ABCQ:
    """Arctangent of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> q = u.quantity.BareQuantity(1, "")
    >>> jnp.atan(q)
    BareQuantity(Array(0.7853982, dtype=float32, ...), unit='rad')

    """
    return type_np(x)(lax.atan(ustrip(one, x)), unit=radian)


@register(lax.atan_p)
def atan_p_q(x: ABCPQ["dimensionless"]) -> ABCPQ["angle"]:
    """Arctangent of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> q = u.Q(1, "")
    >>> jnp.atan(q)
    Quantity(Array(0.7853982, dtype=float32, ...), unit='rad')

    """
    return Quantity(lax.atan(ustrip(one, x)), unit=radian)


# ==============================================================================


@register(lax.atanh_p)
def atanh_p_aq(x: ABCQ) -> ABCQ:
    """Inverse hyperbolic tangent of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> q = u.quantity.BareQuantity(2, "")
    >>> jnp.atanh(q)
    BareQuantity(Array(nan, dtype=float32, ...), unit='rad')

    """
    return type_np(x)(lax.atanh(ustrip(one, x)), unit=radian)


@register(lax.atanh_p)
def atanh_p_q(x: ABCPQ["dimensionless"]) -> ABCPQ["angle"]:
    """Inverse hyperbolic tangent of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> q = u.Q(2, "")
    >>> jnp.atanh(q)
    Quantity(Array(nan, dtype=float32, ...), unit='rad')

    """
    return type_np(x)(lax.atanh(ustrip(one, x)), unit=radian)


# ==============================================================================


@register(lax.bessel_i0e_p)
def bessel_i0e_p(x: ABCQ, /, **kwargs: Any) -> ABCQ:
    r"""Return modified Bessel function of the first kind of order zero.

    Examples
    --------
    >>> import quaxed.lax as qlax

    >>> x = u.quantity.BareQuantity(1.0, "")
    >>> qlax.bessel_i0e(x)
    BareQuantity(Array(0.46575963, dtype=float32, weak_type=True), unit='')

    >>> x = u.Q(1.0, "")
    >>> qlax.bessel_i0e(x)
    Quantity(Array(0.46575963, dtype=float32, weak_type=True), unit='')

    """
    return replace(x, value=lax.bessel_i0e_p.bind(ustrip(one, x), **kwargs))


@register(lax.bessel_i1e_p)
def bessel_i1e_p(x: ABCQ, /, **kwargs: Any) -> ABCQ:
    r"""Return modified Bessel function of the first kind of order one.

    Examples
    --------
    >>> import quaxed.lax as qlax

    >>> x = u.quantity.BareQuantity(1.0, "")
    >>> qlax.bessel_i1e(x)
    BareQuantity(Array(0.20791042, dtype=float32, ...), unit='')

    >>> x = u.Q(1.0, "")
    >>> qlax.bessel_i1e(x)
    Quantity(Array(0.20791042, dtype=float32, ...), unit='')

    """
    return replace(x, value=lax.bessel_i1e_p.bind(ustrip(one, x), **kwargs))


# ==============================================================================


@register(lax.bitcast_convert_type_p)
def bitcast_convert_type_p(x: ABCQ, /, *, new_dtype: DTypeLike) -> ABCQ:
    """Bitcast convert type of a quantity.

    Examples
    --------
    >>> import quaxed.lax as qlax

    >>> x = u.quantity.BareQuantity(1.0, "")
    >>> qlax.bitcast_convert_type(x, jnp.int16)
    BareQuantity(Array([    0, 16256], dtype=int16), unit='')

    >>> x = u.Q(1.0, "")
    >>> qlax.bitcast_convert_type(x, jnp.int16)
    Quantity(Array([    0, 16256], dtype=int16), unit='')

    """
    return replace(
        x, value=lax.bitcast_convert_type_p.bind(ustrip(x), new_dtype=new_dtype)
    )


# ==============================================================================


@register(lax.broadcast_in_dim_p)
def broadcast_in_dim_p(operand: ABCQ, /, **kw: Any) -> ABCQ:
    """Broadcast a quantity in a specific dimension."""
    value = lax.broadcast_in_dim_p.bind(ustrip(operand), **kw)  # type: ignore[no-untyped-call,unused-ignore]
    return replace(operand, value=value)


# ==============================================================================


@register(lax.cbrt_p)
def cbrt_p_q(x: ABCQ, /, **kw: Any) -> ABCQ:
    """Cube root of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> q = u.quantity.BareQuantity(8, "m3")
    >>> jnp.cbrt(q)
    BareQuantity(Array(2., dtype=float32, ...), unit='m')

    >>> q = u.Q(8, "m3")
    >>> jnp.cbrt(q)
    Quantity(Array(2., dtype=float32, ...), unit='m')

    """
    return type_np(x)(lax.cbrt_p.bind(ustrip(x), **kw), unit=x.unit ** (1 / 3))


# TODO: can this be done with promotion/conversion/default rule instead?
@register(lax.cbrt_p)
def cbrt_p_abstractangle(x: AbstractAngle, /, **kw: Any) -> ABCQ:
    """Cube root of an angle.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> q = u.Angle(8, "rad")
    >>> jnp.cbrt(q)
    Quantity(Array(2., dtype=float32, ...), unit='rad(1/3)')

    """
    return cbrt_p_q(convert(x, Quantity), **kw)


# ==============================================================================


@register(lax.ceil_p)
def ceil_p(x: ABCQ, /) -> ABCQ:
    """Ceiling of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> q = u.quantity.BareQuantity(1.5, "m")
    >>> jnp.ceil(q)
    BareQuantity(Array(2., dtype=float32, ...), unit='m')

    >>> q = u.Q(1.5, "m")
    >>> jnp.ceil(q)
    Quantity(Array(2., dtype=float32, ...), unit='m')

    """
    return replace(x, value=qlax.ceil(ustrip(x)))


# ==============================================================================


@register(lax.clamp_p)
def clamp_p(min: ABCQ, x: ABCQ, max: ABCQ) -> ABCQ:
    """Clamp a quantity between two other quantities.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import quaxed.lax as lax
    >>> import unxt as u

    >>> min = u.quantity.BareQuantity(0, "m")
    >>> max = u.quantity.BareQuantity(2, "m")
    >>> q = u.quantity.BareQuantity([-1, 1, 3], "m")
    >>> lax.clamp(min, q, max)
    BareQuantity(Array([0, 1, 2], dtype=int32), unit='m')

    >>> jnp.clip(q.astype(float), min, max)
    BareQuantity(Array([0., 1., 2.], dtype=float32), unit='m')

    >>> min = u.Q(0, "m")
    >>> max = u.Q(2, "m")
    >>> q = u.Q([-1, 1, 3], "m")
    >>> lax.clamp(min, q, max)
    Quantity(Array([0, 1, 2], dtype=int32), unit='m')

    >>> jnp.clip(q.astype(float), min, max)
    Quantity(Array([0., 1., 2.], dtype=float32), unit='m')

    """
    return replace(
        x, value=qlax.clamp(ustrip(x.unit, min), ustrip(x), ustrip(x.unit, max))
    )


# ---------------------------


@register(lax.clamp_p)
def clamp_p_vaqaq(min: ArrayLike, x: ABCQ, max: ABCQ) -> ABCQ:
    """Clamp a quantity between a value and another quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import quaxed.lax as lax
    >>> import unxt as u

    >>> min = jnp.asarray(0)
    >>> max = u.quantity.BareQuantity(2, "")
    >>> q = u.quantity.BareQuantity([-1, 1, 3], "")
    >>> lax.clamp(min, q, max)
    BareQuantity(Array([0, 1, 2], dtype=int32), unit='')

    >>> min = jnp.asarray(0)
    >>> max = u.Q(2, "")
    >>> q = u.Q([-1, 1, 3], "")
    >>> lax.clamp(min, q, max)
    Quantity(Array([0, 1, 2], dtype=int32), unit='')

    """
    return replace(x, value=qlax.clamp(min, ustrip(one, x), ustrip(one, max)))


# ---------------------------


@register(lax.clamp_p)
def clamp_p_aqvaq(min: ABCQ, x: ArrayLike, max: ABCQ) -> ArrayLike:
    """Clamp a value between two quantities.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import quaxed.lax as lax
    >>> import unxt as u

    >>> min = u.quantity.BareQuantity(0, "")
    >>> max = u.quantity.BareQuantity(2, "")
    >>> x = jnp.asarray([-1, 1, 3])
    >>> lax.clamp(min, x, max)
    Array([0, 1, 2], dtype=int32)

    """
    return lax.clamp(ustrip(one, min), x, ustrip(one, max))


@register(lax.clamp_p)
def clamp_p_qvq(
    min: ABCPQ["dimensionless"], x: ArrayLike, max: ABCPQ["dimensionless"]
) -> ArrayLike:
    """Clamp a value between two quantities.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import quaxed.lax as lax

    >>> min = u.Q(0, "")
    >>> max = u.Q(2, "")
    >>> x = jnp.asarray([-1, 1, 3])
    >>> lax.clamp(min, x, max)
    Array([0, 1, 2], dtype=int32)

    """
    return lax.clamp(ustrip(one, min), x, ustrip(one, max))


# ---------------------------


@register(lax.clamp_p)
def clamp_p_aqaqv(min: ABCQ, x: ABCQ, max: ArrayLike) -> ABCQ:
    """Clamp a quantity between a quantity and a value.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import quaxed.lax as lax
    >>> import unxt as u

    >>> min = u.quantity.BareQuantity(0, "")
    >>> max = jnp.asarray(2)
    >>> q = u.quantity.BareQuantity([-1, 1, 3], "")
    >>> lax.clamp(min, q, max)
    BareQuantity(Array([0, 1, 2], dtype=int32), unit='')

    """
    return replace(x, value=qlax.clamp(ustrip(one, min), ustrip(one, x), max))


@register(lax.clamp_p)
def clamp_p_qqv(
    min: ABCPQ["dimensionless"], x: ABCPQ["dimensionless"], max: ArrayLike
) -> ABCPQ["dimensionless"]:
    """Clamp a quantity between a quantity and a value.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import quaxed.lax as lax

    >>> min = u.Q(0, "")
    >>> max = jnp.asarray(2)
    >>> q = u.Q([-1, 1, 3], "")
    >>> lax.clamp(min, q, max)
    Quantity(Array([0, 1, 2], dtype=int32), unit='')

    """
    return replace(x, value=qlax.clamp(ustrip(one, min), ustrip(one, x), max))


# ==============================================================================


@register(lax.clz_p)
def clz_p(x: ABCQ, /) -> ABCQ:
    """Count leading zeros of a quantity.

    Examples
    --------
    >>> import quaxed.lax as qlax
    >>> import unxt as u

    >>> q = u.quantity.BareQuantity(1, "")
    >>> qlax.clz(q)
    BareQuantity(Array(31, dtype=int32, ...), unit='')

    >>> q = u.Q(1, "")
    >>> qlax.clz(q)
    Quantity(Array(31, dtype=int32, ...), unit='')

    """
    return replace(x, value=lax.clz_p.bind(ustrip(x)))


# ==============================================================================


@register(lax.complex_p)
def complex_p(x: ABCQ, y: ABCQ) -> ABCQ:
    """Complex number from two quantities.

    Examples
    --------
    >>> from quaxed import lax
    >>> import unxt as u

    >>> x = u.quantity.BareQuantity(1.0, "m")
    >>> y = u.quantity.BareQuantity(2.0, "m")
    >>> lax.complex(x, y)
    BareQuantity(Array(1.+2.j, dtype=complex64, ...), unit='m')

    >>> x = u.Q(1.0, "m")
    >>> y = u.Q(2.0, "m")
    >>> lax.complex(x, y)
    Quantity(Array(1.+2.j, dtype=complex64, ...), unit='m')

    """
    x, y = promote(x, y)  # e.g. Distance -> Quantity
    y_ = ustrip(x.unit, y)
    return replace(x, value=qlax.complex(ustrip(x), y_))


# ==============================================================================
# Concatenation


@register(lax.concatenate_p)
def concatenate_p_aq(*operands: ABCQ, dimension: Any) -> ABCQ:
    """Concatenate quantities.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> q1 = u.quantity.BareQuantity([1.0], "km")
    >>> q2 = u.quantity.BareQuantity([2_000.0], "m")
    >>> jnp.concat([q1, q2])
    BareQuantity(Array([1., 2.], dtype=float32), unit='km')

    >>> q1 = u.Q([1.0], "km")
    >>> q2 = u.Q([2_000.0], "m")
    >>> jnp.concat([q1, q2])
    Quantity(Array([1., 2.], dtype=float32), unit='km')

    """
    operand0 = operands[0]
    u = operand0.unit
    return replace(
        operand0,
        value=qlax.concatenate([ustrip(u, op) for op in operands], dimension=dimension),
    )


# ---------------------------


@register(lax.concatenate_p, precedence=1)
def concatenate_p_qnd(
    operand0: ABCPQ["dimensionless"],
    *operands: ABCPQ["dimensionless"] | ArrayLike,
    dimension: Any,
) -> ABCPQ["dimensionless"]:
    """Concatenate quantities and arrays with dimensionless units.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> theta = u.Q(45, "deg")
    >>> Rz = jnp.asarray(
    ...     [
    ...         [jnp.cos(theta), -jnp.sin(theta), 0],
    ...         [jnp.sin(theta), jnp.cos(theta), 0],
    ...         [0, 0, 1],
    ...     ]
    ... )
    >>> Rz
    Quantity(Array([[ 0.70710677, -0.70710677,  0.        ],
                         [ 0.70710677,  0.70710677,  0.        ],
                         [ 0.        ,  0.        ,  1.        ]], dtype=float32),
                  unit='')

    """
    return type_np(operand0)(
        lax.concatenate(
            [
                (ustrip(one, op) if hasattr(op, "unit") else op)
                for op in (operand0, *operands)
            ],
            dimension=dimension,
        ),
        unit=one,
    )


@register(lax.concatenate_p)
def concatenate_p_vqnd(operand0: ArrayLike, *operands: ABCQ, dimension: Any) -> ABCQ:
    """Concatenate quantities and arrays with dimensionless units.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> theta = u.Q(45, "deg")
    >>> Rx = jnp.asarray(
    ...     [
    ...         [1.0, 0.0, 0.0],
    ...         [0.0, jnp.cos(theta), -jnp.sin(theta)],
    ...         [0.0, jnp.sin(theta), jnp.cos(theta)],
    ...     ]
    ... )
    >>> Rx
    Quantity(Array([[ 1.        ,  0.        ,  0.        ],
                         [ 0.        ,  0.70710677, -0.70710677],
                         [ 0.        ,  0.70710677,  0.70710677]], dtype=float32),
                  unit='')

    """
    arrs = [operand0, *(ustrip(one, op) for op in operands)]
    return Quantity(lax.concatenate(arrs, dimension=dimension), unit=one)


# ==============================================================================


@register(lax.cond_p)  # TODO: implement
def cond_p_q(index: ABCQ, consts: ABCQ) -> ABCQ:
    raise NotImplementedError


@register(lax.cond_p)  # TODO: implement
def cond_p_vq(index: ArrayLike, consts: ABCQ, *, branches: Any) -> ABCQ:
    """Conditional on a value and a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    """
    return lax.cond_p.bind(index, ustrip(consts), branches=branches)  # type: ignore[no-untyped-call]


# ==============================================================================


@register(lax.conj_p)
def conj_p(x: ABCQ, *, input_dtype: Any) -> ABCQ:
    """Conjugate of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> q = u.quantity.BareQuantity(1 + 2j, "m")
    >>> jnp.conj(q)
    BareQuantity(Array(1.-2.j, dtype=complex64, ...), unit='m')

    >>> q = u.Q(1 + 2j, "m")
    >>> jnp.conj(q)
    Quantity(Array(1.-2.j, dtype=complex64, ...), unit='m')

    """
    del input_dtype  # TODO: use this?
    return replace(x, value=qlax.conj(ustrip(x)))


# ==============================================================================


@register(lax.convert_element_type_p)
def convert_element_type_p(operand: ABCQ, /, **kw: Any) -> ABCQ:
    """Convert the element type of a quantity."""
    # TODO: examples
    # For StaticQuantity, use numpy's astype to avoid converting to JAX array
    if isinstance(operand, StaticQuantity):
        new_dtype = kw.get("new_dtype")
        value = StaticValue(operand.value.array.astype(new_dtype))
    else:
        value = lax.convert_element_type_p.bind(ustrip(operand), **kw)

    return replace(operand, value=value)


# ==============================================================================


@register(lax.copy_p)
def copy_p(x: ABCQ) -> ABCQ:
    """Copy a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> q = u.quantity.BareQuantity(1, "m")
    >>> jnp.copy(q)
    BareQuantity(Array(1, dtype=int32, ...), unit='m')

    >>> q = u.Q(1, "m")
    >>> jnp.copy(q)
    Quantity(Array(1, dtype=int32, ...), unit='m')

    """
    return replace(x, value=lax.copy_p.bind(ustrip(x)))  # type: ignore[no-untyped-call]


# ==============================================================================


@register(lax.cos_p)
def cos_p_aq(x: ABCQ, /, **kw: Any) -> ABCQ:
    """Cosine of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> q = u.quantity.BareQuantity(1, "rad")
    >>> jnp.cos(q)
    BareQuantity(Array(0.5403023, dtype=float32, ...), unit='')

    >>> q = u.quantity.BareQuantity(1, "")
    >>> jnp.cos(q)
    BareQuantity(Array(0.5403023, dtype=float32, ...), unit='')

    """
    return type_np(x)(lax.cos_p.bind(_to_val_rad_or_one(x), **kw), unit=one)


@register(lax.cos_p)
def cos_p_q(x: ABCPQ["angle"] | Q["dimensionless"], /, **kw: Any) -> Q["dimensionless"]:
    """Cosine of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> q = u.Q(1, "rad")
    >>> jnp.cos(q)
    Quantity(Array(0.5403023, dtype=float32, ...), unit='')

    >>> q = u.Q(1, "")
    >>> jnp.cos(q)
    Quantity(Array(0.5403023, dtype=float32, ...), unit='')

    """
    return Quantity(lax.cos_p.bind(_to_val_rad_or_one(x), **kw), unit=one)


@register(lax.cos_p)
def cos_p_abstractangle(x: AbstractAngle, /, **kw: Any) -> Quantity:
    """Cosine of an Angle.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> q = u.Angle(0, "deg")
    >>> jnp.cos(q)
    Quantity(Array(1., dtype=float32, ...), unit='')

    """
    return cos_p_q(convert(x, Quantity), **kw)


# ==============================================================================


@register(lax.cosh_p)
def cosh_p_aq(x: ABCQ) -> ABCQ:
    """Cosine of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> q = u.quantity.BareQuantity(1, "rad")
    >>> jnp.cosh(q)
    BareQuantity(Array(1.5430806, dtype=float32, ...), unit='')

    >>> q = u.quantity.BareQuantity(1, "")
    >>> jnp.cosh(q)
    BareQuantity(Array(1.5430806, dtype=float32, ...), unit='')

    """
    return type_np(x)(lax.cosh(_to_val_rad_or_one(x)), unit=one)


@register(lax.cosh_p)
def cosh_p_q(x: ABCPQ["angle"] | Q["dimensionless"]) -> ABCPQ["dimensionless"]:
    """Cosine of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> q = u.Q(1, "rad")
    >>> jnp.cosh(q)
    Quantity(Array(1.5430806, dtype=float32, ...), unit='')

    >>> q = u.Q(1, "")
    >>> jnp.cosh(q)
    Quantity(Array(1.5430806, dtype=float32, ...), unit='')

    """
    return type_np(x)(lax.cosh(_to_val_rad_or_one(x)), unit=one)


# ==============================================================================


@register(lax.cumlogsumexp_p)
def cumlogsumexp_p(operand: ABCQ, *, axis: Any, reverse: Any) -> ABCQ:
    """Cumulative log sum exp of a quantity.

    Examples
    --------
    >>> from quaxed import lax
    >>> import unxt as u

    >>> q = u.quantity.BareQuantity([-1.0, -2, -3], "")
    >>> lax.cumlogsumexp(q)
    BareQuantity(Array([-1. , -0.6867383 , -0.59239405], dtype=float32), unit='')

    >>> q = u.Q([-1.0, -2, -3], "")
    >>> lax.cumlogsumexp(q)
    Quantity(Array([-1. , -0.6867383 , -0.59239405], dtype=float32),
                              unit='')

    """
    # TODO: double check units make sense here.
    return replace(
        operand, value=qlax.cumlogsumexp(ustrip(operand), axis=axis, reverse=reverse)
    )


# ==============================================================================


@register(lax.cummax_p)
def cummax_p(operand: ABCQ, *, axis: Any, reverse: Any) -> ABCQ:
    """Cumulative maximum of a quantity.

    Examples
    --------
    >>> from quaxed import lax
    >>> import unxt as u

    >>> q = u.quantity.BareQuantity([1, 2, 1], "m")
    >>> lax.cummax(q)
    BareQuantity(Array([1, 2, 2], dtype=int32), unit='m')

    >>> q = u.Q([1, 2, 1], "m")
    >>> lax.cummax(q)
    Quantity(Array([1, 2, 2], dtype=int32), unit='m')

    """
    return replace(
        operand, value=qlax.cummax(ustrip(operand), axis=axis, reverse=reverse)
    )


# ==============================================================================


@register(lax.cummin_p)
def cummin_p(operand: ABCQ, *, axis: Any, reverse: Any) -> ABCQ:
    """Cumulative maximum of a quantity.

    Examples
    --------
    >>> from quaxed import lax
    >>> import unxt as u

    >>> q = u.quantity.BareQuantity([2, 1, 3], "m")
    >>> lax.cummin(q)
    BareQuantity(Array([2, 1, 1], dtype=int32), unit='m')

    >>> q = u.Q([2, 1, 3], "m")
    >>> lax.cummin(q)
    Quantity(Array([2, 1, 1], dtype=int32), unit='m')

    """
    return replace(
        operand, value=qlax.cummin(ustrip(operand), axis=axis, reverse=reverse)
    )


# ==============================================================================


@register(lax.cumprod_p)
def cumprod_p(operand: ABCQ, *, axis: Any, reverse: Any) -> ABCQ:
    """Cumulative product of a quantity.

    Examples
    --------
    >>> from quaxed import lax
    >>> import unxt as u

    >>> q = u.quantity.BareQuantity([1, 2, 3], "")
    >>> lax.cumprod(q)
    BareQuantity(Array([1, 2, 6], dtype=int32), unit='')

    >>> q = u.Q([1, 2, 3], "")
    >>> lax.cumprod(q)
    Quantity(Array([1, 2, 6], dtype=int32), unit='')

    """
    return replace(
        operand, value=qlax.cumprod(ustrip(one, operand), axis=axis, reverse=reverse)
    )


# ==============================================================================


@register(lax.cumsum_p)
def cumsum_p(operand: ABCQ, *, axis: Any, reverse: Any) -> ABCQ:
    """Cumulative sum of a quantity.

    Examples
    --------
    >>> from quaxed import lax
    >>> import unxt as u

    >>> q = u.quantity.BareQuantity([1, 2, 3], "m")
    >>> lax.cumsum(q)
    BareQuantity(Array([1, 3, 6], dtype=int32), unit='m')

    >>> q = u.Q([1, 2, 3], "m")
    >>> lax.cumsum(q)
    Quantity(Array([1, 3, 6], dtype=int32), unit='m')

    """
    return replace(
        operand, value=qlax.cumsum(ustrip(operand), axis=axis, reverse=reverse)
    )


# ==============================================================================


@register(lax.linear_solve_p)
def custom_linear_solve_q(
    x0: ABCQ, x1: ABCQ, x2: ABCQ, x3: Array, x4: ABCQ, x5: Array, x6: ABCQ, /, **kw: Any
) -> Any:
    """Handle linear solve Ax = b where A and b are Quantities.

    Returns a list [result] to match JAX primitive conventions.
    This prevents Quax from trying to iterate over the Quantity result
    (which would cause tracer leaks in vectorized/JIT contexts).

    For linear solve Ax = b, the result x has units b/A.
    x0-x4 are related to the matrix A, x6 is the RHS b.
    """
    # For linear solve Ax = b, the result x has units b/A
    # x0-x4 are related to the matrix A
    # x6 is the right-hand side b
    u0 = unit_of(x0)
    u6 = unit_of(x6)
    result_unit = unit(u6 / u0)

    result_value = lax.linear_solve_p.bind(  # type: ignore[no-untyped-call]
        ustrip(u0, x0),
        ustrip(u0, x1),
        ustrip(u0, x2),
        x3,
        ustrip(u0, x4),
        x5,
        ustrip(u6, x6),
        **kw,
    )

    # Cannot use replace() because physical type may change (e.g., m*s -> s when
    # dividing by m) Use type_np to get non-parametric Quantity constructor
    # Return as list to match JAX primitive conventions
    return [type_np(x6)(result_value, unit=result_unit)]


@register(lax.linear_solve_p)
def custom_linear_solve_q_array_arg6(
    x0: ABCQ,
    x1: ABCQ,
    x2: ABCQ,
    x3: Array,
    x4: ABCQ,
    x5: Array,
    x6: Array,
    /,
    **kw: Any,
) -> Any:
    """Handle case where x6 is a plain Array instead of a Quantity.

    If x6 (the RHS) is a plain array, treat it as dimensionless.
    Returns a list [result] to match JAX primitive conventions.
    """
    u0 = unit_of(x0)
    # x6 is dimensionless, so result has units 1/matrix_unit
    result_unit = unit(one / u0)

    result_value = lax.linear_solve_p.bind(  # type: ignore[no-untyped-call]
        ustrip(u0, x0), ustrip(u0, x1), ustrip(u0, x2), x3, ustrip(u0, x4), x5, x6, **kw
    )

    # Cannot use replace() because physical type changes (m -> 1/m)
    # Use type_np to get non-parametric Quantity constructor
    # Return as list to match JAX primitive conventions
    return [type_np(x0)(result_value, unit=result_unit)]


# ==============================================================================


@overload
def svd_p_q(
    x: ABCQ, /, *, compute_uv: Literal[False], **params: Any
) -> tuple[ABCQ]: ...


@overload
def svd_p_q(
    x: ABCQ, /, *, compute_uv: Literal[True] = ..., **params: Any
) -> tuple[ABCQ, ArrayLike, ArrayLike]: ...


@register(lax_linalg.svd_p)
def svd_p_q(
    x: ABCQ, /, *, compute_uv: Literal[True, False] = True, **params: Any
) -> tuple[ABCQ] | tuple[ABCQ, ArrayLike, ArrayLike]:
    """SVD decomposition of a quantity matrix.

    For a matrix with units (e.g., 'm'), SVD returns:
    - When compute_uv=True: [U, S, VT] where S has input units
    - When compute_uv=False: [S] (singular values with input units)

    Note: The SVD primitive always returns a list, even when compute_uv=False.
    The high-level jnp.linalg.svd unpacks this appropriately.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> x = u.Q([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], "m")
    >>> u, s, vt = jnp.linalg.svd(x, full_matrices=False)
    >>> isinstance(s, Quantity)
    True
    >>> s.unit
    Unit("m")

    """
    u_in = unit_of(x)

    # Perform SVD on the unitless values
    # Result is always a list: [U, S, VT] or [S]
    result = lax_linalg.svd_p.bind(ustrip(u_in, x), compute_uv=compute_uv, **params)

    if compute_uv:
        # result is [S, U, VT]
        # U and VT are orthonormal matrices (dimensionless)
        # S contains singular values (same units as input)
        s, u_mat, vt = result
        # Return list of Quantities for all three, with U and VT dimensionless
        return (type_np(x)(s, unit=u_in), u_mat, vt)
    # result is [S] (list with single element)
    (s,) = result
    return (type_np(x)(s, unit=u_in),)


# ==============================================================================


@register(lax_linalg.cholesky_p)
def cholesky_p_q(x: ABCQ, /, **params: Any) -> ABCQ:
    """Cholesky decomposition of a quantity matrix.

    For a positive definite matrix A = L·L^T, if A has units u^2,
    then L has units u (square root of input units).

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> x = u.Q([[4.0, 2.0], [2.0, 3.0]], "m2")
    >>> L = jnp.linalg.cholesky(x)
    >>> isinstance(L, Quantity)
    True
    >>> L.unit
    Unit("m")

    """
    u_in = unit_of(x)
    # Cholesky: A = L·L^T, so if A has units u^2, L has units u
    u_out = unit(u_in**0.5)

    # Perform Cholesky decomposition on unitless values
    result = lax_linalg.cholesky_p.bind(ustrip(x), **params)

    # Use type_np to avoid physical type constraints
    return type_np(x)(result, unit=u_out)


# ==============================================================================


@register(lax_linalg.qr_p)
def qr_p_q(x: ABCQ, /, **params: Any) -> Any:
    """QR decomposition of a quantity matrix.

    For QR decomposition A = Q·R:
    - Q is orthonormal (dimensionless)
    - R is upper triangular with the same units as the input

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> x = u.Q([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], "m")
    >>> q, r = jnp.linalg.qr(x)
    >>> isinstance(q, Quantity)
    True
    >>> q.unit
    Unit(dimensionless)
    >>> r.unit
    Unit("m")

    """
    u_in = unit_of(x)

    # Perform QR decomposition on unitless values
    # Result is a list [Q, R]
    result = lax_linalg.qr_p.bind(ustrip(x), **params)

    q_mat, r_mat = result
    # Q is orthonormal (dimensionless), R has input units
    # Return as list to match JAX primitive conventions
    return [
        type_np(x)(q_mat, unit=one),
        type_np(x)(r_mat, unit=u_in),
    ]


@register(lax_linalg.lu_p)
def lu_p_q(x: ABCQ, /, **params: Any) -> Any:
    """LU decomposition of a quantity matrix.

    For a matrix A with units, LU decomposition returns:

    - L: lower triangular with same units as A (diagonal = 1, so overall unit is
      A's unit)
    - U: upper triangular with same units as A
    - Permutation: dimensionless integer array

    Note: JAX's LU returns the LU factorization in a packed format plus pivot indices.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> x = u.Q([[1.0, 2.0], [3.0, 4.0]], "m")
    >>> lu, pivots, permutation = jnp.linalg.lu(x)  # doctest: +SKIP

    """
    u_in = unit_of(x)

    # Perform LU decomposition on the unitless values
    # Result is a list: [LU_packed, pivots, permutation]
    result = lax_linalg.lu_p.bind(ustrip(u_in, x), **params)

    # Unpack the result
    lu_packed, pivots, permutation = result

    # LU_packed contains both L and U in a packed format, preserving input units
    # Pivots and permutation are dimensionless integers
    return [type_np(x)(lu_packed, unit=u_in), pivots, permutation]


# ==============================================================================


@register(lax.device_put_p)
def device_put_p(x: ABCQ, /, **kw: Any) -> ABCQ:
    """Put a quantity on a device.

    Examples
    --------
    >>> from quaxed import device_put
    >>> import unxt as u

    >>> q = u.quantity.BareQuantity(1, "m")
    >>> device_put(q)
    BareQuantity(Array(1, dtype=int32, ...), unit='m')

    >>> q = u.Q(1, "m")
    >>> device_put(q)
    Quantity(Array(1, dtype=int32, ...), unit='m')

    """
    return jt.map(lambda y: lax.device_put_p.bind(y, **kw), x)  # type: ignore[no-untyped-call]


# ==============================================================================


@register(lax.digamma_p)
def digamma_p(x: ABCQ, /) -> ABCQ:
    """Digamma function of a quantity.

    Examples
    --------
    >>> from quaxed import lax
    >>> import unxt as u

    >>> q = u.quantity.BareQuantity(1.0, "")
    >>> lax.digamma(q)
    BareQuantity(Array(-0.5772154, dtype=float32, ...), unit='')

    >>> q = u.Q(1.0, "")
    >>> lax.digamma(q)
    Quantity(Array(-0.5772154, dtype=float32, ...), unit='')

    """
    return replace(x, value=qlax.digamma(ustrip(one, x)))


# ==============================================================================
# Division


@register(lax.div_p)
def div_p_sqsq(x: StaticQuantity, y: StaticQuantity, /) -> StaticQuantity:
    """Division of two StaticQuantities.

    Examples
    --------
    >>> import unxt as u
    >>> import quaxed.lax as qlax

    Integer division for integer dtypes:

    >>> q1 = u.StaticQuantity(6, "m")
    >>> q2 = u.StaticQuantity(2, "s")
    >>> qlax.div(q1, q2)
    StaticQuantity(array(3), unit='m / s')

    True division for float dtypes:

    >>> q1 = u.StaticQuantity(6.0, "m")
    >>> q2 = u.StaticQuantity(2.0, "s")
    >>> qlax.div(q1, q2)
    StaticQuantity(array(3.), unit='m / s')

    """
    u = unit(x.unit / y.unit)
    xv, yv = ustrip(x), ustrip(y)
    xv, yv = promote_dtypes_if_needed((x.dtype, y.dtype), xv, yv)
    # Use lax.div for integer division, Python / for float division
    result = (
        np.floor_divide(xv, yv)
        if jnp.issubdtype(xv.dtype, jnp.integer)
        else np.divide(xv, yv)
    )
    return StaticQuantity(result, unit=u)


@register(lax.div_p)
def div_p_qq(x: ABCQ, y: ABCQ, /) -> ABCQ:
    """Division of two quantities.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> q1 = u.quantity.BareQuantity(1, "m")
    >>> q2 = u.quantity.BareQuantity(2, "s")
    >>> jnp.divide(q1, q2)
    BareQuantity(Array(0.5, dtype=float32, ...), unit='m / s')
    >>> q1 / q2
    BareQuantity(Array(0.5, dtype=float32, ...), unit='m / s')

    >>> q1 = u.Q(1, "m")
    >>> q2 = u.Q(2, "s")
    >>> jnp.divide(q1, q2)
    Quantity(Array(0.5, dtype=float32, ...), unit='m / s')
    >>> q1 / q2
    Quantity(Array(0.5, dtype=float32, ...), unit='m / s')

    """
    x, y = promote(x, y)
    u = unit(x.unit / y.unit)
    xv, yv = ustrip(x), ustrip(y)
    xv, yv = promote_dtypes_if_needed((x.dtype, y.dtype), xv, yv)
    return type_np(x)(qlax.div(xv, yv), unit=u)


@register(lax.div_p)
def div_p_vq(x: ArrayLike, y: ABCQ, /) -> ABCQ:
    """Division of an array by a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> x = jnp.asarray([1.0, 2, 3])

    >>> q = u.quantity.BareQuantity(2.0, "m")
    >>> jnp.divide(x, q)
    BareQuantity(Array([0.5, 1. , 1.5], dtype=float32), unit='1 / m')
    >>> x / q
    BareQuantity(Array([0.5, 1. , 1.5], dtype=float32), unit='1 / m')

    >>> q = u.Q(2.0, "m")
    >>> jnp.divide(x, q)
    Quantity(Array([0.5, 1. , 1.5], dtype=float32), unit='1 / m')
    >>> x / q
    Quantity(Array([0.5, 1. , 1.5], dtype=float32), unit='1 / m')

    """
    u = (1 / y.unit).unit  # TODO: better construction of the unit
    return type_np(y)(lax.div(x, ustrip(y)), unit=u)


@register(lax.div_p)
def div_p_qv(x: ABCQ, y: ArrayLike, /) -> ABCQ:
    """Division of a quantity by an array.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> y = jnp.asarray([1, 2, 3])

    >>> q = u.quantity.BareQuantity(6.0, "m")
    >>> jnp.divide(q, y)
    BareQuantity(Array([6., 3., 2.], dtype=float32, ...), unit='m')
    >>> q / y
    BareQuantity(Array([6., 3., 2.], dtype=float32, ...), unit='m')

    >>> q = u.Q(6.0, "m")
    >>> jnp.divide(q, y)
    Quantity(Array([6., 3., 2.], dtype=float32, ...), unit='m')
    >>> q / y
    Quantity(Array([6., 3., 2.], dtype=float32, ...), unit='m')

    """
    return replace(x, value=ustrip(x) / y)


# TODO: can this be done with promotion/conversion/default rule instead?
@register(lax.div_p)
def div_p_a(x: AbstractAngle, y: AbstractAngle, /) -> ABCQ:
    """Division of a Quantity by an Angle.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> angle = u.Angle(1, "deg")
    >>> q = u.Q(2, "km")
    >>> jnp.divide(q, angle)
    Quantity(Array(2., dtype=float32, ...), unit='km / deg')

    """
    x, y = promote(x, y)
    return div_p_qq(convert(x, Quantity), convert(y, Quantity))


# ==============================================================================


@register(lax.dot_general_p)
def dot_general_jq(lhs: ArrayLike, rhs: ABCQ, /, **kw: Any) -> ABCQ:
    """Dot product of an array and a quantity.

    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> theta = jnp.pi / 4  # 45 degrees
    >>> Rz = jnp.asarray(
    ...     [
    ...         [jnp.cos(theta), -jnp.sin(theta), 0],
    ...         [jnp.sin(theta), jnp.cos(theta), 0],
    ...         [0, 0, 1],
    ...     ]
    ... )

    >>> q = u.quantity.BareQuantity([1, 0, 0], "m")
    >>> jnp.linalg.matmul(Rz, q)
    BareQuantity(Array([0.70710677, 0.70710677, 0. ], dtype=float32), unit='m')
    >>> Rz @ q
    BareQuantity(Array([0.70710677, 0.70710677, 0. ], dtype=float32), unit='m')

    >>> q = u.Q([1, 0, 0], "m")
    >>> jnp.linalg.matmul(Rz, q)
    Quantity(Array([0.70710677, 0.70710677, 0. ], dtype=float32), unit='m')
    >>> Rz @ q
    Quantity(Array([0.70710677, 0.70710677, 0. ], dtype=float32), unit='m')

    """
    return type_np(rhs)(lax.dot_general_p.bind(lhs, ustrip(rhs), **kw), unit=rhs.unit)


@register(lax.dot_general_p)
def dot_general_qj(lhs: ABCQ, rhs: ArrayLike, /, **kw: Any) -> ABCQ:
    """Dot product of a quantity and an array.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> theta = jnp.pi / 2  # 90 degrees
    >>> Rz = u.Q(
    ...     [
    ...         [jnp.cos(theta), -jnp.sin(theta), 0],
    ...         [jnp.sin(theta), jnp.cos(theta), 0],
    ...         [0, 0, 1],
    ...     ],
    ...     "km",
    ... )

    >>> q = jnp.asarray([1, 2, 3])
    >>> jnp.linalg.matmul(Rz, q).round(2)
    Quantity(Array([-2.,  1.,  3.], dtype=float32), unit='km')

    """
    return type_np(lhs)(lax.dot_general_p.bind(ustrip(lhs), rhs, **kw), unit=lhs.unit)


@register(lax.dot_general_p)
def dot_general_qq(lhs: ABCQ, rhs: ABCQ, /, **kw: Any) -> ABCQ:
    """Dot product of two quantities.

    Examples
    --------
    This is a dot product of two quantities.

    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> q1 = u.quantity.BareQuantity([1, 2, 3], "m")
    >>> q2 = u.quantity.BareQuantity([4, 5, 6], "m")
    >>> jnp.vecdot(q1, q2)
    BareQuantity(Array(32, dtype=int32), unit='m2')
    >>> q1 @ q2
    BareQuantity(Array(32, dtype=int32), unit='m2')

    >>> q1 = u.Q([1, 2, 3], "m")
    >>> q2 = u.Q([4, 5, 6], "m")
    >>> jnp.vecdot(q1, q2)
    Quantity(Array(32, dtype=int32), unit='m2')
    >>> q1 @ q2
    Quantity(Array(32, dtype=int32), unit='m2')

    This rule is also used by `jnp.matmul` for quantities.

    >>> Rz = jnp.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    >>> q = u.Q([1, 0, 0], "m")
    >>> Rz @ q
    Quantity(Array([0, 1, 0], dtype=int32), unit='m')

    This uses `matmul` for quantities.

    >>> jnp.linalg.matmul(Rz, q)
    Quantity(Array([0, 1, 0], dtype=int32), unit='m')

    """
    lhs, rhs = promote(lhs, rhs)
    return type_np(lhs)(
        lax.dot_general_p.bind(ustrip(lhs), ustrip(rhs), **kw),
        unit=lhs.unit * rhs.unit,
    )


@register(lax.dot_general_p)
def dot_general_abstractangle_abstractangle(
    lhs: AbstractAngle, rhs: AbstractAngle, /, **kwargs: Any
) -> Quantity:
    """Dot product of two Angles.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> q1 = u.Angle([1, 2, 3], "deg")
    >>> q2 = u.Angle([4, 5, 6], "deg")
    >>> jnp.vecdot(q1, q2)
    Quantity(Array(32, dtype=int32), unit='deg2')

    >>> q1 @ q2
    Quantity(Array(32, dtype=int32), unit='deg2')

    """
    value = lax.dot_general_p.bind(lhs.value, rhs.value, **kwargs)
    return Quantity(value, unit=lhs.unit * rhs.unit)


# ==============================================================================


@register(lax.dynamic_slice_p)
def dynamic_slice_q(operand: ABCQ, /, *indices: ArrayLike, **kw: Any) -> ABCQ:
    """Dynamic slice of a quantity.

    Examples
    --------
    >>> from quaxed import lax
    >>> import unxt as u

    >>> q = u.Q([1, 2, 3, 4, 5], "m")
    >>> lax.dynamic_slice(q, (1,), (3,))
    Quantity(Array([2, 3, 4], dtype=int32), unit='m')

    """
    return replace(
        operand, value=lax.dynamic_slice_p.bind(ustrip(operand), *indices, **kw)
    )


# ==============================================================================


@register(lax.dynamic_update_slice_p)
def dynamic_update_slice_p(
    operand: ABCQ,
    update: ABCQ,
    /,
    *indices: ArrayLike,
    **kw: Any,
) -> ABCQ:
    """Dynamic update slice of a quantity.

    Examples
    --------
    >>> from quaxed import lax
    >>> import unxt as u

    >>> q = u.Q([1, 2, 3, 4, 5], "m")
    >>> update = u.Q([6, 7], "m")
    >>> lax.dynamic_update_slice(q, update, (1,))
    Quantity(Array([1, 6, 7, 4, 5], dtype=int32), unit='m')

    """
    return replace(
        operand,
        value=lax.dynamic_update_slice_p.bind(
            ustrip(operand), ustrip(update), *indices, **kw
        ),
    )


# ==============================================================================


@register(lax.linalg.eigh_p)
def eigh_p(x: ABCQ, /, **kw: Any) -> tuple[Array, ABCQ]:
    """Eigenvalues and eigenvectors of a Hermitian matrix quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> q = u.Q([[2, -1], [1, 3]], "eV")
    >>> w, v = jnp.linalg.eigh(q)
    >>> w
    Quantity(Array([2., 3.], dtype=float32), unit='eV')
    >>> v
    Array([[1., 0.],
           [0., 1.]], dtype=float32)

    """
    v, w = lax.linalg.eigh_p.bind(ustrip(x), **kw)
    return v, replace(x, value=w)


# ==============================================================================


@register(lax.eq_p)
def eq_p_qq(x: ABCQ, y: ABCQ) -> ArrayLike:
    """Equality of two quantities.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> q1 = u.quantity.BareQuantity(1, "m")
    >>> q2 = u.quantity.BareQuantity(1, "m")
    >>> jnp.equal(q1, q2)
    Array(True, dtype=bool, ...)
    >>> q1 == q2
    Array(True, dtype=bool, ...)

    >>> q1 = u.Q(1, "m")
    >>> q2 = u.Q(1, "m")
    >>> jnp.equal(q1, q2)
    Array(True, dtype=bool, ...)
    >>> q1 == q2
    Array(True, dtype=bool, ...)

    """
    xv = ustrip(x)
    xv = eqx.error_if(  # TODO: customize Exception type
        xv,
        not is_unit_convertible(x.unit, y.unit),
        f"Cannot compare Q(x, {x.unit}) == Q(y, {y.unit}).",
    )
    return qlax.eq(xv, ustrip(x.unit, y))  # re-dispatch on the values


@register(lax.eq_p)
def eq_p_vq(x: ArrayLike, y: ABCQ, /) -> ArrayLike:
    """Equality of an array and a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> x = jnp.asarray([1.0, 2, 3])

    >>> q = u.quantity.BareQuantity(2.0, "")
    >>> jnp.equal(x, q)
    Array([False,  True, False], dtype=bool)

    >>> q = u.Q(2.0, "")
    >>> jnp.equal(x, q)
    Array([False,  True, False], dtype=bool)

    >>> q = u.Q(2.0, "m")
    >>> try:
    ...     jnp.equal(x, q)
    ... except Exception as e:
    ...     print("can't compare")
    can't compare

    """
    yv = ustrip(y)
    yv = eqx.error_if(  # TODO: customize Exception type
        yv,
        not is_unit_convertible(one, y.unit) and jnp.logical_not(jnp.all(x == 0)),
        f"Cannot compare x == Q(y, {y.unit}) (except for x=0).",
    )
    return qlax.eq(x, yv)  # re-dispatch on the value


@register(lax.eq_p)
def eq_p_aqv(x: ABCQ, y: ArrayLike, /) -> ArrayLike:
    """Equality of an array and a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> y = jnp.asarray([1.0, 2, 3])

    >>> q = u.quantity.BareQuantity(2.0, "")
    >>> jnp.equal(q, y)
    Array([False,  True, False], dtype=bool)

    >>> q = u.quantity.BareQuantity([3.0, 2, 1], "")
    >>> jnp.equal(q, y)
    Array([False,  True, False], dtype=bool)

    >>> q = u.quantity.BareQuantity([3.0, 2, 1], "m")
    >>> try:
    ...     jnp.equal(q, y)
    ... except Exception as e:
    ...     print("can't compare")
    can't compare

    >>> q = u.Q(2.0, "")
    >>> jnp.equal(q, y)
    Array([False,  True, False], dtype=bool)

    >>> q = u.Q([3.0, 2, 1], "")
    >>> jnp.equal(q, y)
    Array([False,  True, False], dtype=bool)

    >>> q = u.Q([3.0, 2, 1], "m")
    >>> try:
    ...     jnp.equal(q, y)
    ... except Exception as e:
    ...     print("can't compare")
    can't compare

    Check against the special cases:

    >>> q == 0
    Array([False, False, False], dtype=bool)

    >>> q == jnp.inf
    Array([False, False, False], dtype=bool)

    """
    special_vals = jnp.logical_or(jnp.all(y == 0), jnp.all(jnp.isinf(y)))
    xv = ustrip(x)
    xv = eqx.error_if(  # TODO: customize Exception type
        xv,
        not is_unit_convertible(one, x.unit) and jnp.logical_not(special_vals),
        f"Cannot compare Q(x, {x.unit}) == y (except for y=0,infinity).",
    )
    return qlax.eq(xv, y)  # re-dispatch on the value


# ==============================================================================


@register(lax.erf_inv_p)
def erf_inv_p(x: ABCQ) -> ABCQ:
    """Inverse error function of a quantity.

    Examples
    --------
    >>> from quaxed import lax
    >>> import unxt as u

    >>> q = u.quantity.BareQuantity(0.5, "")
    >>> lax.erf_inv(q)
    BareQuantity(Array(0.47693628, dtype=float32, ...), unit='')

    >>> q = u.Q(0.5, "")
    >>> lax.erf_inv(q)
    Quantity(Array(0.47693628, dtype=float32, ...), unit='')

    """
    # TODO: can this support non-dimensionless quantities?
    return replace(x, value=qlax.erf_inv(ustrip(one, x)))


# ==============================================================================


@register(lax.erf_p)
def erf_p(x: ABCQ) -> ABCQ:
    """Error function of a quantity.

    Examples
    --------
    >>> from quaxed import lax
    >>> import unxt as u

    >>> q = u.quantity.BareQuantity(0.5, "")
    >>> lax.erf(q)
    BareQuantity(Array(0.5204999, dtype=float32, ...), unit='')

    >>> q = u.Q(0.5, "")
    >>> lax.erf(q)
    Quantity(Array(0.5204999, dtype=float32, ...), unit='')

    """
    # TODO: can this support non-dimensionless quantities?
    return replace(x, value=qlax.erf(ustrip(one, x)))


# ==============================================================================


@register(lax.erfc_p)
def erfc_p(x: ABCQ) -> ABCQ:
    """Complementary error function of a quantity.

    Examples
    --------
    >>> from quaxed import lax
    >>> import unxt as u

    >>> q = u.quantity.BareQuantity(0.5, "")
    >>> lax.erfc(q)
    BareQuantity(Array(0.47950017, dtype=float32, ...), unit='')

    >>> q = u.Q(0.5, "")
    >>> lax.erfc(q)
    Quantity(Array(0.47950017, dtype=float32, ...), unit='')

    """
    # TODO: can this support non-dimensionless quantities?
    return replace(x, value=qlax.erfc(ustrip(one, x)))


# ==============================================================================


@register(lax.exp2_p)
def exp2_p(x: ABCQ, /, **kw: Any) -> ABCQ:
    """2^x of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> q = u.quantity.BareQuantity(3, "")
    >>> jnp.exp2(q)
    BareQuantity(Array(8., dtype=float32, ...), unit='')

    >>> q = u.Q(3, "")
    >>> jnp.exp2(q)
    Quantity(Array(8., dtype=float32, ...), unit='')

    """
    return replace(x, value=lax.exp2_p.bind(ustrip(one, x), **kw))


# ==============================================================================


@register(lax.exp_p)
def exp_p(x: ABCQ, /, **kw: Any) -> ABCQ:
    """Exponential of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> q = u.quantity.BareQuantity(1, "")
    >>> jnp.exp(q)
    BareQuantity(Array(2.7182817, dtype=float32, ...), unit='')

    >>> q = u.Q(1, "")
    >>> jnp.exp(q)
    Quantity(Array(2.7182817, dtype=float32, ...), unit='')

    Euler's crown jewel:

    >>> jnp.exp(u.Q(jnp.pi * 1j, "")) + 1
    Quantity(Array(0.-8.742278e-08j, dtype=complex64, ...), unit='')

    Pretty close to zero!

    """
    # TODO: more meaningful error message.
    return replace(x, value=lax.exp_p.bind(ustrip(one, x), **kw))


# ==============================================================================


@register(lax.expm1_p)
def expm1_p(x: ABCQ, /, **kw: Any) -> ABCQ:
    """Exponential of a quantity minus 1.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> q = u.quantity.BareQuantity(0, "")
    >>> jnp.expm1(q)
    BareQuantity(Array(0., dtype=float32, ...), unit='')

    >>> q = u.Q(0, "")
    >>> jnp.expm1(q)
    Quantity(Array(0., dtype=float32, ...), unit='')

    """
    return replace(x, value=lax.expm1_p.bind(ustrip(one, x), **kw))


# ==============================================================================


@register(lax.fft_p)
def fft_p(x: ABCQ, *, fft_type: Any, fft_lengths: Any) -> ABCQ:
    """Fast Fourier transform of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> q = u.quantity.BareQuantity([1, 2, 3], "")
    >>> jnp.fft.fft(q)
    BareQuantity(Array([ 6. +0.j       , -1.5+0.8660254j, -1.5-0.8660254j],
                       dtype=complex64), unit='')

    >>> q = u.Q([1, 2, 3], "")
    >>> jnp.fft.fft(q)
    Quantity(Array([ 6. +0.j       , -1.5+0.8660254j, -1.5-0.8660254j],
                                    dtype=complex64), unit='')

    """
    return type_np(x)(qlax.fft(ustrip(x), fft_type, fft_lengths), unit=x.unit**-1)


# ==============================================================================


@register(lax.floor_p)
def floor_p(x: ABCQ) -> ABCQ:
    """Floor of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> q = u.quantity.BareQuantity(1.5, "")
    >>> jnp.floor(q)
    BareQuantity(Array(1., dtype=float32, ...), unit='')

    >>> q = u.Q(1.5, "")
    >>> jnp.floor(q)
    Quantity(Array(1., dtype=float32, ...), unit='')

    """
    return replace(x, value=qlax.floor(ustrip(x)))


# ==============================================================================


# used in `jnp.cross`
@register(lax.gather_p)
def gather_p(operand: ABCQ, start_indices: ArrayLike, /, **kw: Any) -> ABCQ:
    # TODO: examples
    return replace(
        operand, value=lax.gather_p.bind(ustrip(operand), start_indices, **kw)
    )


# ==============================================================================


@register(lax.ge_p)
def ge_p_qq(x: ABCQ, y: ABCQ) -> ArrayLike:
    """Greater than or equal to of two quantities.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> q1 = u.quantity.BareQuantity(1_001.0, "m")
    >>> q2 = u.quantity.BareQuantity(1.0, "km")
    >>> jnp.greater_equal(q1, q2)
    Array(True, dtype=bool, ...)
    >>> q1 >= q2
    Array(True, dtype=bool, ...)

    >>> q1 = u.Q(1_001.0, "m")
    >>> q2 = u.Q(1.0, "km")
    >>> jnp.greater_equal(q1, q2)
    Array(True, dtype=bool, ...)
    >>> q1 >= q2
    Array(True, dtype=bool, ...)

    """
    xv = ustrip(x)
    xv = eqx.error_if(  # TODO: customize Exception type
        xv,
        not is_unit_convertible(x.unit, y.unit),
        f"Cannot compare Q(x, {x.unit}) >= Q(y, {y.unit}).",
    )
    return qlax.ge(xv, ustrip(x.unit, y))  # re-dispatch on the values


@register(lax.ge_p)
def ge_p_vq(x: ArrayLike, y: ABCQ, /) -> ArrayLike:
    """Greater than or equal to of an array and a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> x = jnp.asarray(1_001.0)

    >>> q2 = u.quantity.BareQuantity(1.0, "")
    >>> jnp.greater_equal(x, q2)
    Array(True, dtype=bool, ...)

    >>> q2 = u.Q(1.0, "")
    >>> jnp.greater_equal(x, q2)
    Array(True, dtype=bool, ...)

    >>> q2 = u.Q(1.0, "m")
    >>> try:
    ...     jnp.greater_equal(x, q2)
    ... except Exception as e:
    ...     print("can't compare")
    can't compare

    """
    yv = ustrip(y)
    yv = eqx.error_if(  # TODO: customize Exception type
        yv,
        not is_unit_convertible(one, y.unit) and jnp.logical_not(jnp.all(x == 0)),
        f"Cannot compare x >= Q(y, {y.unit}) (except for x=0).",
    )
    return qlax.ge(x, yv)  # re-dispatch on the value


@register(lax.ge_p)
def ge_p_qv(x: ABCQ, y: ArrayLike, /) -> ArrayLike:
    """Greater than or equal to of a quantity and an array.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> y = jnp.asarray(0.9)

    >>> q1 = u.quantity.BareQuantity(1.0, "")
    >>> jnp.greater_equal(q1, y)
    Array(True, dtype=bool, ...)

    >>> q1 = u.Q(1.0, "")
    >>> jnp.greater_equal(q1, y)
    Array(True, dtype=bool, ...)

    >>> q1 = u.Q(1.0, "m")
    >>> try:
    ...     jnp.greater_equal(q1, y)
    ... except Exception as e:
    ...     print("can't compare")
    can't compare

    """
    xv = ustrip(x)
    xv = eqx.error_if(  # TODO: customize Exception type
        xv,
        not is_unit_convertible(one, x.unit) and jnp.logical_not(jnp.all(y == 0)),
        f"Cannot compare Q(x, {x.unit}) >= y (except for y=0).",
    )
    return qlax.ge(xv, y)  # re-dispatch on the value


# ==============================================================================


@register(lax.gt_p)
def gt_p_qq(x: ABCQ, y: ABCQ) -> ArrayLike:
    """Greater than of two quantities.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> q1 = u.quantity.BareQuantity(1_001.0, "m")
    >>> q2 = u.quantity.BareQuantity(1.0, "km")
    >>> jnp.greater_equal(q1, q2)
    Array(True, dtype=bool, ...)

    >>> q1 = u.Q(1_001.0, "m")
    >>> q2 = u.Q(1.0, "km")
    >>> jnp.greater_equal(q1, q2)
    Array(True, dtype=bool, ...)

    """
    xv = ustrip(x)
    xv = eqx.error_if(  # TODO: customize Exception type
        xv,
        not is_unit_convertible(x.unit, y.unit),
        f"Cannot compare Q(x, {x.unit}) > Q(y, {y.unit}).",
    )
    yv = ustrip(x.unit, y)
    xv, yv = promote_dtypes_if_needed((x.dtype, y.dtype), xv, yv)
    return qlax.gt(xv, yv)  # re-dispatch on the values


@register(lax.gt_p)
def gt_p_vq(x: ArrayLike, y: ABCQ) -> ArrayLike:
    """Greater than of an array and a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> x = jnp.asarray(1_001.0)

    >>> q2 = u.quantity.BareQuantity(1.0, "")
    >>> jnp.greater_equal(x, q2)
    Array(True, dtype=bool, ...)

    >>> q2 = u.Q(1.0, "")
    >>> jnp.greater_equal(x, q2)
    Array(True, dtype=bool, ...)

    >>> q2 = u.Q(1.0, "m")
    >>> try:
    ...     jnp.greater_equal(x, q2)
    ... except Exception as e:
    ...     print("can't compare")
    can't compare

    """
    yv = ustrip(y)
    yv = eqx.error_if(  # TODO: customize Exception type
        yv,
        not is_unit_convertible(one, y.unit) and jnp.logical_not(jnp.all(x == 0)),
        f"Cannot compare x > Q(y, {y.unit}) (except for x=0).",
    )
    return qlax.gt(x, yv)  # re-dispatch on the value


@register(lax.gt_p)
def gt_p_qv(x: ABCQ, y: ArrayLike) -> ArrayLike:
    """Greater than comparison between a quantity and an array.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> y = jnp.asarray(0.9)

    >>> q1 = u.quantity.BareQuantity(1.0, "")
    >>> jnp.greater_equal(q1, y)
    Array(True, dtype=bool, ...)

    >>> q1 = u.Q(1.0, "")
    >>> jnp.greater_equal(q1, y)
    Array(True, dtype=bool, ...)

    >>> q1 = u.Q(1.0, "m")
    >>> try:
    ...     jnp.greater_equal(q1, y)
    ... except Exception as e:
    ...     print("can't compare")
    can't compare

    """
    xv = ustrip(x)
    xv = eqx.error_if(  # TODO: customize Exception type
        xv,
        not is_unit_convertible(one, x.unit) and jnp.logical_not(jnp.all(y == 0)),
        f"Cannot compare Q(x, {x.unit}) > y (except for y=0).",
    )
    return qlax.gt(xv, y)  # re-dispatch on the value


# ==============================================================================


@register(lax.igamma_p)
def igamma_p(a: float | int | ABCQ, x: ABCQ) -> ABCQ:
    """Regularized incomplete gamma function of a and x.

    Examples
    --------
    >>> from quaxed import lax
    >>> import unxt as u


    >>> x = u.quantity.BareQuantity(1.0, "")
    >>> lax.igamma(1.0, x)
    BareQuantity(Array(0.6321202, dtype=float32, ...), unit='')

    >>> a = u.quantity.BareQuantity(1.0, "")
    >>> lax.igamma(a, x)
    BareQuantity(Array(0.6321202, dtype=float32, ...), unit='')

    >>> a = u.Q(1.0, "")
    >>> x = u.Q(1.0, "")
    >>> lax.igamma(a, x)
    Quantity(Array(0.6321202, dtype=float32, ...), unit='')

    """
    return replace(x, value=qlax.igamma(ustrip(AllowValue, one, a), ustrip(one, x)))


# ==============================================================================


@register(lax.igammac_p)
def igammac_p(a: float | int | ABCQ, x: ABCQ) -> ABCQ:
    """Regularized upper incomplete gamma function of a and x.

    Examples
    --------
    >>> import quaxed.lax as qlax


    >>> x = u.quantity.BareQuantity(1.0, "")
    >>> qlax.igammac(1.0, x)
    BareQuantity(Array(0.36787927, dtype=float32, ...), unit='')

    >>> a = u.quantity.BareQuantity(1.0, "")
    >>> qlax.igammac(a, x)
    BareQuantity(Array(0.36787927, dtype=float32, ...), unit='')

    >>> a = u.Q(1.0, "")
    >>> x = u.Q(1.0, "")
    >>> qlax.igammac(a, x)
    Quantity(Array(0.36787927, dtype=float32, ...), unit='')

    """
    return replace(x, value=qlax.igammac(ustrip(AllowValue, one, a), ustrip(one, x)))


# ==============================================================================


@register(lax.imag_p)
def imag_p(x: ABCQ) -> ABCQ:
    return replace(x, value=qlax.imag(ustrip(x)))


# ==============================================================================


@register(lax.integer_pow_p)
def integer_pow_p(x: ABCQ, *, y: Any) -> ABCQ:
    """Integer power of a quantity.

    Examples
    --------
    >>> import unxt as u
    >>> q = u.quantity.BareQuantity(2, "m")
    >>> q**3
    BareQuantity(Array(8, dtype=int32, ...), unit='m3')

    >>> q = u.Q(2, "m")
    >>> q**3
    Quantity(Array(8, dtype=int32, ...), unit='m3')

    """
    return type_np(x)(value=qlax.integer_pow(ustrip(x), y), unit=x.unit**y)


@register(lax.integer_pow_p)
def integer_pow_p_abstractangle(x: AbstractAngle, /, *, y: Any) -> ABCQ:
    """Integer power of an Angle.

    Examples
    --------
    >>> import unxt as u
    >>> q = u.Angle(2, "deg")

    >>> q**3
    Quantity(Array(8, dtype=int32, ...), unit='deg3')

    """
    return integer_pow_p(convert(x, Quantity), y=y)


# ==============================================================================


@register(lax.is_finite_p)
def is_finite_p(x: ABCQ) -> ArrayLike:
    """Check if a quantity is finite.

    Examples
    --------
    .. invisible-code-block: python

        import jax
        from packaging.version import Version
        jax_version_gte_0_8 = Version(jax.__version__) >= Version("0.8.0")

    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> q = u.quantity.BareQuantity(1, "m")

    .. skip: next if(jax_version_gte_0_8, reason="jax >= 0.8 returns TypedNdArray")

    >>> jnp.isfinite(q)
    array(True)

    >>> q = u.quantity.BareQuantity(float("inf"), "m")
    >>> jnp.isfinite(q)
    Array(False, dtype=bool, ...)

    >>> q = u.Q(1, "m")

    .. skip: next if(jax_version_gte_0_8, reason="jax >= 0.8 returns TypedNdArray")

    >>> jnp.isfinite(q)
    array(True)

    >>> q = u.Q(float("inf"), "m")
    >>> jnp.isfinite(q)
    Array(False, dtype=bool, ...)

    """
    return lax.is_finite(ustrip(x))


# ==============================================================================


@register(lax.le_p)
def le_p_qq(x: ABCQ, y: ABCQ, /) -> ArrayLike:
    """Less than or equal to of two quantities.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> q1 = u.quantity.BareQuantity(1_001.0, "m")
    >>> q2 = u.quantity.BareQuantity(1.0, "km")
    >>> jnp.less_equal(q1, q2)
    Array(False, dtype=bool, ...)

    >>> q1 = u.Q(1_001.0, "m")
    >>> q2 = u.Q(1.0, "km")
    >>> jnp.less_equal(q1, q2)
    Array(False, dtype=bool, ...)

    """
    xv = ustrip(x)
    xv = eqx.error_if(  # TODO: customize Exception type
        xv,
        not is_unit_convertible(x.unit, y.unit),
        f"Cannot compare Q(x, {x.unit}) <= Q(y, {y.unit}).",
    )
    return qlax.le(xv, ustrip(x.unit, y))  # re-dispatch on the values


@register(lax.le_p)
def le_p_vq(x: ArrayLike, y: ABCQ, /) -> ArrayLike:
    """Less than or equal to of an array and a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> x1 = jnp.asarray(1.001)

    >>> q2 = u.quantity.BareQuantity(1.0, "")
    >>> jnp.less_equal(x1, q2)
    Array(False, dtype=bool, ...)

    >>> q2 = u.Q(1.0, "")
    >>> jnp.less_equal(x1, q2)
    Array(False, dtype=bool, ...)

    >>> q2 = u.Q(1.0, "m")
    >>> try:
    ...     jnp.less_equal(x1, q2)
    ... except Exception as e:
    ...     print("can't compare")
    can't compare

    """
    yv = ustrip(y)
    yv = eqx.error_if(  # TODO: customize Exception type
        yv,
        not is_unit_convertible(one, y.unit) and jnp.logical_not(jnp.all(x == 0)),
        f"Cannot compare x <= Q(y, {y.unit}) (except for x=0).",
    )
    return qlax.le(x, yv)  # re-dispatch on the value


@register(lax.le_p)
def le_p_qv(x: ABCQ, y: ArrayLike, /) -> ArrayLike:
    """Less than or equal to of a quantity and an array.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> y1 = jnp.asarray(0.9)

    >>> q1 = u.quantity.BareQuantity(1.0, "")
    >>> jnp.less_equal(q1, y1)
    Array(False, dtype=bool, ...)

    >>> q1 = u.Q(1.0, "")
    >>> jnp.less_equal(q1, y1)
    Array(False, dtype=bool, ...)

    >>> q1 = u.Q(1.0, "m")
    >>> try:
    ...     jnp.less_equal(q1, y1)
    ... except Exception as e:
    ...     print("can't compare")
    can't compare

    """
    xv = ustrip(x)
    xv = eqx.error_if(  # TODO: customize Exception type
        xv,
        not is_unit_convertible(one, x.unit) and jnp.logical_not(jnp.all(y == 0)),
        f"Cannot compare Q(x, {x.unit}) <= y (except for y=0).",
    )
    return qlax.le(xv, y)  # re-dispatch on the value


# ==============================================================================


@register(lax.lgamma_p)
def lgamma_p(x: ABCQ) -> ABCQ:
    """Log-gamma function of a quantity.

    Examples
    --------
    >>> import quaxed.scipy as jsp

    >>> q = u.quantity.BareQuantity(3, "")
    >>> jsp.special.gammaln(q)
    BareQuantity(Array(0.6931474, dtype=float32, ...), unit='')

    >>> q = u.Q(3, "")
    >>> jsp.special.gammaln(q)
    Quantity(Array(0.6931474, dtype=float32, ...), unit='')

    """
    # TODO: are there any units that this can support?
    return replace(x, value=qlax.lgamma(ustrip(one, x)))


# ==============================================================================


@register(lax.log1p_p)
def log1p_p(x: ABCQ, /, **kw: Any) -> ABCQ:
    """Logarithm of 1 plus a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> q = u.quantity.BareQuantity(-1, "")
    >>> jnp.log1p(q)
    BareQuantity(Array(-inf, dtype=float32, ...), unit='')

    >>> q = u.Q(-1, "")
    >>> jnp.log1p(q)
    Quantity(Array(-inf, dtype=float32, weak_type=True), unit='')

    """
    return replace(x, value=lax.log1p_p.bind(ustrip(one, x), **kw))


# ==============================================================================


@register(lax.log_p)
def log_p(x: ABCQ, /, **kw: Any) -> ABCQ:
    """Logarithm of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> q = u.quantity.BareQuantity(1, "")
    >>> jnp.log(q)
    BareQuantity(Array(0., dtype=float32, ...), unit='')

    >>> q = u.Q(1, "")
    >>> jnp.log(q)
    Quantity(Array(0., dtype=float32, weak_type=True), unit='')

    """
    return replace(x, value=lax.log_p.bind(ustrip(one, x), **kw))


# ==============================================================================


@register(lax.logistic_p)
def logistic_p(x: ABCQ, /, **kw: Any) -> ABCQ:
    """Logarithm of a quantity.

    Examples
    --------
    >>> import quaxed.lax as qlax
    >>> import unxt as u

    >>> q = u.quantity.BareQuantity(1.0, "")
    >>> qlax.logistic(q)
    BareQuantity(Array(0.7310586, dtype=float32, ...), unit='')

    >>> q = u.Q(1.0, "")
    >>> qlax.logistic(q)
    Quantity(Array(0.7310586, dtype=float32, ...), unit='')

    """
    return replace(x, value=lax.logistic_p.bind(ustrip(one, x), **kw))


# ==============================================================================


@register(lax.lt_p)
def lt_p_qq(x: ABCQ, y: ABCQ, /) -> ArrayLike:
    """Less than of two quantities.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    `BareQuantity`:

    >>> x = u.quantity.BareQuantity(1.0, "km")
    >>> y = u.quantity.BareQuantity(2000.0, "m")
    >>> x < y
    Array(True, dtype=bool, ...)

    >>> jnp.less(x, y)
    Array(True, dtype=bool, ...)

    >>> x = u.quantity.BareQuantity([1.0, 2, 3], "km")
    >>> x < y
    Array([ True, False, False], dtype=bool)

    >>> jnp.less(x, y)
    Array([ True, False, False], dtype=bool)

    `Quantity`:

    >>> x = u.Q(1.0, "km")
    >>> y = u.Q(2000.0, "m")
    >>> x < y
    Array(True, dtype=bool, ...)

    >>> jnp.less(x, y)
    Array(True, dtype=bool, ...)

    >>> x = u.Q([1.0, 2, 3], "km")
    >>> x < y
    Array([ True, False, False], dtype=bool)

    >>> jnp.less(x, y)
    Array([ True, False, False], dtype=bool)

    """
    # Check if the units are convertible.
    xv = ustrip(x)
    xv = eqx.error_if(  # TODO: customize Exception type
        xv,
        not is_unit_convertible(x.unit, y.unit),
        f"Cannot compare Q(x, {x.unit}) < Q(y, {y.unit}).",
    )
    # Strip the units to compare the values.
    yv = ustrip(x.unit, y)  # this can change the dtype
    xv, yv = promote_dtypes_if_needed((x.dtype, y.dtype), xv, yv)

    return qlax.lt(xv, yv)  # re-dispatch on the values


@register(lax.lt_p)
def lt_p_vq(x: ArrayLike, y: ABCQ, /) -> ArrayLike:
    """Less than of an array and a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    `BareQuantity`:

    >>> x = jnp.asarray([1.0])
    >>> y = u.quantity.BareQuantity(2.0, "")

    Note that `JAX` does support passing the comparison to
    a different class.

    >>> x < y
    Array([ True], dtype=bool)

    But we can always use the `jnp.less` function.

    >>> jnp.less(x, y)
    Array([ True], dtype=bool)

    >>> x = jnp.asarray([1.0, 2, 3])
    >>> jnp.less(x, y)
    Array([ True, False, False], dtype=bool)

    `Quantity`:

    >>> y = u.Q(2.0, "")
    >>> jnp.less(x, y)
    Array([ True, False, False], dtype=bool)

    >>> x = jnp.asarray([1.0, 2, 3])
    >>> jnp.less(x, y)
    Array([ True, False, False], dtype=bool)

    >>> y = u.Q(2.0, "m")
    >>> try:
    ...     jnp.less(x, y)
    ... except Exception as e:
    ...     print("can't compare")
    can't compare

    """
    yv = ustrip(y)
    yv = eqx.error_if(  # TODO: customize Exception type
        yv,
        not is_unit_convertible(one, y.unit) and jnp.logical_not(jnp.all(x == 0)),
        f"Cannot compare x < Q(y, {y.unit}) (except for x=0).",
    )
    return qlax.lt(x, yv)  # re-dispatch on the value


@register(lax.lt_p)
def lt_p_qv(x: ABCQ, y: ArrayLike, /) -> ArrayLike:
    """Compare a unitless Quantity to a value.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    `BareQuantity`:

    >>> x = u.quantity.BareQuantity(1, "")
    >>> y = 2
    >>> x < y
    Array(True, dtype=bool, ...)

    >>> jnp.less(x, y)
    Array(True, dtype=bool, ...)

    >>> x = u.quantity.BareQuantity([1, 2, 3], "")
    >>> x < y
    Array([ True, False, False], dtype=bool)

    >>> jnp.less(x, y)
    Array([ True, False, False], dtype=bool)

    `Quantity`:

    >>> x = u.Q(1, "")
    >>> y = 2
    >>> x < y
    Array(True, dtype=bool, ...)

    >>> jnp.less(x, y)
    Array(True, dtype=bool, ...)

    >>> x = u.Q([1, 2, 3], "")
    >>> x < y
    Array([ True, False, False], dtype=bool)

    >>> jnp.less(x, y)
    Array([ True, False, False], dtype=bool)

    >>> x = u.Q([1, 2], "m")
    >>> try:
    ...     jnp.less(x, y)
    ... except Exception as e:
    ...     print("can't compare")
    can't compare

    """
    xv = ustrip(x)
    xv = eqx.error_if(  # TODO: customize Exception type
        xv,
        not is_unit_convertible(one, x.unit) and jnp.logical_not(jnp.all(y == 0)),
        f"Cannot compare Q(x, {x.unit}) < y (except for y=0).",
    )
    return qlax.lt(xv, y)  # re-dispatch on the value


# ==============================================================================


@register(lax.linalg.lu_p)
def lu_p_q(x: ABCQ, /) -> tuple[ABCQ, Int[Array, "..."], Int[Array, "..."]]:
    """LU decomposition of a quantity.

    Examples
    --------
    >>> from quaxed import lax
    >>> import unxt as u

    >>> x = u.Q([[1.0, 2.0], [3.0, 4.0]], "m")
    >>> lu, pivots, permutation = lax.linalg.lu(x)
    >>> lu
    Quantity(Array([[3.        , 4.        ],
                    [0.33333334, 0.6666666 ]], dtype=float32), unit='m')

    >>> pivots
    Array([1, 1], dtype=int32)

    """
    lu, pivots, permutation = lax.linalg.lu_p.bind(ustrip(x))
    return Quantity(lu, unit=x.unit), pivots, permutation


# ==============================================================================


@register(lax.max_p)
def max_p_qq(x: ABCQ, y: ABCQ, /) -> ABCQ:
    """Maximum of two quantities.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> q1 = u.quantity.BareQuantity(1, "m")
    >>> q2 = u.quantity.BareQuantity(2, "m")
    >>> jnp.maximum(q1, q2)
    BareQuantity(Array(2, dtype=int32, ...), unit='m')

    >>> q1 = u.Q(1, "m")
    >>> q2 = u.Q(2, "m")
    >>> jnp.maximum(q1, q2)
    Quantity(Array(2, dtype=int32, ...), unit='m')

    """
    yv = ustrip(x.unit, y)
    return replace(x, value=qlax.max(ustrip(x), yv))


@register(lax.max_p)
def max_p_vq(x: ArrayLike, y: ABCQ, /) -> ABCQ:
    """Maximum of an array and quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> x = jnp.array([1.0])
    >>> q2 = u.quantity.BareQuantity(2, "")
    >>> jnp.maximum(x, q2)
    BareQuantity(Array([2.], dtype=float32), unit='')

    >>> q2 = u.Q(2, "")
    >>> jnp.maximum(x, q2)
    Quantity(Array([2.], dtype=float32), unit='')

    """
    yv = ustrip(one, y)
    return replace(y, value=qlax.max(x, yv))


@register(lax.max_p)
def max_p_qv(x: ABCQ, y: ArrayLike, /) -> ABCQ:
    """Maximum of an array and quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> q1 = u.quantity.BareQuantity(2, "")
    >>> y = jnp.array([1.0])
    >>> jnp.maximum(q1, y)
    BareQuantity(Array([2.], dtype=float32), unit='')

    >>> q1 = u.Q(2, "")
    >>> jnp.maximum(q1, y)
    Quantity(Array([2.], dtype=float32), unit='')

    """
    xv = ustrip(one, x)
    return replace(x, value=qlax.max(xv, y))


# ==============================================================================


@register(lax.min_p)
def min_p_qq(x: ABCQ, y: ABCQ, /) -> ABCQ:
    """Minimum of two quantities.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> q1 = u.quantity.BareQuantity([1, 2, 3], "m")
    >>> q2 = u.quantity.BareQuantity([2, 1, 3], "m")
    >>> jnp.minimum(q1, q2)
    BareQuantity(Array([1, 1, 3], dtype=int32), unit='m')

    >>> q3 = u.Q([1, 2, 3], "m")
    >>> q4 = u.Q([2, 1, 3], "m")
    >>> jnp.minimum(q3, q4)
    Quantity(Array([1, 1, 3], dtype=int32), unit='m')

    >>> jnp.minimum(q1, q4)
    BareQuantity(Array([1, 1, 3], dtype=int32), unit='m')

    >>> jnp.minimum(q3, q2)
    Quantity(Array([1, 1, 3], dtype=int32), unit='m')

    """
    return replace(x, value=qlax.min(ustrip(x), ustrip(x.unit, y)))


@register(lax.min_p)
def min_p_vq(x: ArrayLike, y: ABCQ, /) -> ABCQ:
    """Minimum of an array and quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> x = jnp.array([1, 2, 3])
    >>> q = u.quantity.BareQuantity(2, "")
    >>> jnp.minimum(x, q)
    BareQuantity(Array([1, 2, 2], dtype=int32), unit='')

    >>> q = u.Q(2, "")
    >>> jnp.minimum(x, q)
    Quantity(Array([1, 2, 2], dtype=int32), unit='')

    """
    return replace(y, value=qlax.min(x, ustrip(one, y)))


@register(lax.min_p)
def min_p_qv(x: ABCQ, y: ArrayLike, /) -> ABCQ:
    """Minimum of a quantity and an array.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> q = u.quantity.BareQuantity(2, "")
    >>> x = jnp.array([1, 2, 3])
    >>> jnp.minimum(q, x)
    BareQuantity(Array([1, 2, 2], dtype=int32), unit='')

    >>> q = u.Q(2, "")
    >>> jnp.minimum(q, x)
    Quantity(Array([1, 2, 2], dtype=int32), unit='')

    """
    return replace(x, value=qlax.min(ustrip(one, x), y))


# ==============================================================================
# Multiplication


@register(lax.mul_p)
def mul_p_qq(x: ABCQ, y: ABCQ, /) -> ABCQ:
    """Multiplication of two quantities.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> q1 = u.quantity.BareQuantity(2, "m")
    >>> q2 = u.quantity.BareQuantity(3, "m")
    >>> jnp.multiply(q1, q2)
    BareQuantity(Array(6, dtype=int32, ...), unit='m2')

    >>> q1 = u.Q(2, "m")
    >>> q2 = u.Q(3, "m")
    >>> jnp.multiply(q1, q2)
    Quantity(Array(6, dtype=int32, ...), unit='m2')

    >>> q1 = u.quantity.BareQuantity(2, "m")
    >>> q2 = u.Q(3, "m")
    >>> jnp.multiply(q1, q2)
    Quantity(Array(6, dtype=int32, weak_type=True), unit='m2')

    """
    # Promote to a common type
    x, y = promote(x, y)
    # Multiply the units
    u = unit(x.unit * y.unit)
    # Multiply the values (use * to preserve numpy arrays for StaticQuantity)
    return type_np(x)(ustrip(x) * ustrip(y), unit=u)


@register(lax.mul_p)
def mul_p_vq(x: ArrayLike, y: ABCQ, /) -> ABCQ:
    """Multiplication of an array-like and a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> q = u.quantity.BareQuantity(2, "m")

    >>> 2.0 * q
    BareQuantity(Array(4., dtype=float32, ...), unit='m')

    >>> jnp.asarray(2) * q
    BareQuantity(Array(4, dtype=int32, ...), unit='m')

    >>> jnp.asarray([2, 3]) * q
    BareQuantity(Array([4, 6], dtype=int32), unit='m')

    >>> q = u.Q(2, "m")

    >>> 2.0 * q
    Quantity(Array(4., dtype=float32, ...), unit='m')

    >>> jnp.asarray(2) * q
    Quantity(Array(4, dtype=int32, ...), unit='m')

    >>> jnp.asarray([2, 3]) * q
    Quantity(Array([4, 6], dtype=int32), unit='m')

    """
    return replace(y, value=qlax.mul(x, ustrip(y)))


@register(lax.mul_p)
def mul_p_qv(x: ABCQ, y: ArrayLike, /) -> ABCQ:
    """Multiplication of a quantity and an array-like.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> q = u.quantity.BareQuantity(2, "m")

    >>> q * 2.0
    BareQuantity(Array(4., dtype=float32, ...), unit='m')

    >>> q * jnp.asarray(2)
    BareQuantity(Array(4, dtype=int32, ...), unit='m')

    >>> q * jnp.asarray([2, 3])
    BareQuantity(Array([4, 6], dtype=int32), unit='m')

    >>> q = u.Q(2, "m")

    >>> q * 2.0
    Quantity(Array(4., dtype=float32, ...), unit='m')

    >>> q * jnp.asarray(2)
    Quantity(Array(4, dtype=int32, weak_type=True), unit='m')

    >>> q * jnp.asarray([2, 3])
    Quantity(Array([4, 6], dtype=int32), unit='m')

    """
    return replace(x, value=qlax.mul(ustrip(x), y))


# ==============================================================================


@register(lax.ne_p)
def ne_p_qq(x: ABCQ, y: ABCQ, /) -> ArrayLike:
    """Inequality of two quantities.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> q1 = u.quantity.BareQuantity(1, "m")
    >>> q2 = u.quantity.BareQuantity(2, "m")
    >>> jnp.not_equal(q1, q2)
    Array(True, dtype=bool, ...)
    >>> q1 != q2
    Array(True, dtype=bool, ...)

    >>> q2 = u.quantity.BareQuantity(1, "m")
    >>> jnp.not_equal(q1, q2)
    Array(False, dtype=bool, ...)
    >>> q1 != q2
    Array(False, dtype=bool, ...)

    >>> q1 = u.Q(1, "m")
    >>> q2 = u.Q(2, "m")
    >>> jnp.not_equal(q1, q2)
    Array(True, dtype=bool, ...)
    >>> q1 != q2
    Array(True, dtype=bool, ...)

    >>> q2 = u.Q(1, "m")
    >>> jnp.not_equal(q1, q2)
    Array(False, dtype=bool, ...)
    >>> q1 != q2
    Array(False, dtype=bool, ...)

    """
    if not is_unit_convertible(x.unit, y.unit):
        msg = f"Cannot compare Q(x, {x.unit}) != Q(y, {y.unit})."
        raise UnitConversionError(msg)
    return qlax.ne(ustrip(x), ustrip(x.unit, y))  # re-dispatch on the values


@register(lax.ne_p)
def ne_p_vq(x: ArrayLike, y: ABCQ, /) -> ArrayLike:
    """Inequality of an array and a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> x = 1
    >>> q2 = u.quantity.BareQuantity(2, "")
    >>> jnp.not_equal(x, q2)
    Array(True, dtype=bool, ...)
    >>> x != q2
    Array(True, dtype=bool, ...)

    >>> q2 = u.quantity.BareQuantity(1, "")
    >>> jnp.not_equal(x, q2)
    Array(False, dtype=bool, ...)
    >>> x != q2
    Array(False, dtype=bool, ...)

    >>> x = u.Q(1, "")
    >>> q2 = u.Q(2, "")
    >>> jnp.not_equal(x, q2)
    Array(True, dtype=bool, ...)
    >>> x != q2
    Array(True, dtype=bool, ...)

    >>> q2 = u.Q(1, "")
    >>> jnp.not_equal(x, q2)
    Array(False, dtype=bool, ...)
    >>> x != q2
    Array(False, dtype=bool, ...)

    """
    yv = ustrip(y)
    yv = eqx.error_if(  # TODO: customize Exception type
        yv,
        not is_unit_convertible(one, y.unit) and jnp.logical_not(jnp.all(x == 0)),
        f"Cannot compare x != Q(y, {y.unit}) (except for x=0).",
    )
    return qlax.ne(x, yv)  # re-dispatch on the value


@register(lax.ne_p)
def ne_p_qv(x: ABCQ, y: ArrayLike, /) -> ArrayLike:
    """Inequality of a quantity and an array.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> x = 1
    >>> q1 = u.quantity.BareQuantity(2, "")
    >>> jnp.not_equal(q1, x)
    Array(True, dtype=bool, ...)
    >>> q1 != x
    Array(True, dtype=bool, ...)

    >>> q1 = u.quantity.BareQuantity(1, "")
    >>> jnp.not_equal(q1, x)
    Array(False, dtype=bool, ...)
    >>> q1 != x
    Array(False, dtype=bool, ...)

    >>> x = u.Q(1, "")
    >>> q1 = u.Q(2, "")
    >>> jnp.not_equal(q1, x)
    Array(True, dtype=bool, ...)
    >>> q1 != x
    Array(True, dtype=bool, ...)

    >>> q1 = u.Q(1, "")
    >>> jnp.not_equal(q1, x)
    Array(False, dtype=bool, ...)
    >>> q1 != x
    Array(False, dtype=bool, ...)

    """
    xv = ustrip(x)
    xv = eqx.error_if(  # TODO: customize Exception type
        xv,
        not is_unit_convertible(one, x.unit) and jnp.logical_not(jnp.all(y == 0)),
        f"Cannot compare Q(x, {x.unit}) != y (except for y=0).",
    )
    return qlax.ne(xv, y)  # re-dispatch on the value


# @register(lax.ne_p)
# def ne_p_qv(x: ABCPQ, y: ArrayLike) -> ArrayLike:
#     return lax.


# ==============================================================================


@register(lax.neg_p)
def neg_p(x: ABCQ, /) -> ABCQ:
    """Negation of a quantity.

    Examples
    --------
    >>> import unxt as u

    >>> q = u.quantity.BareQuantity(1, "m")
    >>> -q
    BareQuantity(Array(-1, dtype=int32, ...), unit='m')

    >>> q = u.Q(1, "m")
    >>> -q
    Quantity(Array(-1, dtype=int32, weak_type=True), unit='m')

    """
    return replace(x, value=qlax.neg(ustrip(x)))


# =============================================================================


@register(lax.nextafter_p)
def nextafter_p(x1: ABCQ, x2: ABCQ, /) -> ABCQ:
    """Next representable value after a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> q1 = u.quantity.BareQuantity(1, "")
    >>> q2 = u.quantity.BareQuantity(2, "")
    >>> jnp.nextafter(q1, q2)
    BareQuantity(Array(1.0000001, dtype=float32, ...), unit='')

    >>> q1 = u.Q(1, "")
    >>> q2 = u.Q(2, "")
    >>> jnp.nextafter(q1, q2)
    Quantity(Array(1.0000001, dtype=float32, ...), unit='')

    """
    u = unit_of(x1)
    return replace(x1, value=qlax.nextafter(ustrip(u, x1), ustrip(u, x2)))


# =============================================================================


@register(lax.not_p)
def not_p(x: ABCQ, /) -> ABCQ:
    """Logical negation of a quantity.

    Examples
    --------
    >>> import unxt as u

    >>> q = u.quantity.BareQuantity(1, "")
    >>> ~q
    BareQuantity(Array(-2, dtype=int32, weak_type=True), unit='')

    >>> q = u.Q(1, "")
    >>> ~q
    Quantity(Array(-2, dtype=int32, weak_type=True), unit='')

    """
    return replace(x, value=qlax.bitwise_not(ustrip(one, x)))


# ==============================================================================


@register(lax.or_p)
def or_p_qq(x: ABCQ, y: ABCQ, /) -> ABCQ:
    """Logical or of two quantities.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> q1 = u.quantity.BareQuantity(1, "")
    >>> q2 = u.quantity.BareQuantity(2, "")
    >>> jnp.bitwise_or(q1, q2)
    BareQuantity(Array(3, dtype=int32, ...), unit='')

    >>> q1 = u.Q(1, "")
    >>> q2 = u.Q(2, "")
    >>> jnp.bitwise_or(q1, q2)
    Quantity(Array(3, dtype=int32, weak_type=True), unit='')

    """
    return replace(x, value=qlax.bitwise_or(ustrip(one, x), ustrip(one, y)))


# ==============================================================================


@register(lax.pad_p)
def pad_p(operand: ABCQ, padding_value: ABCQ, /, *, padding_config: Any) -> ABCQ:
    """Pad a quantity with another quantity.

    Examples
    --------
    >>> import quaxed.lax as qlax
    >>> import unxt as u

    >>> x = u.quantity.BareQuantity([1, 2, 3], "m")
    >>> padding_value = u.quantity.BareQuantity(0, "m")
    >>> qlax.pad(x, padding_value, padding_config=((1, 1, 0),))
    BareQuantity(Array([0, 1, 2, 3, 0], dtype=int32), unit='m')

    >>> x = u.Q([1, 2, 3], "m")
    >>> padding_value = u.Q(0, "m")
    >>> qlax.pad(x, padding_value, padding_config=((1, 1, 0),))
    Quantity(Array([0, 1, 2, 3, 0], dtype=int32), unit='m')

    """
    padding_value_stripped = ustrip(operand.unit, padding_value)
    operand_stripped = ustrip(operand)

    # Promote dtypes to ensure compatibility
    operand_stripped, padding_value_stripped = promote_dtypes_if_needed(
        (operand.dtype, padding_value.dtype), operand_stripped, padding_value_stripped
    )

    return replace(
        operand,
        value=lax.pad_p.bind(
            operand_stripped, padding_value_stripped, padding_config=padding_config
        ),
    )


@register(lax.pad_p)
def pad_p_array_padding(
    operand: ABCQ, padding_value: ArrayLike, /, *, padding_config: Any
) -> ABCQ:
    """Pad a quantity with an array padding value.

    This is only allowed when the padding value is zero everywhere.
    This enables operations like jnp.diag that internally use pad with 0.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> x = u.quantity.BareQuantity([1, 2, 3], "m")
    >>> jnp.diag(x)
    BareQuantity(Array([[1, 0, 0],
                        [0, 2, 0],
                        [0, 0, 3]], dtype=int32), unit='m')

    >>> x = u.Q([1, 2, 3], "m")
    >>> result = jnp.diag(x)
    >>> result.shape
    (3, 3)
    >>> result.unit
    Unit("m")

    """
    # Convert padding_value to array to ensure it has dtype
    pad_val = jnp.asarray(padding_value)

    # Check that padding_value is zero everywhere
    pad_val = eqx.error_if(
        pad_val,
        jnp.logical_not(jnp.all(jnp.equal(pad_val, 0))),
        "Array padding values must be zero everywhere",
    )

    op_val = ustrip(operand)  # Promote dtypes to ensure compatibility
    op_val, pad_val = promote_dtypes_if_needed(
        (operand.dtype, pad_val.dtype), op_val, pad_val
    )

    return replace(
        operand, value=lax.pad_p.bind(op_val, pad_val, padding_config=padding_config)
    )


# ==============================================================================


@register(lax.polygamma_p)
def polygamma_p(m: ArrayLike, x: ABCQ, /) -> ABCQ:
    """Polygamma function of a quantity.

    Examples
    --------
    >>> import quaxed.lax as qlax

    >>> q = u.quantity.BareQuantity(3.0, "")
    >>> qlax.polygamma(1.0, q)
    BareQuantity(Array(0.39493403, dtype=float32, ...), unit='')

    >>> q = u.Q(3.0, "")
    >>> qlax.polygamma(1.0, q)
    Quantity(Array(0.39493403, dtype=float32, ...), unit='')

    """
    return replace(x, value=qlax.polygamma(m, ustrip(one, x)))


# ==============================================================================


@register(lax.population_count_p)
def population_count_p(x: ABCQ, /) -> ABCQ:
    r"""Return population count of a quantity.

    Examples
    --------
    >>> import quaxed.lax as qlax

    >>> q = u.quantity.BareQuantity(3, "")
    >>> qlax.population_count(q)
    BareQuantity(Array(2, dtype=int32, ...), unit='')

    >>> q = u.Q(3, "")
    >>> qlax.population_count(q)
    Quantity(Array(2, dtype=int32, weak_type=True), unit='')

    """
    return replace(x, value=lax.population_count(ustrip(one, x)))


# ==============================================================================


@register(lax.pow_p)
def pow_p_qq(x: ABCQ, y: ABCPQ["dimensionless"], /) -> ABCQ:
    """Power of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> q1 = u.quantity.BareQuantity(2.0, "m")
    >>> p = u.Q(3, "")
    >>> jnp.power(q1, p)
    BareQuantity(Array(8., dtype=float32, ...), unit='m3')
    >>> q1**p
    BareQuantity(Array(8., dtype=float32, ...), unit='m3')

    >>> q1 = u.Q(2.0, "m")
    >>> jnp.power(q1, p)
    Quantity(Array(8., dtype=float32, ...), unit='m3')
    >>> q1**p
    Quantity(Array(8., dtype=float32, ...), unit='m3')

    """
    yv = ustrip(one, y)
    y0 = yv[(0,) * yv.ndim]
    yv = eqx.error_if(yv, jnp.any(yv != y0), "power must be a scalar")
    return type_np(x)(value=qlax.pow(ustrip(x), y0), unit=x.unit**y0)


@register(lax.pow_p)
def pow_p_qf(x: ABCQ, y: ArrayLike, /) -> ABCQ:
    """Power of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> q1 = u.quantity.BareQuantity(2.0, "m")
    >>> y = jnp.array(3)
    >>> jnp.power(q1, y)
    BareQuantity(Array(8., dtype=float32, weak_type=True), unit='m3')
    >>> q1**y
    BareQuantity(Array(8., dtype=float32, weak_type=True), unit='m3')

    >>> q1 = u.Q(2.0, "m")
    >>> jnp.power(q1, y)
    Quantity(Array(8., dtype=float32, weak_type=True), unit='m3')
    >>> q1**y
    Quantity(Array(8., dtype=float32, weak_type=True), unit='m3')

    """
    return type_np(x)(value=qlax.pow(ustrip(x), y), unit=x.unit**y)


@register(lax.pow_p)
def pow_p_vq(x: ArrayLike, y: ABCPQ["dimensionless"], /) -> ABCQ:
    """Array raised to a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> x = jnp.array([2.0])
    >>> p = u.Q(3, "")
    >>> jnp.power(x, p)
    Quantity(Array([8.], dtype=float32), unit='')

    """
    return replace(y, value=qlax.pow(x, ustrip(y)))


@register(lax.pow_p)
def pow_p_abstractangle_arraylike(x: AbstractAngle, y: ArrayLike, /) -> ABCQ:
    """Power of an Angle by redispatching to Quantity.

    Examples
    --------
    >>> import math
    >>> import unxt as u

    >>> q1 = u.Angle(10.0, "deg")
    >>> y = 3.0
    >>> q1**y
    Quantity(Array(1000., dtype=float32, ...), unit='deg3')

    """
    return pow_p_qf(convert(x, Quantity), y)


# ==============================================================================


@register(lax.real_p)
def real_p(x: ABCQ, /) -> ABCQ:
    """Real part of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> jnp.real(u.quantity.BareQuantity(1.0, "m"))
    BareQuantity(Array(1., dtype=float32, ...), unit='m')

    >>> jnp.real(u.quantity.BareQuantity(1 + 2j, "m"))
    BareQuantity(Array(1., dtype=float32, ...), unit='m')

    >>> jnp.real(u.Q(1.0, "m"))
    Quantity(Array(1., dtype=float32, ...), unit='m')

    >>> jnp.real(u.Q(1 + 2j, "m"))
    Quantity(Array(1., dtype=float32, weak_type=True), unit='m')

    """
    return replace(x, value=qlax.real(ustrip(x)))


# ==============================================================================


@register(lax.reduce_and_p)
def reduce_and_p(operand: ABCQ, /, *, axes: Sequence[int]) -> Any:
    return lax.reduce_and_p.bind(ustrip(operand), axes=tuple(axes))


# ==============================================================================


@register(lax.reduce_max_p)
def reduce_max_p(operand: ABCQ, /, *, axes: Axes) -> ABCQ:
    return replace(operand, value=lax.reduce_max_p.bind(ustrip(operand), axes=axes))


# ==============================================================================


@register(lax.reduce_min_p)
def reduce_min_p(operand: ABCQ, /, *, axes: Axes) -> ABCQ:
    return replace(operand, value=lax.reduce_min_p.bind(ustrip(operand), axes=axes))


# ==============================================================================


@register(lax.reduce_or_p)
def reduce_or_p(operand: ABCQ, /, *, axes: Axes) -> ABCQ:
    return type_np(operand)(lax.reduce_or_p.bind(ustrip(operand), axes=axes), unit=one)


# ==============================================================================


@register(lax.reduce_prod_p)
def reduce_prod_p(operand: ABCQ, /, *, axes: Axes) -> ABCQ:
    value = lax.reduce_prod_p.bind(ustrip(operand), axes=axes)
    u = operand.unit ** prod(operand.shape[ax] for ax in axes)
    return type_np(operand)(value, unit=u)


# ==============================================================================


@register(lax.reduce_sum_p)
def reduce_sum_p(operand: ABCQ, /, **kw: Any) -> ABCQ:
    return replace(operand, value=lax.reduce_sum_p.bind(ustrip(operand), **kw))


# ==============================================================================


@register(lax.regularized_incomplete_beta_p)
def regularized_incomplete_beta_q(
    a: ArrayLike | ABCQ, b: ArrayLike | ABCQ, x: ArrayLike, /
) -> Array:
    """Regularized incomplete beta function.

    Examples
    --------
    >>> import quaxed.scipy.special as jsp
    >>> import unxt as u

    >>> a = u.quantity.BareQuantity(2.0, "")
    >>> b = u.quantity.BareQuantity(3.0, "")
    >>> x = 0.5
    >>> jsp.betainc(a, b, x).round(7)
    Array(0.6874998, dtype=float32, weak_type=True)

    >>> a = u.Q(2.0, "")
    >>> b = u.Q(3.0, "")
    >>> jsp.betainc(a, b, x).round(7)
    Array(0.6874998, dtype=float32, weak_type=True)

    """
    a = ustrip(AllowValue, one, a)
    b = ustrip(AllowValue, one, b)
    return lax.regularized_incomplete_beta_p.bind(a, b, x)


@register(lax.regularized_incomplete_beta_p)
def regularized_incomplete_beta_q(
    a: ArrayLike | ABCQ, b: ArrayLike | ABCQ, x: ABCQ, /
) -> ABCQ:
    """Regularized incomplete beta function.

    Examples
    --------
    >>> import quaxed.scipy.special as jsp
    >>> import unxt as u

    >>> a = 2.0
    >>> b = 3.0
    >>> x = u.quantity.BareQuantity(0.5, "")
    >>> jsp.betainc(a, b, x).round(7)
    BareQuantity(Array(0.6874998, dtype=float32, weak_type=True), unit='')

    >>> x = u.Q(0.5, "")
    >>> jsp.betainc(a, b, x).round(7)
    Quantity(Array(0.6874998, dtype=float32, weak_type=True), unit='')

    >>> x = u.Q(0.5, "m")
    >>> try:
    ...     jsp.betainc(a, b, x)
    ... except Exception as e:
    ...     print(e)
    'm' (length) and '' (dimensionless) are not convertible

    """
    a = ustrip(AllowValue, one, a)
    b = ustrip(AllowValue, one, b)
    xv = ustrip(one, x)
    return replace(x, value=lax.regularized_incomplete_beta_p.bind(a, b, xv))


# ==============================================================================


@register(lax.rem_p)
def rem_p_aa(x: AbstractAngle, y: ABCQ, /) -> AbstractAngle:
    """Remainder of an angle and a quantity.

    Examples
    --------
    >>> import unxt as u

    >>> q = u.Angle([1, 2, 3], "deg")
    >>> q % u.Q(4, "deg")
    Angle(Array([1, 2, 3], dtype=int32), unit='deg')

    """
    # Don't promote - preserve the Angle type
    return replace(x, value=ustrip(x) % ustrip(x.unit, y))


@register(lax.rem_p)
def rem_p_qq(x: ABCQ, y: ABCQ, /) -> ABCQ:
    """Remainder of two quantities.

    Examples
    --------
    >>> import unxt as u

    >>> q1 = u.quantity.BareQuantity(10, "m")
    >>> q2 = u.quantity.BareQuantity(3, "m")
    >>> q1 % q2
    BareQuantity(Array(1, dtype=int32, ...), unit='m')

    >>> q1 = u.Q(10, "m")
    >>> q2 = u.Q(3, "m")
    >>> q1 % q2
    Quantity(Array(1, dtype=int32, ...), unit='m')

    """
    x, y = promote(x, y)
    # Use % to preserve numpy arrays for StaticQuantity
    return replace(x, value=ustrip(x) % ustrip(x.unit, y))


@register(lax.rem_p)
def rem_p_uqv(
    x: Quantity["dimensionless"], y: ArrayLike, /
) -> Quantity["dimensionless"]:
    """Remainder of two quantities.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from unxt import Quantity

    >>> q1 = u.Q(10, "")
    >>> q2 = jnp.array(3)
    >>> q1 % q2
    Quantity(Array(1, dtype=int32, ...), unit='')

    """
    return replace(x, value=ustrip(x) % y)


# ==============================================================================


@register(lax.reshape_p)
def reshape_p(operand: ABCQ, /, **kw: Any) -> ABCQ:
    """Reshape a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> q = u.quantity.BareQuantity(jnp.arange(6), "m")
    >>> jnp.reshape(q, (3, 2))
    BareQuantity(Array([[0, 1],
                             [2, 3],
                             [4, 5]], dtype=int32), unit='m')

    >>> q = u.Q(jnp.arange(6), "m")
    >>> jnp.reshape(q, (3, 2))
    Quantity(Array([[0, 1],
                              [2, 3],
                              [4, 5]], dtype=int32), unit='m')

    """
    return replace(operand, value=lax.reshape_p.bind(ustrip(operand), **kw))


# ==============================================================================


@register(lax.rev_p)
def rev_p(operand: ABCQ, /, *, dimensions: Any) -> ABCQ:
    """Reverse a quantity.

    Examples
    --------
    >>> import quaxed.lax as qlax
    >>> import unxt as u

    >>> q = u.quantity.BareQuantity([0, 1, 2, 3], "m")
    >>> qlax.rev(q, dimensions=(0,))
    BareQuantity(Array([3, 2, 1, 0], dtype=int32), unit='m')

    >>> q = u.Q([0, 1, 2, 3], "m")
    >>> qlax.rev(q, dimensions=(0,))
    Quantity(Array([3, 2, 1, 0], dtype=int32), unit='m')

    """
    return replace(operand, value=qlax.rev(ustrip(operand), dimensions))


# ==============================================================================


@register(lax.round_p)
def round_p(x: ABCQ, /, *, rounding_method: Any) -> ABCQ:
    """Round a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> q = u.quantity.BareQuantity(1.234, "m")
    >>> jnp.round(q, 2)
    BareQuantity(Array(1.23, dtype=float32, ...), unit='m')

    >>> q = u.Q(1.234, "m")
    >>> jnp.round(q, 2)
    Quantity(Array(1.23, dtype=float32, ...), unit='m')

    """
    return replace(x, value=qlax.round(ustrip(x), rounding_method))


# ==============================================================================


@register(lax.rsqrt_p)
def rsqrt_p(x: ABCQ, /, **kw: Any) -> ABCQ:
    """Reciprocal square root of a quantity.

    Examples
    --------
    >>> import quaxed.lax as qlax
    >>> import unxt as u

    >>> q = u.quantity.BareQuantity(1 / 4, "m")
    >>> qlax.rsqrt(q)
    BareQuantity(Array(2., dtype=float32, ...), unit='1 / m(1/2)')

    >>> q = u.Q(1 / 4, "m")
    >>> qlax.rsqrt(q)
    Quantity(Array(2., dtype=float32, ...), unit='1 / m(1/2)')

    """
    return type_np(x)(lax.rsqrt_p.bind(ustrip(x), **kw), unit=x.unit ** (-1 / 2))


# ==============================================================================


@register(lax.scan_p)
def scan_p(arg0: ABCQ, arg1: ABCQ, /, *args: ArrayLike, **kw: Any) -> list[Array]:
    """Scan operator, e.g. for ``numpy.digitize``.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> x = u.Q(jnp.arange(0, 10), "deg")
    >>> x_bins = u.Q(jnp.linspace(0, 10, 4), "deg")
    >>> jnp.digitize(x, x_bins)
    Array([1, 1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=int32)

    """
    u = unit_of(arg0)
    arg0_ = ustrip(u, arg0)
    arg1_ = ustrip(u, arg1)
    return lax.scan_p.bind(arg0_, arg1_, *args, **kw)  # type: ignore[no-untyped-call]


# ==============================================================================


@register(lax.scatter_add_p)
def scatter_add_p_qvq(
    operand: ABCQ, scatter_indices: ArrayLike, updates: ABCQ, /, **kw: Any
) -> ABCQ:
    """Scatter-add operator.

    Examples
    --------
    >>> import quaxed.lax as qlax
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> indices = jnp.array([[4], [3], [1], [7]])

    # >>> updates = u.quantity.BareQuantity([9, 10, 11, 12], "m")
    # >>> tensor = u.quantity.BareQuantity(jnp.ones([8]), "m")
    # >>> qlax.scatter_add(
    # ...     tensor, indices, updates, dimension_numbers=qlax.ScatterDimensionNumbers
    # ... )

    """
    return replace(
        operand,
        value=lax.scatter_add_p.bind(
            ustrip(operand), scatter_indices, ustrip(operand.unit, updates), **kw
        ),
    )


@register(lax.scatter_add_p)
def scatter_add_p_vvq(
    operand: ArrayLike, scatter_indices: ArrayLike, updates: ABCQ, /, **kw: Any
) -> ABCQ:
    """Scatter-add operator between an Array and a Quantity.

    This is an interesting case where the Quantity is the `updates` and the Array
    is the `operand`. For some reason when doing a ``scatter_add`` between two
    Quantity objects an intermediate Array operand is created. Therefore we
    need to pretend that the Array has the same units as the `updates`.

    """
    return replace(
        updates,
        value=lax.scatter_add_p.bind(operand, scatter_indices, ustrip(updates), **kw),
    )


# ==============================================================================


@register(lax.select_n_p)
def select_n_p(which: ABCQ, /, *cases: ABCQ) -> ABCQ:
    """Select from a list of quantities using a quantity selector.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> a = u.Q([1.0, 5.0, 9.0], "km")
    >>> b = u.Q([2.0, 4.0, 10.0], "km")
    >>> which = u.Q(a > b, "")
    >>> jnp.where(which, a, b)
    Quantity(Array([ 2.,  5., 10.], dtype=float32), unit='km')

    """
    u = cases[0].unit
    cases_ = (ustrip(u, case) for case in cases)
    return type_np(which)(lax.select_n(ustrip(one, which), *cases_), unit=u)


@register(lax.select_n_p)
def select_n_p_vq(which: ABCQ, case0: ABCQ, case1: ArrayLike, /) -> ABCQ:
    """Select from a quantity and array using a quantity selector."""
    # encountered from jnp.hypot
    u = case0.unit
    return type_np(which)(
        lax.select_n(ustrip(one, which), ustrip(u, case0), case1), unit=u
    )


@register(lax.select_n_p)
def select_n_p_jjq(which: ArrayLike, case0: ArrayLike, case1: ABCQ, /) -> ABCQ:
    """Select from an array and quantity using a quantity selector."""
    # Used by a `jnp.linalg.trace`
    return replace(case1, value=qlax.select_n(which, case0, ustrip(case1)))


@register(lax.select_n_p)
def select_n_p_jqj(which: ArrayLike, case0: ABCQ, case1: ArrayLike, /) -> ABCQ:
    """Select from a quantity and array using a non-quantity selector.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> x = u.Q([1.0, 5.0, 9.0], "km")
    >>> y = u.Q([2.0, 4.0, 10.0], "km")

    >>> jnp.hypot(x, y)
    Quantity(Array([ 2.236068 ,  6.4031243, 13.453625 ], dtype=float32), unit='km')

    >>> jnp.triu(u.Q([[1, 2], [3, 4]], "km"))
    Quantity(Array([[1, 2],
                              [0, 4]], dtype=int32), unit='km')

    """
    return replace(case0, value=qlax.select_n(which, ustrip(case0), case1))


@register(lax.select_n_p)
def select_n_p_jsqj(
    which: ArrayLike, case0: StaticQuantity, case1: ArrayLike, /
) -> Quantity:
    """Select from a StaticQuantity and array using a non-quantity selector.

    StaticQuantity operations that go through JAX's select_n produce JAX arrays,
    so we return a Quantity instead.
    """
    return Quantity(qlax.select_n(which, ustrip(case0), case1), unit=unit_of(case0))


@register(lax.select_n_p)
def select_n_p_jaq(
    which: ArrayLike, case0: AbstractAngle, /, *cases: ABCQ
) -> AbstractAngle:
    """Select from Angle and quantities using a non-quantity selector.

    This preserves the Angle type when the first case is an Angle.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> a = u.Angle([480, 720], "deg")
    >>> a % u.Q(360, "deg")
    Angle(Array([120,   0], dtype=int32), unit='deg')

    """
    u = unit_of(case0)
    all_cases = (case0, *cases)
    dtypes = tuple(case.dtype for case in all_cases)
    casesv = promote_dtypes_if_needed(dtypes, *(ustrip(u, case) for case in all_cases))

    return replace(case0, value=qlax.select_n(which, *casesv))


@register(lax.select_n_p)
def select_n_p_jsq(
    which: ArrayLike, case0: StaticQuantity, /, *cases: ABCQ
) -> Quantity:
    """Select from StaticQuantity and quantities using a non-quantity selector.

    StaticQuantity operations that go through JAX's select_n produce JAX arrays,
    so we return a Quantity instead of trying to put a JAX array into a StaticQuantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> sq1 = u.StaticQuantity(10, "s")
    >>> sq2 = u.StaticQuantity(3, "s")
    >>> sq1 // sq2
    Quantity(Array(3, dtype=int32), unit='')

    """
    u = unit_of(case0)
    all_cases = (case0, *cases)
    dtypes = tuple(case.dtype for case in all_cases)
    casesv = promote_dtypes_if_needed(dtypes, *(ustrip(u, case) for case in all_cases))

    return Quantity(qlax.select_n(which, *casesv), unit=u)


@register(lax.select_n_p)
def select_n_p_jqq(which: ArrayLike, /, *cases: ABCQ) -> ABCQ:
    """Select from a list of quantities using a non-quantity selector.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    We can use a non-quantity selector to select from a list of quantities.

    >>> a = u.Q([1.0, 5.0, 9.0], "kpc")
    >>> b = u.Q([2.0, 6.0, 10.0], "kpc")
    >>> jnp.select(([a > Q(4, "kpc"), b < Q(8, "kpc")]), [a, b], default=Q(0, "kpc"))
    Quantity(Array([2., 5., 9.], dtype=float32), unit='kpc')

    This selection dispatch also happens when using ``jnp.hypot``.

    >>> a = u.Q([3], "kpc")
    >>> b = u.Q([4], "kpc")
    >>> jnp.hypot(a, b)
    Quantity(Array([5.], dtype=float32), unit='kpc')

    """
    # Promote all cases to a common type (e.g., StaticQuantity + Quantity -> Quantity)
    cases = promote(*cases)
    u = unit_of(cases[0])
    dtypes = tuple(case.dtype for case in cases)
    casesv = promote_dtypes_if_needed(dtypes, *(ustrip(u, case) for case in cases))

    # If result is a JAX array but promoted type is StaticQuantity, use Quantity
    return replace(cases[0], value=qlax.select_n(which, *casesv))


# ==============================================================================


@register(lax.shift_right_arithmetic_p)
def shift_right_arithmetic_p(x: ABCQ, y: ABCQ | float | int, /) -> ABCQ:
    """Shift right arithmetic of a quantity.

    Examples
    --------
    >>> import quaxed.lax as qlax
    >>> import unxt as u

    >>> q = u.quantity.BareQuantity(1, "")
    >>> qlax.shift_right_arithmetic(q, 2)
    BareQuantity(Array(0, dtype=int32, ...), unit='')

    >>> q = u.Q(1, "")
    >>> qlax.shift_right_arithmetic(q, 2)
    Quantity(Array(0, dtype=int32, weak_type=True), unit='')

    """
    return replace(
        x, value=qlax.shift_right_arithmetic(ustrip(one, x), ustrip(AllowValue, one, y))
    )


# ==============================================================================


@register(lax.sign_p)
def sign_p(x: ABCQ, /) -> ArrayLike:
    """Sign of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> q = u.quantity.BareQuantity(10, "m")
    >>> jnp.sign(q)
    Array(1, dtype=int32, ...)

    >>> q = u.Q(10, "m")
    >>> jnp.sign(q)
    Array(1, dtype=int32, ...)

    """
    return lax.sign(ustrip(x))


# ==============================================================================


@register(lax.sin_p)
def sin_p(x: ABCQ, /, **kw: Any) -> ABCQ:
    """Sine of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> q = u.quantity.BareQuantity(90, "deg")
    >>> jnp.sin(q)
    BareQuantity(Array(1., dtype=float32, ...), unit='')

    >>> q = u.quantity.BareQuantity(jnp.pi / 2, "")
    >>> jnp.sin(q)
    BareQuantity(Array(1., dtype=float32, ...), unit='')

    >>> q = u.Q(90, "deg")
    >>> jnp.sin(q)
    Quantity(Array(1., dtype=float32, ...), unit='')

    >>> q = u.Q(jnp.pi / 2, "")
    >>> jnp.sin(q)
    Quantity(Array(1., dtype=float32, ...), unit='')

    """
    return type_np(x)(lax.sin_p.bind(_to_val_rad_or_one(x), **kw), unit=one)


@register(lax.sin_p)
def sin_p_abstractangle(x: AbstractAngle, /, **kw: Any) -> ABCQ:
    """Sine of an Angle.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> q = u.Angle(90, "deg")
    >>> jnp.sin(q)
    Quantity(Array(1., dtype=float32, ...), unit='')

    """
    return sin_p(convert(x, Quantity), **kw)


# ==============================================================================


@register(lax.sinh_p)
def sinh_p(x: ABCQ, /) -> ABCQ:
    """Sinh of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> q = u.quantity.BareQuantity(90, "deg")
    >>> jnp.sinh(q)
    BareQuantity(Array(2.301299, dtype=float32, ...), unit='')

    >>> q = u.quantity.BareQuantity(jnp.pi / 2, "")
    >>> jnp.sinh(q)
    BareQuantity(Array(2.301299, dtype=float32, ...), unit='')

    >>> q = u.Q(90, "deg")
    >>> jnp.sinh(q)
    Quantity(Array(2.301299, dtype=float32, ...), unit='')

    >>> q = u.Q(jnp.pi / 2, "")
    >>> jnp.sinh(q)
    Quantity(Array(2.301299, dtype=float32, ...), unit='')

    """
    return type_np(x)(lax.sinh(_to_val_rad_or_one(x)), unit=one)


# ==============================================================================


@register(lax.shift_left_p)
def shift_left_p(x: ABCQ, y: ABCQ | float | int, /, **kw: Any) -> ABCQ:
    """Shift left of a quantity.

    Examples
    --------
    >>> import quaxed.lax as qlax
    >>> import unxt as u

    >>> q = u.quantity.BareQuantity(1, "")
    >>> qlax.shift_left(q, 2)
    BareQuantity(Array(4, dtype=int32, ...), unit='')

    >>> q = u.Q(1, "")
    >>> qlax.shift_left(q, 2)
    Quantity(Array(4, dtype=int32, weak_type=True), unit='')

    """
    return replace(x, value=qlax.shift_left(ustrip(x), ustrip(AllowValue, one, y)))


# ==============================================================================


@register(lax.slice_p)
def slice_p(
    operand: ABCQ, /, *, start_indices: Any, limit_indices: Any, strides: Any
) -> ABCQ:
    return replace(
        operand,
        value=lax.slice_p.bind(
            ustrip(operand),
            start_indices=start_indices,
            limit_indices=limit_indices,
            strides=strides,
        ),
    )


# ==============================================================================


# Called by `argsort`
@register(lax.sort_p)
def sort_p_two_operands(
    operand0: ABCQ,
    operand1: ArrayLike,
    /,
    *,
    dimension: int,
    is_stable: bool,
    num_keys: int,
) -> tuple[ABCQ, ArrayLike]:
    out0, out1 = lax.sort_p.bind(  # type: ignore[no-untyped-call]
        ustrip(operand0),
        operand1,
        dimension=dimension,
        is_stable=is_stable,
        num_keys=num_keys,
    )
    return (replace(operand0, value=out0), out1)


# Called by `sort`
@register(lax.sort_p)
def sort_p_one_operand(
    operand: ABCQ, /, *, dimension: int, is_stable: bool, num_keys: int
) -> tuple[ABCQ]:
    (out,) = lax.sort_p.bind(  # type: ignore[no-untyped-call]
        ustrip(operand), dimension=dimension, is_stable=is_stable, num_keys=num_keys
    )
    return (type_np(operand)(out, unit=operand.unit),)


# ==============================================================================


@register(lax.split_p)
def split_p(x: ABCQ, /, **kw: Any) -> list[ABCQ]:
    cls, u = type(x), x.unit
    return [cls(arr, unit=u) for arr in lax.split_p.bind(x.value, **kw)]  # type: ignore[no-untyped-call]


# ==============================================================================


@register(lax.square_p)
def square_p(x: ABCQ, /) -> ABCQ:
    """Square of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> q = u.quantity.BareQuantity(3, "m")
    >>> jnp.square(q)
    BareQuantity(Array(9, dtype=int32, ...), unit='m2')

    >>> q = u.Q(3, "m")
    >>> jnp.square(q)
    Quantity(Array(9, dtype=int32, ...), unit='m2')

    """
    return type_np(x)(lax.square(ustrip(x)), unit=x.unit**2)


# ==============================================================================


@register(lax.sqrt_p)
def sqrt_p_q(x: ABCQ, /, **kw: Any) -> ABCQ:
    """Square root of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> q = u.quantity.BareQuantity(9, "m")
    >>> jnp.sqrt(q)
    BareQuantity(Array(3., dtype=float32, ...), unit='m(1/2)')

    >>> q = u.Q(9, "m")
    >>> jnp.sqrt(q)
    Quantity(Array(3., dtype=float32, ...), unit='m(1/2)')

    """
    # Apply sqrt to the value and adjust the unit
    return type_np(x)(lax.sqrt_p.bind(ustrip(x), **kw), unit=x.unit ** (1 / 2))


@register(lax.sqrt_p)
def sqrt_p_abstractangle(x: AbstractAngle, /, **kw: Any) -> ABCQ:
    """Square root of an Angle.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> q = u.Angle(9, "deg")
    >>> jnp.sqrt(q)
    Quantity(Array(3., dtype=float32, ...), unit='deg(1/2)')

    """
    return sqrt_p_q(convert(x, Quantity), **kw)


# ==============================================================================


@register(lax.squeeze_p)
def squeeze_p(x: ABCQ, /, *, dimensions: Any) -> ABCQ:
    """Squeeze a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> q = u.quantity.BareQuantity(jnp.array([[[1], [2], [3]]]), "m")
    >>> jnp.squeeze(q)
    BareQuantity(Array([1, 2, 3], dtype=int32), unit='m')

    >>> q = u.Q(jnp.array([[[1], [2], [3]]]), "m")
    >>> jnp.squeeze(q)
    Quantity(Array([1, 2, 3], dtype=int32), unit='m')

    """
    return type_np(x)(lax.squeeze(ustrip(x), dimensions), unit=x.unit)


# ==============================================================================


@register(lax.stop_gradient_p)
def stop_gradient_p(x: ABCQ, /) -> ABCQ:
    """Stop gradient of a quantity.

    Examples
    --------
    >>> import quaxed.lax as qlax
    >>> import unxt as u

    >>> q = u.quantity.BareQuantity(1.0, "m")
    >>> qlax.stop_gradient(q)
    BareQuantity(Array(1., dtype=float32, ...), unit='m')

    """
    return replace(x, value=qlax.stop_gradient(ustrip(x)))


# ==============================================================================
# Subtraction


@register(lax.sub_p)
def sub_p_qq(x: ABCQ, y: ABCQ, /) -> ABCQ:
    """Subtract two quantities.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> q1 = u.quantity.BareQuantity(1.0, "km")
    >>> q2 = u.quantity.BareQuantity(500.0, "m")
    >>> jnp.subtract(q1, q2)
    BareQuantity(Array(0.5, dtype=float32, ...), unit='km')
    >>> q1 - q2
    BareQuantity(Array(0.5, dtype=float32, ...), unit='km')

    >>> q1 = u.Q(1.0, "km")
    >>> q2 = u.Q(500.0, "m")
    >>> jnp.subtract(q1, q2)
    Quantity(Array(0.5, dtype=float32, ...), unit='km')
    >>> q1 - q2
    Quantity(Array(0.5, dtype=float32, ...), unit='km')

    """
    x, y = promote(x, y)

    # Get the values, promoting if needed
    xv = ustrip(x)
    yv = ustrip(x.unit, y)
    xv, yv = promote_dtypes_if_needed((x.dtype, y.dtype), xv, yv)
    # Return the subtracted values, and the unit of the first operand
    return replace(x, value=xv - yv)


@register(lax.sub_p)
def sub_p_vq(x: ArrayLike, y: ABCQ, /) -> ABCQ:
    """Subtract a quantity from an array.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> x = 1_000
    >>> q = u.quantity.BareQuantity(500.0, "")
    >>> jnp.subtract(x, q)
    BareQuantity(Array(500., dtype=float32, ...), unit='')

    >>> x - q
    BareQuantity(Array(500., dtype=float32, ...), unit='')

    >>> q = u.Q(500.0, "")
    >>> jnp.subtract(x, q)
    Quantity(Array(500., dtype=float32, ...), unit='')

    >>> x - q
    Quantity(Array(500., dtype=float32, ...), unit='')

    """
    y = uconvert(one, y)
    return replace(y, value=x - ustrip(y))


@register(lax.sub_p)
def sub_p_qv(x: ABCQ, y: ArrayLike, /) -> ABCQ:
    """Subtract an array from a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> q = u.quantity.BareQuantity(500.0, "")
    >>> y = 1_000
    >>> jnp.subtract(q, y)
    BareQuantity(Array(-500., dtype=float32, ...), unit='')

    >>> q - y
    BareQuantity(Array(-500., dtype=float32, ...), unit='')

    >>> q = u.Q(500.0, "")
    >>> jnp.subtract(q, y)
    Quantity(Array(-500., dtype=float32, ...), unit='')

    >>> q - y
    Quantity(Array(-500., dtype=float32, ...), unit='')

    """
    x = uconvert(one, x)
    return replace(x, value=ustrip(x) - y)


# ==============================================================================


@register(lax.tan_p)
def tan_p(x: ABCQ, /, **kw: Any) -> ABCQ:
    """Tangent of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> q = u.quantity.BareQuantity(45, "deg")
    >>> jnp.tan(q)
    BareQuantity(Array(1., dtype=float32, weak_type=True), unit='')

    >>> q = u.quantity.BareQuantity(jnp.pi / 4, "")
    >>> jnp.tan(q)
    BareQuantity(Array(1., dtype=float32, weak_type=True), unit='')

    >>> q = u.Q(45, "deg")
    >>> jnp.tan(q)
    Quantity(Array(1., dtype=float32, weak_type=True), unit='')

    >>> q = u.Q(jnp.pi / 4, "")
    >>> jnp.tan(q)
    Quantity(Array(1., dtype=float32, weak_type=True), unit='')

    """
    return type_np(x)(lax.tan_p.bind(_to_val_rad_or_one(x), **kw), unit=one)


@register(lax.tan_p)
def tan_p_abstractangle(x: AbstractAngle, /, **kw: Any) -> ABCQ:
    """Tangent of an Angle.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> q = u.Angle(45, "deg")
    >>> jnp.tan(q)
    Quantity(Array(1., dtype=float32, ...), unit='')

    """
    return tan_p(convert(x, Quantity), **kw)


# ==============================================================================


@register(lax.tanh_p)
def tanh_p(x: ABCQ, /, **kw: Any) -> ABCQ:
    """Hyperbolic tangent of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> q = u.quantity.BareQuantity(45, "deg")
    >>> jnp.tanh(q)
    BareQuantity(Array(0.65579426, dtype=float32, weak_type=True), unit='')

    >>> q = u.quantity.BareQuantity(jnp.pi / 4, "")
    >>> jnp.tanh(q)
    BareQuantity(Array(0.65579426, dtype=float32, weak_type=True), unit='')

    >>> q = u.Q(45, "deg")
    >>> jnp.tanh(q)
    Quantity(Array(0.65579426, dtype=float32, weak_type=True), unit='')

    >>> q = u.Q(jnp.pi / 4, "")
    >>> jnp.tanh(q)
    Quantity(Array(0.65579426, dtype=float32, weak_type=True), unit='')

    """
    return type_np(x)(lax.tanh_p.bind(_to_val_rad_or_one(x), **kw), unit=one)


# ==============================================================================


@register(lax.top_k_p)
def top_k_p(operand: ABCQ, /, **kw: Any) -> ABCQ:
    """Top k elements of a quantity.

    Examples
    --------
    >>> import quaxed.lax as qlax
    >>> import unxt as u

    >>> q = u.quantity.BareQuantity([1, 2, 3], "m")
    >>> qlax.top_k(q, k=2)
    [BareQuantity(Array([3, 2], dtype=int32), unit='m'),
     BareQuantity(Array([2, 1], dtype=int32), unit='m')]

    >>> q = u.Q([1, 2, 3], "m")
    >>> qlax.top_k(q, k=2)
    [Quantity(Array([3, 2], dtype=int32), unit='m'),
     Quantity(Array([2, 1], dtype=int32), unit='m')]

    """
    return replace(operand, value=lax.top_k_p.bind(ustrip(operand), **kw))  # type: ignore[no-untyped-call]


# ==============================================================================


@register(lax.transpose_p)
def transpose_p(operand: ABCQ, /, *, permutation: Any) -> ABCQ:
    """Transpose a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> x = jnp.arange(6).reshape(2, 3)

    >>> q = u.quantity.BareQuantity(x, "m")
    >>> jnp.transpose(q)
    BareQuantity(Array([[0, 3], [1, 4], [2, 5]], dtype=int32), unit='m')

    >>> q = u.Q(x, "m")
    >>> jnp.transpose(q)
    Quantity(Array([[0, 3], [1, 4], [2, 5]], dtype=int32), unit='m')

    """
    return replace(
        operand, value=lax.transpose_p.bind(ustrip(operand), permutation=permutation)
    )


# ==============================================================================


@register(lax.xor_p)
def xor_p_qq(x: ABCQ, y: ABCQ, /) -> ABCQ:
    """Logical or of two quantities.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u

    >>> q1 = u.quantity.BareQuantity(1, "")
    >>> q2 = u.quantity.BareQuantity(2, "")
    >>> jnp.bitwise_xor(q1, q2)
    BareQuantity(Array(3, dtype=int32, ...), unit='')

    >>> q1 = u.Q(1, "")
    >>> q2 = u.Q(2, "")
    >>> jnp.bitwise_xor(q1, q2)
    Quantity(Array(3, dtype=int32, weak_type=True), unit='')

    """
    return replace(x, value=qlax.bitwise_xor(ustrip(one, x), ustrip(one, y)))


# ==============================================================================


@register(lax.zeta_p)
def zeta_p(x: ABCQ, q: ArrayLike, /) -> ABCQ:
    return replace(x, value=lax.zeta_p.bind(ustrip(x), q))
