# pylint: disable=import-error, too-many-lines

__all__: list[str] = []

from collections.abc import Callable, Sequence
from dataclasses import replace
from math import prod
from typing import Any, TypeAlias, TypeVar

import equinox as eqx
import jax.tree as jt
from astropy.units import (  # pylint: disable=no-name-in-module
    UnitConversionError,
    dimensionless_unscaled as one,
    radian,
)
from jax import lax, numpy as jnp
from jax._src.ad_util import add_any_p
from jax.core import Primitive
from jaxtyping import Array, ArrayLike
from plum import promote
from plum.parametric import type_unparametrized as type_np
from quax import register as register_

from quaxed import lax as qlax

from .base import AbstractQuantity, can_convert_unit
from .base_parametric import AbstractParametricQuantity
from .core import Quantity
from .distance import AbstractDistance
from .functional import uconvert, ustrip
from unxt._unxt.units.core import units

T = TypeVar("T")

Axes: TypeAlias = tuple[int, ...]


def register(primitive: Primitive) -> Callable[[T], T]:
    """:func`quax.register`, but makes mypy happy."""
    return register_(primitive)


def _to_value_rad_or_one(q: AbstractQuantity) -> ArrayLike:
    return ustrip(radian if can_convert_unit(q.unit, radian) else one, q)


def _bshape(arrs: tuple[Any, ...], /) -> tuple[int, ...]:
    return jnp.broadcast_shapes(*map(jnp.shape, arrs))


################################################################################
# Registering Primitives

# ==============================================================================


@register(lax.abs_p)
def _abs_p(x: AbstractQuantity) -> AbstractQuantity:
    """Absolute value of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from unxt import Quantity
    >>> q = Quantity(-1, "m")
    >>> jnp.abs(q)
    Quantity['length'](Array(1, dtype=int32, ...), unit='m')
    >>> abs(q)
    Quantity['length'](Array(1, dtype=int32, ...), unit='m')

    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(-1, "m")
    >>> jnp.abs(q)
    UncheckedQuantity(Array(1, dtype=int32, ...), unit='m')
    >>> abs(q)
    UncheckedQuantity(Array(1, dtype=int32, ...), unit='m')

    >>> from unxt import Distance
    >>> d = Distance(-1, "m")
    >>> jnp.abs(d)
    Distance(Array(1, dtype=int32, ...), unit='m')

    >>> from unxt import Parallax
    >>> p = Parallax(-1, "mas", check_negative=False)
    >>> jnp.abs(p)
    Parallax(Array(1, dtype=int32, ...), unit='mas')

    >>> from unxt import DistanceModulus
    >>> dm = DistanceModulus(-1, "mag")
    >>> jnp.abs(dm)
    DistanceModulus(Array(1, dtype=int32, weak_type=True), unit='mag')

    """
    return replace(x, value=lax.abs(x.value))


# ==============================================================================


@register(lax.acos_p)
def _acos_p_aq(x: AbstractQuantity) -> AbstractQuantity:
    """Inverse cosine of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as xp
    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(-1, "")
    >>> jnp.acos(q)
    UncheckedQuantity(Array(3.1415927, dtype=float32, ...), unit='rad')

    >>> from unxt import Quantity
    >>> q = Quantity(-1, "")
    >>> jnp.acos(q)
    Quantity['angle'](Array(3.1415927, dtype=float32, ...), unit='rad')

    """
    x_ = ustrip(one, x)
    return type_np(x)(value=lax.acos(x_), unit=radian)


# ==============================================================================


@register(lax.acosh_p)
def _acosh_p_aq(x: AbstractQuantity) -> AbstractQuantity:
    """Inverse hyperbolic cosine of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as xp
    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(2.0, "")
    >>> jnp.acosh(q)
    UncheckedQuantity(Array(1.316958, dtype=float32, ...), unit='rad')

    >>> from unxt import Quantity
    >>> q = Quantity(2.0, "")
    >>> jnp.acosh(q)
    Quantity['angle'](Array(1.316958, dtype=float32, ...), unit='rad')

    """
    x_ = ustrip(one, x)
    return type_np(x)(value=lax.acosh(x_), unit=radian)


# ==============================================================================
# Addition


@register(lax.add_p)
def _add_p_aqaq(x: AbstractQuantity, y: AbstractQuantity) -> AbstractQuantity:
    """Add two quantities.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from unxt import UncheckedQuantity

    >>> q1 = UncheckedQuantity(1.0, "km")
    >>> q2 = UncheckedQuantity(500.0, "m")
    >>> jnp.add(q1, q2)
    UncheckedQuantity(Array(1.5, dtype=float32, ...), unit='km')
    >>> q1 + q2
    UncheckedQuantity(Array(1.5, dtype=float32, ...), unit='km')

    >>> from unxt import Quantity
    >>> q1 = Quantity(1.0, "km")
    >>> q2 = Quantity(500.0, "m")
    >>> jnp.add(q1, q2)
    Quantity['length'](Array(1.5, dtype=float32, ...), unit='km')
    >>> q1 + q2
    Quantity['length'](Array(1.5, dtype=float32, ...), unit='km')

    >>> from unxt import Distance
    >>> d1 = Distance(1.0, "km")
    >>> d2 = Distance(500.0, "m")
    >>> jnp.add(d1, d2)
    Distance(Array(1.5, dtype=float32, ...), unit='km')

    >>> from unxt import Parallax
    >>> p1 = Parallax(1.0, "mas")
    >>> p2 = Parallax(500.0, "uas")
    >>> jnp.add(p1, p2)
    Parallax(Array(1.5, dtype=float32, ...), unit='mas')

    >>> from unxt import DistanceModulus
    >>> dm1 = DistanceModulus(1.0, "mag")
    >>> dm2 = DistanceModulus(500.0, "mag")
    >>> jnp.add(dm1, dm2)
    DistanceModulus(Array(501., dtype=float32, ...), unit='mag')

    """
    return replace(x, value=lax.add(x.value, ustrip(x.unit, y)))


@register(lax.add_p)
def _add_p_vaq(x: ArrayLike, y: AbstractQuantity) -> AbstractQuantity:
    """Add a value and a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp

    >>> x = jnp.asarray(500.0)

    `unxt.UncheckedQuantity`:

    >>> from unxt import UncheckedQuantity
    >>> y = UncheckedQuantity(1.0, "km")

    >>> try: jnp.add(x, y)
    ... except Exception as e: print(e)
    'km' (length) and '' (dimensionless) are not convertible

    >>> try: x + y
    ... except Exception as e: print(e)
    'km' (length) and '' (dimensionless) are not convertible

    >>> y = UncheckedQuantity(100.0, "")
    >>> jnp.add(x, y)
    UncheckedQuantity(Array(600., dtype=float32, ...), unit='')

    >>> x + y
    UncheckedQuantity(Array(600., dtype=float32, ...), unit='')

    >>> q2 = UncheckedQuantity(1.0, "km")
    >>> q3 = UncheckedQuantity(1_000.0, "m")
    >>> jnp.add(x, q2 / q3)
    UncheckedQuantity(Array(501., dtype=float32, weak_type=True), unit='')

    `unxt.Quantity`:

    >>> from unxt import Quantity
    >>> x = jnp.asarray(500.0)
    >>> q2 = Quantity(1.0, "km")
    >>> try: x + q2
    ... except Exception as e: print(e)
    'km' (length) and '' (dimensionless) are not convertible

    >>> q2 = Quantity(100.0, "")
    >>> jnp.add(x, q2)
    Quantity['dimensionless'](Array(600., dtype=float32, ...), unit='')

    >>> x + q2
    Quantity['dimensionless'](Array(600., dtype=float32, ...), unit='')

    >>> q2 = Quantity(1.0, "km")
    >>> q3 = Quantity(1_000.0, "m")
    >>> jnp.add(x, q2 / q3)
    Quantity['dimensionless'](Array(501., dtype=float32, weak_type=True), unit='')

    """
    y = uconvert(one, y)
    return replace(y, value=lax.add(x, y.value))


@register(lax.add_p)
def _add_p_aqv(x: AbstractQuantity, y: ArrayLike) -> AbstractQuantity:
    """Add a quantity and a value.

    Examples
    --------
    >>> import quaxed.numpy as jnp

    >>> y = jnp.asarray(500.0)

    `unxt.UncheckedQuantity`:

    >>> from unxt import UncheckedQuantity
    >>> q1 = UncheckedQuantity(1.0, "km")

    >>> try: jnp.add(q1, y)
    ... except Exception as e: print(e)
    'km' (length) and '' (dimensionless) are not convertible

    >>> try: q1 + y
    ... except Exception as e: print(e)
    'km' (length) and '' (dimensionless) are not convertible

    >>> q1 = UncheckedQuantity(100.0, "")
    >>> jnp.add(q1, y)
    UncheckedQuantity(Array(600., dtype=float32, ...), unit='')

    >>> q1 + y
    UncheckedQuantity(Array(600., dtype=float32, ...), unit='')

    >>> q2 = UncheckedQuantity(1.0, "km")
    >>> q3 = UncheckedQuantity(1_000.0, "m")
    >>> jnp.add(q2 / q3, y)
    UncheckedQuantity(Array(501., dtype=float32, weak_type=True), unit='')

    `unxt.Quantity`:

    >>> from unxt import Quantity
    >>> q1 = Quantity(1.0, "km")

    >>> try: jnp.add(q1, y)
    ... except Exception as e: print(e)
    'km' (length) and '' (dimensionless) are not convertible

    >>> try: q1 + y
    ... except Exception as e: print(e)
    'km' (length) and '' (dimensionless) are not convertible

    >>> q1 = Quantity(100.0, "")
    >>> jnp.add(q1, y)
    Quantity[...](Array(600., dtype=float32, ...), unit='')

    >>> q1 + y
    Quantity[...](Array(600., dtype=float32, ...), unit='')

    >>> q2 = Quantity(1.0, "km")
    >>> q3 = Quantity(1_000.0, "m")
    >>> jnp.add(q2 / q3, y)
    Quantity['dimensionless'](Array(501., dtype=float32, weak_type=True), unit='')

    """
    x = uconvert(one, x)
    return replace(x, value=lax.add(x.value, y))


# ==============================================================================


@register(add_any_p)
def _add_any_p(
    x: AbstractParametricQuantity, y: AbstractParametricQuantity
) -> AbstractParametricQuantity:
    """Add two quantities using the ``jax._src.ad_util.add_any_p``."""
    return replace(x, value=add_any_p.bind(x.value, y.value))


# ==============================================================================


@register(lax.and_p)
def _and_p_aq(x1: AbstractQuantity, x2: AbstractQuantity, /) -> ArrayLike:
    """Bitwise AND of two quantities.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from unxt import UncheckedQuantity
    >>> x1 = UncheckedQuantity(1, "")
    >>> x2 = UncheckedQuantity(2, "")
    >>> jnp.bitwise_and(x1, x2)
    Array(0, dtype=int32, ...)

    >>> from unxt import Quantity
    >>> x1 = Quantity(1, "")
    >>> x2 = Quantity(2, "")
    >>> jnp.bitwise_and(x1, x2)
    Array(0, dtype=int32, ...)

    """
    return lax.and_p.bind(ustrip(one, x1), ustrip(one, x2))


# ==============================================================================


@register(lax.argmax_p)
def _argmax_p(
    operand: AbstractQuantity, *, axes: Any, index_dtype: Any
) -> AbstractQuantity:
    """Argmax of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from unxt import Quantity
    >>> x = Quantity([1, 2, 3], "m")
    >>> jnp.argmax(x)
    Quantity['length'](Array(2, dtype=int32), unit='m')

    >>> from unxt import UncheckedQuantity
    >>> x = UncheckedQuantity([1, 2, 3], "m")
    >>> jnp.argmax(x)
    UncheckedQuantity(Array(2, dtype=int32), unit='m')

    """
    return replace(operand, value=lax.argmax(operand.value, axes[0], index_dtype))


# ==============================================================================


@register(lax.argmin_p)
def _argmin_p(
    operand: AbstractQuantity, *, axes: Any, index_dtype: Any
) -> AbstractQuantity:
    """Argmin of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from unxt import Quantity
    >>> x = Quantity([1, 2, 3], "m")
    >>> jnp.argmin(x)
    Quantity['length'](Array(0, dtype=int32), unit='m')

    >>> from unxt import UncheckedQuantity
    >>> x = UncheckedQuantity([1, 2, 3], "m")
    >>> jnp.argmin(x)
    UncheckedQuantity(Array(0, dtype=int32), unit='m')

    """
    return replace(operand, value=lax.argmin(operand.value, axes[0], index_dtype))


# ==============================================================================


@register(lax.asin_p)
def _asin_p_aq(x: AbstractQuantity) -> AbstractQuantity:
    """Inverse sine of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(1, "")
    >>> jnp.asin(q)
    UncheckedQuantity(Array(1.5707964, dtype=float32, ...), unit='rad')

    """
    return type_np(x)(lax.asin(ustrip(one, x)), unit=radian)


@register(lax.asin_p)
def _asin_p_q(
    x: AbstractParametricQuantity["dimensionless"],
) -> AbstractParametricQuantity["angle"]:
    """Inverse sine of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from unxt import Quantity
    >>> q = Quantity(1, "")
    >>> jnp.asin(q)
    Quantity['angle'](Array(1.5707964, dtype=float32, ...), unit='rad')

    """
    return type_np(x)(lax.asin(ustrip(one, x)), unit=radian)


# ==============================================================================


@register(lax.asinh_p)
def _asinh_p_aq(x: AbstractQuantity) -> AbstractQuantity:
    """Inverse hyperbolic sine of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(2, "")
    >>> jnp.asinh(q)
    UncheckedQuantity(Array(1.4436355, dtype=float32, ...), unit='rad')

    """
    return type_np(x)(lax.asinh(ustrip(one, x)), unit=radian)


@register(lax.asinh_p)
def _asinh_p_q(
    x: AbstractParametricQuantity["dimensionless"],
) -> AbstractParametricQuantity["angle"]:
    """Inverse hyperbolic sine of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from unxt import Quantity
    >>> q = Quantity(2, "")
    >>> jnp.asinh(q)
    Quantity['angle'](Array(1.4436355, dtype=float32, ...), unit='rad')

    """
    return type_np(x)(lax.asinh(ustrip(one, x)), unit=radian)


# ==============================================================================


@register(lax.atan2_p)
def _atan2_p_aqaq(x: AbstractQuantity, y: AbstractQuantity) -> AbstractQuantity:
    """Arctangent2 of two abstract quantities.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from unxt import UncheckedQuantity
    >>> q1 = UncheckedQuantity(1, "m")
    >>> q2 = UncheckedQuantity(3, "m")
    >>> jnp.atan2(q1, q2)
    UncheckedQuantity(Array(0.32175055, dtype=float32, ...), unit='rad')

    """
    x, y = promote(x, y)  # e.g. Distance -> Quantity
    y_ = ustrip(x.unit, y)
    return type_np(x)(lax.atan2(x.value, y_), unit=radian)


@register(lax.atan2_p)
def _atan2_p_qq(
    x: AbstractParametricQuantity, y: AbstractParametricQuantity
) -> AbstractParametricQuantity["radian"]:
    """Arctangent2 of two quantities.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from unxt import Quantity
    >>> q1 = Quantity(1, "m")
    >>> q2 = Quantity(3, "m")
    >>> jnp.atan2(q1, q2)
    Quantity['angle'](Array(0.32175055, dtype=float32, ...), unit='rad')

    """
    x, y = promote(x, y)  # e.g. Distance -> Quantity
    y_ = ustrip(x.unit, y)
    return type_np(x)(lax.atan2(x.value, y_), unit=radian)


# ---------------------------


@register(lax.atan2_p)
def _atan2_p_vaq(x: ArrayLike, y: AbstractQuantity) -> AbstractQuantity:
    """Arctangent2 of a value and a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from unxt import UncheckedQuantity
    >>> x1 = jnp.asarray(1.0)
    >>> q2 = UncheckedQuantity(3.0, "")
    >>> jnp.atan2(x1, q2)
    UncheckedQuantity(Array(0.32175055, dtype=float32, ...), unit='rad')

    """
    y_ = ustrip(one, y)
    return type_np(y)(lax.atan2(x, y_), unit=radian)


@register(lax.atan2_p)
def _atan2_p_vq(
    x: ArrayLike, y: AbstractParametricQuantity["dimensionless"]
) -> AbstractParametricQuantity["angle"]:
    """Arctangent2 of a value and a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from unxt import Quantity
    >>> x1 = jnp.asarray(1.0)
    >>> q2 = Quantity(3.0, "")
    >>> jnp.atan2(x1, q2)
    Quantity['angle'](Array(0.32175055, dtype=float32, ...), unit='rad')

    """
    y_ = ustrip(one, y)
    return Quantity(lax.atan2(x, y_), unit=radian)


# ---------------------------


@register(lax.atan2_p)
def _atan2_p_aqv(x: AbstractQuantity, y: ArrayLike) -> AbstractQuantity:
    """Arctangent2 of a quantity and a value.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from unxt import UncheckedQuantity
    >>> q1 = UncheckedQuantity(1.0, "")
    >>> x2 = jnp.asarray(3.0)
    >>> jnp.atan2(q1, x2)
    UncheckedQuantity(Array(0.32175055, dtype=float32, ...), unit='rad')

    """
    x_ = ustrip(one, x)
    return type_np(x)(lax.atan2(x_, y), unit=radian)


@register(lax.atan2_p)
def _atan2_p_qv(
    x: AbstractParametricQuantity["dimensionless"], y: ArrayLike
) -> AbstractParametricQuantity["angle"]:
    """Arctangent2 of a quantity and a value.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from unxt import Quantity
    >>> q1 = Quantity(1.0, "")
    >>> x2 = jnp.asarray(3.0)
    >>> jnp.atan2(q1, x2)
    Quantity['angle'](Array(0.32175055, dtype=float32, ...), unit='rad')

    """
    x_ = ustrip(one, x)
    return type_np(x)(lax.atan2(x_, y), unit=radian)


# ==============================================================================


@register(lax.atan_p)
def _atan_p_aq(x: AbstractQuantity) -> AbstractQuantity:
    """Arctangent of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(1, "")
    >>> jnp.atan(q)
    UncheckedQuantity(Array(0.7853982, dtype=float32, ...), unit='rad')

    """
    return type_np(x)(lax.atan(ustrip(one, x)), unit=radian)


@register(lax.atan_p)
def _atan_p_q(
    x: AbstractParametricQuantity["dimensionless"],
) -> AbstractParametricQuantity["angle"]:
    """Arctangent of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from unxt import Quantity
    >>> q = Quantity(1, "")
    >>> jnp.atan(q)
    Quantity['angle'](Array(0.7853982, dtype=float32, ...), unit='rad')

    """
    return Quantity(lax.atan(ustrip(one, x)), unit=radian)


# ==============================================================================


@register(lax.atanh_p)
def _atanh_p_aq(x: AbstractQuantity) -> AbstractQuantity:
    """Inverse hyperbolic tangent of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(2, "")
    >>> jnp.atanh(q)
    UncheckedQuantity(Array(nan, dtype=float32, ...), unit='rad')

    """
    return type_np(x)(lax.atanh(ustrip(one, x)), unit=radian)


@register(lax.atanh_p)
def _atanh_p_q(
    x: AbstractParametricQuantity["dimensionless"],
) -> AbstractParametricQuantity["angle"]:
    """Inverse hyperbolic tangent of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from unxt import Quantity
    >>> q = Quantity(2, "")
    >>> jnp.atanh(q)
    Quantity['angle'](Array(nan, dtype=float32, ...), unit='rad')

    """
    return type_np(x)(lax.atanh(ustrip(one, x)), unit=radian)


# ==============================================================================


@register(lax.broadcast_in_dim_p)
def _broadcast_in_dim_p(
    operand: AbstractQuantity, *, shape: Any, broadcast_dimensions: Any
) -> AbstractQuantity:
    """Broadcast a quantity in a specific dimension."""
    return replace(
        operand, value=lax.broadcast_in_dim(operand.value, shape, broadcast_dimensions)
    )


# ==============================================================================


@register(lax.cbrt_p)
def _cbrt_p(x: AbstractQuantity) -> AbstractQuantity:
    """Cube root of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp

    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(8, "m3")
    >>> jnp.cbrt(q)
    UncheckedQuantity(Array(2., dtype=float32, ...), unit='m')

    >>> from unxt import Quantity
    >>> q = Quantity(8, "m3")
    >>> jnp.cbrt(q)
    Quantity['length'](Array(2., dtype=float32, ...), unit='m')

    """
    return type_np(x)(lax.cbrt(x.value), unit=x.unit ** (1 / 3))


# TODO: can this be done with promotion/conversion instead?
@register(lax.cbrt_p)
def _cbrt_p_d(x: AbstractDistance) -> Quantity:
    """Cube root of a distance.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from unxt import Distance
    >>> d = Distance(8, "m")
    >>> jnp.cbrt(d)
    Quantity['m1/3'](Array(2., dtype=float32, ...), unit='m(1/3)')

    """
    return Quantity(lax.cbrt(x.value), unit=x.unit ** (1 / 3))


# ==============================================================================


@register(lax.ceil_p)
def _ceil_p(x: AbstractQuantity) -> AbstractQuantity:
    """Ceiling of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(1.5, "m")
    >>> jnp.ceil(q)
    UncheckedQuantity(Array(2., dtype=float32, ...), unit='m')

    >>> from unxt import Quantity
    >>> q = Quantity(1.5, "m")
    >>> jnp.ceil(q)
    Quantity['length'](Array(2., dtype=float32, ...), unit='m')

    """
    return replace(x, value=lax.ceil(x.value))


# ==============================================================================


@register(lax.clamp_p)
def _clamp_p(
    min: AbstractQuantity, x: AbstractQuantity, max: AbstractQuantity
) -> AbstractQuantity:
    """Clamp a quantity between two other quantities.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import quaxed.lax as lax

    >>> from unxt import UncheckedQuantity
    >>> min = UncheckedQuantity(0, "m")
    >>> max = UncheckedQuantity(2, "m")
    >>> q = UncheckedQuantity([-1, 1, 3], "m")
    >>> lax.clamp(min, q, max)
    UncheckedQuantity(Array([0, 1, 2], dtype=int32), unit='m')

    >>> from unxt import Quantity
    >>> min = Quantity(0, "m")
    >>> max = Quantity(2, "m")
    >>> q = Quantity([-1, 1, 3], "m")
    >>> lax.clamp(min, q, max)
    Quantity['length'](Array([0, 1, 2], dtype=int32), unit='m')

    """
    return replace(
        x, value=lax.clamp(ustrip(x.unit, min), x.value, ustrip(x.unit, max))
    )


# ---------------------------


@register(lax.clamp_p)
def _clamp_p_vaqaq(
    min: ArrayLike, x: AbstractQuantity, max: AbstractQuantity
) -> AbstractQuantity:
    """Clamp a quantity between a value and another quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import quaxed.lax as lax

    >>> from unxt import UncheckedQuantity
    >>> min = jnp.asarray(0)
    >>> max = UncheckedQuantity(2, "")
    >>> q = UncheckedQuantity([-1, 1, 3], "")
    >>> lax.clamp(min, q, max)
    UncheckedQuantity(Array([0, 1, 2], dtype=int32), unit='')

    >>> from unxt import Quantity
    >>> min = jnp.asarray(0)
    >>> max = Quantity(2, "")
    >>> q = Quantity([-1, 1, 3], "")
    >>> lax.clamp(min, q, max)
    Quantity['dimensionless'](Array([0, 1, 2], dtype=int32), unit='')

    """
    return replace(x, value=lax.clamp(min, ustrip(one, x), ustrip(one, max)))


# ---------------------------


@register(lax.clamp_p)
def _clamp_p_aqvaq(
    min: AbstractQuantity, x: ArrayLike, max: AbstractQuantity
) -> ArrayLike:
    """Clamp a value between two quantities.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import quaxed.lax as lax

    >>> from unxt import UncheckedQuantity
    >>> min = UncheckedQuantity(0, "")
    >>> max = UncheckedQuantity(2, "")
    >>> x = jnp.asarray([-1, 1, 3])
    >>> lax.clamp(min, x, max)
    Array([0, 1, 2], dtype=int32)

    """
    return lax.clamp(ustrip(one, min), x, ustrip(one, max))


@register(lax.clamp_p)
def _clamp_p_qvq(
    min: AbstractParametricQuantity["dimensionless"],
    x: ArrayLike,
    max: AbstractParametricQuantity["dimensionless"],
) -> ArrayLike:
    """Clamp a value between two quantities.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import quaxed.lax as lax

    >>> from unxt import Quantity
    >>> min = Quantity(0, "")
    >>> max = Quantity(2, "")
    >>> x = jnp.asarray([-1, 1, 3])
    >>> lax.clamp(min, x, max)
    Array([0, 1, 2], dtype=int32)

    """
    return lax.clamp(ustrip(one, min), x, ustrip(one, max))


# ---------------------------


@register(lax.clamp_p)
def _clamp_p_aqaqv(
    min: AbstractQuantity, x: AbstractQuantity, max: ArrayLike
) -> AbstractQuantity:
    """Clamp a quantity between a quantity and a value.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import quaxed.lax as lax

    >>> from unxt import UncheckedQuantity
    >>> min = UncheckedQuantity(0, "")
    >>> max = jnp.asarray(2)
    >>> q = UncheckedQuantity([-1, 1, 3], "")
    >>> lax.clamp(min, q, max)
    UncheckedQuantity(Array([0, 1, 2], dtype=int32), unit='')

    """
    return replace(x, value=lax.clamp(ustrip(one, min), ustrip(one, x), max))


@register(lax.clamp_p)
def _clamp_p_qqv(
    min: AbstractParametricQuantity["dimensionless"],
    x: AbstractParametricQuantity["dimensionless"],
    max: ArrayLike,
) -> AbstractParametricQuantity["dimensionless"]:
    """Clamp a quantity between a quantity and a value.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import quaxed.lax as lax

    >>> from unxt import Quantity
    >>> min = Quantity(0, "")
    >>> max = jnp.asarray(2)
    >>> q = Quantity([-1, 1, 3], "")
    >>> lax.clamp(min, q, max)
    Quantity['dimensionless'](Array([0, 1, 2], dtype=int32), unit='')

    """
    return replace(x, value=lax.clamp(ustrip(one, min), ustrip(one, x), max))


# ==============================================================================


@register(lax.complex_p)
def _complex_p(x: AbstractQuantity, y: AbstractQuantity) -> AbstractQuantity:
    """Complex number from two quantities.

    Examples
    --------
    >>> from quaxed import lax
    >>> from unxt import UncheckedQuantity
    >>> x = UncheckedQuantity(1.0, "m")
    >>> y = UncheckedQuantity(2.0, "m")
    >>> lax.complex(x, y)
    UncheckedQuantity(Array(1.+2.j, dtype=complex64, ...), unit='m')

    >>> from unxt import Quantity
    >>> x = Quantity(1.0, "m")
    >>> y = Quantity(2.0, "m")
    >>> lax.complex(x, y)
    Quantity['length'](Array(1.+2.j, dtype=complex64, ...), unit='m')

    """
    x, y = promote(x, y)  # e.g. Distance -> Quantity
    y_ = ustrip(x.unit, y)
    return replace(x, value=lax.complex(x.value, y_))


# ==============================================================================
# Concatenation


@register(lax.concatenate_p)
def _concatenate_p_aq(*operands: AbstractQuantity, dimension: Any) -> AbstractQuantity:
    """Concatenate quantities.

    Examples
    --------
    >>> import quaxed.numpy as jnp

    >>> from unxt import UncheckedQuantity
    >>> q1 = UncheckedQuantity([1.0], "km")
    >>> q2 = UncheckedQuantity([2_000.0], "m")
    >>> jnp.concat([q1, q2])
    UncheckedQuantity(Array([1., 2.], dtype=float32), unit='km')

    >>> from unxt import Quantity
    >>> q1 = Quantity([1.0], "km")
    >>> q2 = Quantity([2_000.0], "m")
    >>> jnp.concat([q1, q2])
    Quantity['length'](Array([1., 2.], dtype=float32), unit='km')

    """
    operand0 = operands[0]
    units_ = operand0.unit
    return replace(
        operand0,
        value=lax.concatenate(
            [ustrip(units_, op) for op in operands], dimension=dimension
        ),
    )


# ---------------------------


@register(lax.concatenate_p)
def _concatenate_p_qnd(
    operand0: AbstractParametricQuantity["dimensionless"],
    *operands: AbstractParametricQuantity["dimensionless"] | ArrayLike,
    dimension: Any,
) -> AbstractParametricQuantity["dimensionless"]:
    """Concatenate quantities and arrays with dimensionless units.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from unxt import Quantity
    >>> theta = Quantity(45, "deg")
    >>> Rz = jnp.asarray([[jnp.cos(theta), -jnp.sin(theta), 0],
    ...                  [jnp.sin(theta), jnp.cos(theta),  0],
    ...                  [0,             0,              1]])
    >>> Rz
    Quantity[...](Array([[ 0.70710677, -0.70710677,  0.        ],
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
def _concatenate_p_vqnd(
    operand0: ArrayLike,
    *operands: AbstractParametricQuantity["dimensionless"],
    dimension: Any,
) -> AbstractParametricQuantity["dimensionless"]:
    """Concatenate quantities and arrays with dimensionless units.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from unxt import Quantity
    >>> theta = Quantity(45, "deg")
    >>> Rx = jnp.asarray([[1.0, 0.0,           0.0           ],
    ...                  [0.0, jnp.cos(theta), -jnp.sin(theta)],
    ...                  [0.0, jnp.sin(theta), jnp.cos(theta) ]])
    >>> Rx
    Quantity[...](Array([[ 1.        ,  0.        ,  0.        ],
                         [ 0.        ,  0.70710677, -0.70710677],
                         [ 0.        ,  0.70710677,  0.70710677]], dtype=float32),
                  unit='')

    """
    return Quantity(
        lax.concatenate(
            [
                (ustrip(one, op) if hasattr(op, "unit") else op)
                for op in (operand0, *operands)
            ],
            dimension=dimension,
        ),
        unit=one,
    )


# ==============================================================================


@register(lax.cond_p)  # TODO: implement
def _cond_p_q(index: AbstractQuantity, consts: AbstractQuantity) -> AbstractQuantity:
    raise NotImplementedError


@register(lax.cond_p)  # TODO: implement
def _cond_p_vq(
    index: ArrayLike, consts: AbstractQuantity, *, branches: Any
) -> AbstractQuantity:
    # print(branches)
    # raise AttributeError
    return lax.cond_p.bind(index, consts.value, branches=branches)


# ==============================================================================


@register(lax.conj_p)
def _conj_p(x: AbstractQuantity, *, input_dtype: Any) -> AbstractQuantity:
    """Conjugate of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp

    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(1 + 2j, "m")
    >>> jnp.conj(q)
    UncheckedQuantity(Array(1.-2.j, dtype=complex64, ...), unit='m')

    >>> from unxt import Quantity
    >>> q = Quantity(1 + 2j, "m")
    >>> jnp.conj(q)
    Quantity['length'](Array(1.-2.j, dtype=complex64, ...), unit='m')

    """
    del input_dtype  # TODO: use this?
    return replace(x, value=lax.conj(x.value))


# ==============================================================================


@register(lax.convert_element_type_p)
def _convert_element_type_p(
    operand: AbstractQuantity, **kwargs: Any
) -> AbstractQuantity:
    """Convert the element type of a quantity."""
    # TODO: examples
    return replace(
        operand, value=lax.convert_element_type_p.bind(operand.value, **kwargs)
    )


# ==============================================================================


@register(lax.copy_p)
def _copy_p(x: AbstractQuantity) -> AbstractQuantity:
    """Copy a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import quaxed.numpy as jnp

    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(1, "m")
    >>> jnp.copy(q)
    UncheckedQuantity(Array(1, dtype=int32, ...), unit='m')

    >>> from unxt import Quantity
    >>> q = Quantity(1, "m")
    >>> jnp.copy(q)
    Quantity['length'](Array(1, dtype=int32, ...), unit='m')

    """
    return replace(x, value=lax.copy_p.bind(x.value))


# ==============================================================================


@register(lax.cos_p)
def _cos_p_aq(x: AbstractQuantity) -> AbstractQuantity:
    """Cosine of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(1, "rad")
    >>> jnp.cos(q)
    UncheckedQuantity(Array(0.5403023, dtype=float32, ...), unit='')

    >>> q = UncheckedQuantity(1, "")
    >>> jnp.cos(q)
    UncheckedQuantity(Array(0.5403023, dtype=float32, ...), unit='')

    """
    return type_np(x)(lax.cos(_to_value_rad_or_one(x)), unit=one)


@register(lax.cos_p)
def _cos_p_q(
    x: AbstractParametricQuantity["angle"] | Quantity["dimensionless"],
) -> AbstractParametricQuantity["dimensionless"]:
    """Cosine of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from unxt import Quantity
    >>> q = Quantity(1, "rad")
    >>> jnp.cos(q)
    Quantity['dimensionless'](Array(0.5403023, dtype=float32, ...), unit='')

    >>> q = Quantity(1, "")
    >>> jnp.cos(q)
    Quantity['dimensionless'](Array(0.5403023, dtype=float32, ...), unit='')

    """
    return Quantity(lax.cos(_to_value_rad_or_one(x)), unit=one)


# ==============================================================================


@register(lax.cosh_p)
def _cosh_p_aq(x: AbstractQuantity) -> AbstractQuantity:
    """Cosine of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(1, "rad")
    >>> jnp.cosh(q)
    UncheckedQuantity(Array(1.5430806, dtype=float32, ...), unit='')

    >>> q = UncheckedQuantity(1, "")
    >>> jnp.cosh(q)
    UncheckedQuantity(Array(1.5430806, dtype=float32, ...), unit='')

    """
    return type_np(x)(lax.cosh(_to_value_rad_or_one(x)), unit=one)


@register(lax.cosh_p)
def _cosh_p_q(
    x: AbstractParametricQuantity["angle"] | Quantity["dimensionless"],
) -> AbstractParametricQuantity["dimensionless"]:
    """Cosine of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from unxt import Quantity
    >>> q = Quantity(1, "rad")
    >>> jnp.cosh(q)
    Quantity['dimensionless'](Array(1.5430806, dtype=float32, ...), unit='')

    >>> q = Quantity(1, "")
    >>> jnp.cosh(q)
    Quantity['dimensionless'](Array(1.5430806, dtype=float32, ...), unit='')

    """
    return type_np(x)(lax.cosh(_to_value_rad_or_one(x)), unit=one)


# ==============================================================================


@register(lax.cumlogsumexp_p)
def _cumlogsumexp_p(
    operand: AbstractQuantity, *, axis: Any, reverse: Any
) -> AbstractQuantity:
    """Cumulative log sum exp of a quantity.

    Examples
    --------
    >>> from quaxed import lax

    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity([-1.0, -2, -3], "")
    >>> lax.cumlogsumexp(q)
    UncheckedQuantity(Array([-1. , -0.6867383 , -0.59239405], dtype=float32), unit='')

    >>> from unxt import Quantity
    >>> q = Quantity([-1.0, -2, -3], "")
    >>> lax.cumlogsumexp(q)
    Quantity['dimensionless'](Array([-1. , -0.6867383 , -0.59239405], dtype=float32),
                              unit='')

    """
    # TODO: double check units make sense here.
    return replace(
        operand,
        value=lax.cumlogsumexp(operand.value, axis=axis, reverse=reverse),
    )


# ==============================================================================


@register(lax.cummax_p)
def _cummax_p(
    operand: AbstractQuantity, *, axis: Any, reverse: Any
) -> AbstractQuantity:
    """Cumulative maximum of a quantity.

    Examples
    --------
    >>> from quaxed import lax

    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity([1, 2, 1], "m")
    >>> lax.cummax(q)
    UncheckedQuantity(Array([1, 2, 2], dtype=int32), unit='m')

    >>> from unxt import Quantity
    >>> q = Quantity([1, 2, 1], "m")
    >>> lax.cummax(q)
    Quantity['length'](Array([1, 2, 2], dtype=int32), unit='m')

    """
    return replace(operand, value=lax.cummax(operand.value, axis=axis, reverse=reverse))


# ==============================================================================


@register(lax.cummin_p)
def _cummin_p(
    operand: AbstractQuantity, *, axis: Any, reverse: Any
) -> AbstractQuantity:
    """Cumulative maximum of a quantity.

    Examples
    --------
    >>> from quaxed import lax

    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity([2, 1, 3], "m")
    >>> lax.cummin(q)
    UncheckedQuantity(Array([2, 1, 1], dtype=int32), unit='m')

    >>> from unxt import Quantity
    >>> q = Quantity([2, 1, 3], "m")
    >>> lax.cummin(q)
    Quantity['length'](Array([2, 1, 1], dtype=int32), unit='m')

    """
    return replace(operand, value=lax.cummin(operand.value, axis=axis, reverse=reverse))


# ==============================================================================


@register(lax.cumprod_p)
def _cumprod_p(
    operand: AbstractQuantity, *, axis: Any, reverse: Any
) -> AbstractQuantity:
    """Cumulative product of a quantity.

    Examples
    --------
    >>> from quaxed import lax

    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity([1, 2, 3], "")
    >>> lax.cumprod(q)
    UncheckedQuantity(Array([1, 2, 6], dtype=int32), unit='')

    >>> from unxt import Quantity
    >>> q = Quantity([1, 2, 3], "")
    >>> lax.cumprod(q)
    Quantity['dimensionless'](Array([1, 2, 6], dtype=int32), unit='')

    """
    return replace(
        operand, value=lax.cumprod(ustrip(one, operand), axis=axis, reverse=reverse)
    )


# ==============================================================================


@register(lax.cumsum_p)
def _cumsum_p(
    operand: AbstractQuantity, *, axis: Any, reverse: Any
) -> AbstractQuantity:
    """Cumulative sum of a quantity.

    Examples
    --------
    >>> from quaxed import lax

    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity([1, 2, 3], "m")
    >>> lax.cumsum(q)
    UncheckedQuantity(Array([1, 3, 6], dtype=int32), unit='m')

    >>> from unxt import Quantity
    >>> q = Quantity([1, 2, 3], "m")
    >>> lax.cumsum(q)
    Quantity['length'](Array([1, 3, 6], dtype=int32), unit='m')

    """
    return replace(operand, value=lax.cumsum(operand.value, axis=axis, reverse=reverse))


# ==============================================================================


@register(lax.device_put_p)
def _device_put_p(x: AbstractQuantity, **kwargs: Any) -> AbstractQuantity:
    """Put a quantity on a device.

    Examples
    --------
    >>> from quaxed import device_put

    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(1, "m")
    >>> device_put(q)
    UncheckedQuantity(Array(1, dtype=int32, ...), unit='m')

    >>> from unxt import Quantity
    >>> q = Quantity(1, "m")
    >>> device_put(q)
    Quantity['length'](Array(1, dtype=int32, ...), unit='m')

    """
    return jt.map(lambda y: lax.device_put_p.bind(y, **kwargs), x)


# ==============================================================================


@register(lax.digamma_p)
def _digamma_p(x: AbstractQuantity) -> AbstractQuantity:
    """Digamma function of a quantity.

    Examples
    --------
    >>> from quaxed import lax

    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(1.0, "")
    >>> lax.digamma(q)
    UncheckedQuantity(Array(-0.5772154, dtype=float32, ...), unit='')

    >>> from unxt import Quantity
    >>> q = Quantity(1.0, "")
    >>> lax.digamma(q)
    Quantity['dimensionless'](Array(-0.5772154, dtype=float32, ...), unit='')

    """
    return replace(x, value=lax.digamma(ustrip(one, x)))


# ==============================================================================
# Division


@register(lax.div_p)
def _div_p_qq(x: AbstractQuantity, y: AbstractQuantity) -> AbstractQuantity:
    """Division of two quantities.

    Examples
    --------
    >>> import quaxed.numpy as jnp

    >>> from unxt import UncheckedQuantity
    >>> q1 = UncheckedQuantity(1, "m")
    >>> q2 = UncheckedQuantity(2, "s")
    >>> jnp.divide(q1, q2)
    UncheckedQuantity(Array(0.5, dtype=float32, ...), unit='m / s')
    >>> q1 / q2
    UncheckedQuantity(Array(0.5, dtype=float32, ...), unit='m / s')

    >>> from unxt import Quantity
    >>> q1 = Quantity(1, "m")
    >>> q2 = Quantity(2, "s")
    >>> jnp.divide(q1, q2)
    Quantity['speed'](Array(0.5, dtype=float32, ...), unit='m / s')
    >>> q1 / q2
    Quantity['speed'](Array(0.5, dtype=float32, ...), unit='m / s')

    """
    x, y = promote(x, y)
    unit = units(x.unit / y.unit)
    return type_np(x)(lax.div(x.value, y.value), unit=unit)


@register(lax.div_p)
def _div_p_vq(x: ArrayLike, y: AbstractQuantity) -> AbstractQuantity:
    """Division of an array by a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> x = jnp.asarray([1.0, 2, 3])

    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(2.0, "m")
    >>> jnp.divide(x, q)
    UncheckedQuantity(Array([0.5, 1. , 1.5], dtype=float32), unit='1 / m')
    >>> x / q
    UncheckedQuantity(Array([0.5, 1. , 1.5], dtype=float32), unit='1 / m')

    >>> from unxt import Quantity
    >>> q = Quantity(2.0, "m")
    >>> jnp.divide(x, q)
    Quantity['wavenumber'](Array([0.5, 1. , 1.5], dtype=float32), unit='1 / m')
    >>> x / q
    Quantity['wavenumber'](Array([0.5, 1. , 1.5], dtype=float32), unit='1 / m')

    """
    units_ = (1 / y.unit).unit  # TODO: better construction of the unit
    return type_np(y)(lax.div(x, y.value), unit=units_)


@register(lax.div_p)
def _div_p_qv(x: AbstractQuantity, y: ArrayLike) -> AbstractQuantity:
    """Division of a quantity by an array.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> y = jnp.asarray([1.0, 2, 3])

    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(6.0, "m")
    >>> jnp.divide(q, y)
    UncheckedQuantity(Array([6., 3., 2.], dtype=float32), unit='m')
    >>> q / y
    UncheckedQuantity(Array([6., 3., 2.], dtype=float32), unit='m')

    >>> from unxt import Quantity
    >>> q = Quantity(6.0, "m")
    >>> jnp.divide(q, y)
    Quantity['length'](Array([6., 3., 2.], dtype=float32), unit='m')
    >>> q / y
    Quantity['length'](Array([6., 3., 2.], dtype=float32), unit='m')

    """
    return replace(x, value=qlax.div(x.value, y))


# ==============================================================================


@register(lax.dot_general_p)
def _dot_general_jq(
    lhs: ArrayLike, rhs: AbstractQuantity, /, **kwargs: Any
) -> AbstractQuantity:
    """Dot product of an array and a quantity.

    >>> import quaxed.numpy as jnp
    >>> from unxt import Quantity, UncheckedQuantity

    >>> theta = jnp.pi / 4  # 45 degrees
    >>> Rz = jnp.asarray([[jnp.cos(theta), -jnp.sin(theta), 0],
    ...                   [jnp.sin(theta), jnp.cos(theta),  0],
    ...                   [0,              0,               1]])

    >>> q = UncheckedQuantity([1, 0, 0], "m")
    >>> jnp.linalg.matmul(Rz, q)
    UncheckedQuantity(Array([0.70710677, 0.70710677, 0. ], dtype=float32), unit='m')
    >>> Rz @ q
    UncheckedQuantity(Array([0.70710677, 0.70710677, 0. ], dtype=float32), unit='m')

    >>> q = Quantity([1, 0, 0], "m")
    >>> jnp.linalg.matmul(Rz, q)
    Quantity['length'](Array([0.70710677, 0.70710677, 0. ], dtype=float32), unit='m')
    >>> Rz @ q
    Quantity['length'](Array([0.70710677, 0.70710677, 0. ], dtype=float32), unit='m')

    """
    return type_np(rhs)(
        lax.dot_general_p.bind(lhs, rhs.value, **kwargs),
        unit=rhs.unit,
    )


@register(lax.dot_general_p)
def _dot_general_qq(
    lhs: AbstractQuantity, rhs: AbstractQuantity, /, **kwargs: Any
) -> AbstractQuantity:
    """Dot product of two quantities.

    Examples
    --------
    This is a dot product of two quantities.

    >>> import quaxed.numpy as jnp
    >>> from unxt import UncheckedQuantity

    >>> q1 = UncheckedQuantity([1, 2, 3], "m")
    >>> q2 = UncheckedQuantity([4, 5, 6], "m")
    >>> jnp.vecdot(q1, q2)
    UncheckedQuantity(Array(32, dtype=int32), unit='m2')
    >>> q1 @ q2
    UncheckedQuantity(Array(32, dtype=int32), unit='m2')

    >>> from unxt import Quantity

    >>> q1 = Quantity([1, 2, 3], "m")
    >>> q2 = Quantity([4, 5, 6], "m")
    >>> jnp.vecdot(q1, q2)
    Quantity['area'](Array(32, dtype=int32), unit='m2')
    >>> q1 @ q2
    Quantity['area'](Array(32, dtype=int32), unit='m2')

    >>> from unxt import Distance

    >>> q1 = Distance([1, 2, 3], "m")
    >>> q2 = Quantity([4, 5, 6], "m")
    >>> jnp.vecdot(q1, q2)
    Quantity['area'](Array(32, dtype=int32), unit='m2')
    >>> q1 @ q2
    Quantity['area'](Array(32, dtype=int32), unit='m2')

    This rule is also used by `jnp.matmul` for quantities.

    >>> Rz = jnp.asarray([[0, -1,  0],
    ...                   [1,  0,  0],
    ...                   [0,  0,  1]])
    >>> q = Quantity([1, 0, 0], "m")
    >>> Rz @ q
    Quantity['length'](Array([0, 1, 0], dtype=int32), unit='m')

    This uses `matmul` for quantities.

    >>> jnp.linalg.matmul(Rz, q)
    Quantity['length'](Array([0, 1, 0], dtype=int32), unit='m')

    """
    lhs, rhs = promote(lhs, rhs)
    return type_np(lhs)(
        lax.dot_general_p.bind(lhs.value, rhs.value, **kwargs),
        unit=lhs.unit * rhs.unit,
    )


@register(lax.dot_general_p)
def _dot_general_dd(
    lhs: AbstractDistance, rhs: AbstractDistance, /, **kwargs: Any
) -> Quantity:
    """Dot product of two Distances.

    Examples
    --------
    This is a dot product of two Distances.

    >>> import quaxed.numpy as jnp
    >>> from unxt import Distance

    >>> q1 = Distance([1, 2, 3], "m")
    >>> q2 = Distance([4, 5, 6], "m")
    >>> jnp.vecdot(q1, q2)
    Quantity['area'](Array(32, dtype=int32), unit='m2')
    >>> q1 @ q2
    Quantity['area'](Array(32, dtype=int32), unit='m2')

    This rule is also used by `jnp.matmul` for quantities.

    >>> Rz = jnp.asarray([[0, -1,  0],
    ...                   [1,  0,  0],
    ...                   [0,  0,  1]])
    >>> q = Quantity([1, 0, 0], "m")
    >>> Rz @ q
    Quantity['length'](Array([0, 1, 0], dtype=int32), unit='m')

    This uses `matmul` for quantities.

    >>> jnp.linalg.matmul(Rz, q)
    Quantity['length'](Array([0, 1, 0], dtype=int32), unit='m')

    """
    return Quantity(
        lax.dot_general_p.bind(lhs.value, rhs.value, **kwargs),
        unit=lhs.unit * rhs.unit,
    )


# ==============================================================================


@register(lax.eq_p)
def _eq_p_qq(x: AbstractQuantity, y: AbstractQuantity) -> ArrayLike:
    """Equality of two quantities.

    Examples
    --------
    >>> import quaxed.numpy as jnp

    >>> from unxt import UncheckedQuantity
    >>> q1 = UncheckedQuantity(1, "m")
    >>> q2 = UncheckedQuantity(1, "m")
    >>> jnp.equal(q1, q2)
    Array(True, dtype=bool, ...)
    >>> q1 == q2
    Array(True, dtype=bool, ...)

    >>> from unxt import Quantity
    >>> q1 = Quantity(1, "m")
    >>> q2 = Quantity(1, "m")
    >>> jnp.equal(q1, q2)
    Array(True, dtype=bool, ...)
    >>> q1 == q2
    Array(True, dtype=bool, ...)

    """
    try:
        yv = ustrip(x.unit, y)
    except UnitConversionError:
        return jnp.full(_bshape((x, y)), fill_value=False, dtype=bool)

    # re-dispatch on the values
    return qlax.eq(x.value, yv)


@register(lax.eq_p)
def _eq_p_vq(x: ArrayLike, y: AbstractQuantity, /) -> ArrayLike:
    """Equality of an array and a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> x = jnp.asarray([1.0, 2, 3])

    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(2.0, "")
    >>> jnp.equal(x, q)
    Array([False,  True, False], dtype=bool)

    >>> from unxt import Quantity
    >>> q = Quantity(2.0, "")
    >>> jnp.equal(x, q)
    Array([False,  True, False], dtype=bool)

    >>> q = Quantity(2.0, "m")
    >>> try: jnp.equal(x, q)
    ... except Exception as e: print(e)
    Array([False, False, False], dtype=bool)

    """
    try:
        yv = ustrip(one, y)
    except UnitConversionError:
        return jnp.full(_bshape((x, y)), fill_value=False, dtype=bool)

    # re-dispatch on the values
    return lax.eq(x, yv)


@register(lax.eq_p)
def _eq_p_aqv(x: AbstractQuantity, y: ArrayLike, /) -> ArrayLike:
    """Equality of an array and a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> y = jnp.asarray([1.0, 2, 3])

    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(2.0, "")
    >>> jnp.equal(q, y)
    Array([False,  True, False], dtype=bool)

    >>> q = UncheckedQuantity([3., 2, 1], "")
    >>> jnp.equal(q, y)
    Array([False,  True, False], dtype=bool)

    >>> q = UncheckedQuantity([3., 2, 1], "m")
    >>> jnp.equal(q, y)
    Array([False, False, False], dtype=bool)

    >>> from unxt import Quantity
    >>> q = Quantity(2.0, "")
    >>> jnp.equal(q, y)
    Array([False,  True, False], dtype=bool)

    >>> q = Quantity([3., 2, 1], "")
    >>> jnp.equal(q, y)
    Array([False,  True, False], dtype=bool)

    >>> q = Quantity([3., 2, 1], "m")
    >>> jnp.equal(q, y)
    Array([False, False, False], dtype=bool)

    Check against the special cases:

    >>> q == 0
    Array([False, False, False], dtype=bool)

    >>> q == jnp.inf
    Array([False, False, False], dtype=bool)

    """
    is_special = jnp.isscalar(y) and (jnp.isinf(y) | (y == 0))

    def special_case(_: Any, /) -> Array:
        return lax.eq(x.value, y)

    def regular_case(_: Any, /) -> Array:
        try:
            xv = ustrip(one, x)
        except UnitConversionError:
            return jnp.full(_bshape((x, y)), fill_value=False, dtype=bool)

        return lax.eq(xv, y)

    return lax.cond(is_special, special_case, regular_case, operand=None)


@register(lax.eq_p)
def _eq_p_aq0(x: AbstractQuantity, y: float | int) -> ArrayLike:
    """Equality of a quantity and 0."""
    y = eqx.error_if(
        y,
        y != 0,
        "Only zero is allowed for comparison with non-dimensionless quantities.",
    )
    return lax.eq(x.value, y)


# ==============================================================================


@register(lax.erf_inv_p)
def _erf_inv_p(x: AbstractQuantity) -> AbstractQuantity:
    """Inverse error function of a quantity.

    Examples
    --------
    >>> from quaxed import lax

    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(0.5, "")
    >>> lax.erf_inv(q)
    UncheckedQuantity(Array(0.47693628, dtype=float32, ...), unit='')

    >>> from unxt import Quantity
    >>> q = Quantity(0.5, "")
    >>> lax.erf_inv(q)
    Quantity['dimensionless'](Array(0.47693628, dtype=float32, ...), unit='')

    """
    # TODO: can this support non-dimensionless quantities?
    return replace(x, value=lax.erf_inv(ustrip(one, x)))


# ==============================================================================


@register(lax.erf_p)
def _erf_p(x: AbstractQuantity) -> AbstractQuantity:
    """Error function of a quantity.

    Examples
    --------
    >>> from quaxed import lax
    >>> from quax import quaxify

    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(0.5, "")
    >>> lax.erf(q)
    UncheckedQuantity(Array(0.5204999, dtype=float32, ...), unit='')

    >>> from unxt import Quantity
    >>> q = Quantity(0.5, "")
    >>> lax.erf(q)
    Quantity['dimensionless'](Array(0.5204999, dtype=float32, ...), unit='')

    """
    # TODO: can this support non-dimensionless quantities?
    return replace(x, value=lax.erf(ustrip(one, x)))


# ==============================================================================


@register(lax.erfc_p)
def _erfc_p(x: AbstractQuantity) -> AbstractQuantity:
    """Complementary error function of a quantity.

    Examples
    --------
    >>> from quaxed import lax

    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(0.5, "")
    >>> lax.erfc(q)
    UncheckedQuantity(Array(0.47950017, dtype=float32, ...), unit='')

    >>> from unxt import Quantity
    >>> q = Quantity(0.5, "")
    >>> lax.erfc(q)
    Quantity['dimensionless'](Array(0.47950017, dtype=float32, ...), unit='')

    """
    # TODO: can this support non-dimensionless quantities?
    return replace(x, value=lax.erfc(ustrip(one, x)))


# ==============================================================================


@register(lax.exp2_p)
def _exp2_p(x: AbstractQuantity) -> AbstractQuantity:
    """2^x of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp

    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(3, "")
    >>> jnp.exp2(q)
    UncheckedQuantity(Array(8., dtype=float32, ...), unit='')

    >>> from unxt import Quantity
    >>> q = Quantity(3, "")
    >>> jnp.exp2(q)
    Quantity['dimensionless'](Array(8., dtype=float32, ...), unit='')

    """
    return replace(x, value=lax.exp2(ustrip(one, x)))


# ==============================================================================


@register(lax.exp_p)
def _exp_p(x: AbstractQuantity) -> AbstractQuantity:
    """Exponential of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp

    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(1, "")
    >>> jnp.exp(q)
    UncheckedQuantity(Array(2.7182817, dtype=float32, ...), unit='')

    >>> from unxt import Quantity
    >>> q = Quantity(1, "")
    >>> jnp.exp(q)
    Quantity['dimensionless'](Array(2.7182817, dtype=float32, ...), unit='')

    Euler's crown jewel:

    >>> jnp.exp(Quantity(jnp.pi * 1j, "")) + 1
    Quantity['dimensionless'](Array(0.-8.742278e-08j, dtype=complex64, ...), unit='')

    Pretty close to zero!

    """
    # TODO: more meaningful error message.
    return replace(x, value=lax.exp(ustrip(one, x)))


# ==============================================================================


@register(lax.expm1_p)
def _expm1_p(x: AbstractQuantity) -> AbstractQuantity:
    """Exponential of a quantity minus 1.

    Examples
    --------
    >>> import quaxed.numpy as jnp

    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(0, "")
    >>> jnp.expm1(q)
    UncheckedQuantity(Array(0., dtype=float32, ...), unit='')

    >>> from unxt import Quantity
    >>> q = Quantity(0, "")
    >>> jnp.expm1(q)
    Quantity['dimensionless'](Array(0., dtype=float32, ...), unit='')

    """
    return replace(x, value=lax.expm1(ustrip(one, x)))


# ==============================================================================


@register(lax.fft_p)
def _fft_p(x: AbstractQuantity, *, fft_type: Any, fft_lengths: Any) -> AbstractQuantity:
    """Fast Fourier transform of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp

    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity([1, 2, 3], "")
    >>> jnp.fft.fft(q)
    UncheckedQuantity(Array([ 6. +0.j       , -1.5+0.8660254j, -1.5-0.8660254j],
                       dtype=complex64), unit='')

    >>> from unxt import Quantity
    >>> q = Quantity([1, 2, 3], "")
    >>> jnp.fft.fft(q)
    Quantity['dimensionless'](Array([ 6. +0.j       , -1.5+0.8660254j, -1.5-0.8660254j],
                                    dtype=complex64), unit='')

    """
    # TODO: what units can this support?
    return replace(
        x,
        value=lax.fft(ustrip(one, x), fft_type, fft_lengths),
    )


# ==============================================================================


@register(lax.floor_p)
def _floor_p(x: AbstractQuantity) -> AbstractQuantity:
    """Floor of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp

    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(1.5, "")
    >>> jnp.floor(q)
    UncheckedQuantity(Array(1., dtype=float32, ...), unit='')

    >>> from unxt import Quantity
    >>> q = Quantity(1.5, "")
    >>> jnp.floor(q)
    Quantity['dimensionless'](Array(1., dtype=float32, ...), unit='')

    """
    return replace(x, value=lax.floor(x.value))


# ==============================================================================


# used in `jnp.cross`
@register(lax.gather_p)
def _gather_p(
    operand: AbstractQuantity, start_indices: ArrayLike, **kwargs: Any
) -> AbstractQuantity:
    # TODO: examples
    return replace(
        operand, value=lax.gather_p.bind(operand.value, start_indices, **kwargs)
    )


# ==============================================================================


@register(lax.ge_p)
def _ge_p_qq(x: AbstractQuantity, y: AbstractQuantity) -> ArrayLike:
    """Greater than or equal to of two quantities.

    Examples
    --------
    >>> import quaxed.numpy as jnp

    >>> from unxt import UncheckedQuantity
    >>> q1 = UncheckedQuantity(1_001., "m")
    >>> q2 = UncheckedQuantity(1., "km")
    >>> jnp.greater_equal(q1, q2)
    Array(True, dtype=bool, ...)
    >>> q1 >= q2
    Array(True, dtype=bool, ...)

    >>> from unxt import Quantity
    >>> q1 = Quantity(1_001., "m")
    >>> q2 = Quantity(1., "km")
    >>> jnp.greater_equal(q1, q2)
    Array(True, dtype=bool, ...)
    >>> q1 >= q2
    Array(True, dtype=bool, ...)

    """
    try:
        yv = ustrip(x.unit, y)
    except UnitConversionError:
        return jnp.full(_bshape((x, y)), fill_value=False, dtype=bool)

    # re-dispatch on the value
    return qlax.ge(x.value, yv)


@register(lax.ge_p)
def _ge_p_vq(x: ArrayLike, y: AbstractQuantity, /) -> ArrayLike:
    """Greater than or equal to of an array and a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp

    >>> x = jnp.asarray(1_001.0)

    >>> from unxt import UncheckedQuantity
    >>> q2 = UncheckedQuantity(1., "")
    >>> jnp.greater_equal(x, q2)
    Array(True, dtype=bool, ...)

    >>> from unxt import Quantity
    >>> q2 = Quantity(1., "")
    >>> jnp.greater_equal(x, q2)
    Array(True, dtype=bool, ...)

    >>> q2 = Quantity(1., "m")
    >>> jnp.greater_equal(x, q2)
    Array(False, dtype=bool)

    """
    try:
        yv = ustrip(one, y)
    except UnitConversionError:
        return jnp.full(_bshape((x, y)), fill_value=False, dtype=bool)

    # re-dispatch on the value
    return qlax.ge(x, yv)


@register(lax.ge_p)
def _ge_p_qv(x: AbstractQuantity, y: ArrayLike, /) -> ArrayLike:
    """Greater than or equal to of a quantity and an array.

    Examples
    --------
    >>> import quaxed.numpy as jnp

    >>> y = jnp.asarray(0.9)

    >>> from unxt import UncheckedQuantity
    >>> q1 = UncheckedQuantity(1., "")
    >>> jnp.greater_equal(q1, y)
    Array(True, dtype=bool, ...)

    >>> from unxt import Quantity
    >>> q1 = Quantity(1., "")
    >>> jnp.greater_equal(q1, y)
    Array(True, dtype=bool, ...)

    >>> q1 = Quantity(1., "m")
    >>> jnp.greater_equal(q1, y)
    Array(False, dtype=bool)

    """
    try:
        xv = ustrip(one, x)
    except UnitConversionError:
        return jnp.full(_bshape((x, y)), fill_value=False, dtype=bool)

    # re-dispatch on the value
    return qlax.ge(xv, y)


# ==============================================================================


@register(lax.gt_p)
def _gt_p_qq(x: AbstractQuantity, y: AbstractQuantity) -> ArrayLike:
    """Greater than of two quantities.

    Examples
    --------
    >>> import quaxed.numpy as jnp

    >>> from unxt import UncheckedQuantity
    >>> q1 = UncheckedQuantity(1_001., "m")
    >>> q2 = UncheckedQuantity(1., "km")
    >>> jnp.greater_equal(q1, q2)
    Array(True, dtype=bool, ...)

    >>> from unxt import Quantity
    >>> q1 = Quantity(1_001., "m")
    >>> q2 = Quantity(1., "km")
    >>> jnp.greater_equal(q1, q2)
    Array(True, dtype=bool, ...)

    """
    try:
        yv = ustrip(x.unit, y)
    except UnitConversionError:
        return jnp.full(_bshape((x, y)), fill_value=False, dtype=bool)

    # re-dispatch on the value
    return qlax.gt(x.value, yv)


@register(lax.gt_p)
def _gt_p_vq(x: ArrayLike, y: AbstractQuantity) -> ArrayLike:
    """Greater than of an array and a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp

    >>> x = jnp.asarray(1_001.0)

    >>> from unxt import UncheckedQuantity
    >>> q2 = UncheckedQuantity(1., "")
    >>> jnp.greater_equal(x, q2)
    Array(True, dtype=bool, ...)

    >>> from unxt import Quantity
    >>> q2 = Quantity(1., "")
    >>> jnp.greater_equal(x, q2)
    Array(True, dtype=bool, ...)

    >>> q2 = Quantity(1., "m")
    >>> jnp.greater_equal(x, q2)
    Array(False, dtype=bool)

    """
    try:
        yv = ustrip(one, y)
    except UnitConversionError:
        return jnp.full(_bshape((x, y)), fill_value=False, dtype=bool)

    # re-dispatch on the value
    return qlax.gt(x, yv)


@register(lax.gt_p)
def _gt_p_qv(x: AbstractQuantity, y: ArrayLike) -> ArrayLike:
    """Greater than or equal to of a quantity and an array.

    Examples
    --------
    >>> import quaxed.numpy as jnp

    >>> y = jnp.asarray(0.9)

    >>> from unxt import UncheckedQuantity
    >>> q1 = UncheckedQuantity(1., "")
    >>> jnp.greater_equal(q1, y)
    Array(True, dtype=bool, ...)

    >>> from unxt import Quantity
    >>> q1 = Quantity(1., "")
    >>> jnp.greater_equal(q1, y)
    Array(True, dtype=bool, ...)

    >>> q1 = Quantity(1., "m")
    >>> jnp.greater_equal(q1, y)
    Array(False, dtype=bool)

    """
    try:
        xv = ustrip(one, x)
    except UnitConversionError:
        return jnp.full(_bshape((x, y)), fill_value=False, dtype=bool)

    # re-dispatch on the value
    return qlax.gt(xv, y)


# ==============================================================================


@register(lax.igamma_p)
def _igamma_p(a: AbstractQuantity, x: AbstractQuantity) -> AbstractQuantity:
    """Regularized incomplete gamma function of a and x.

    Examples
    --------
    >>> from quaxed import lax

    >>> from unxt import UncheckedQuantity
    >>> a = UncheckedQuantity(1.0, "")
    >>> x = UncheckedQuantity(1.0, "")
    >>> lax.igamma(a, x)
    UncheckedQuantity(Array(0.6321202, dtype=float32, ...), unit='')

    >>> from unxt import Quantity
    >>> a = Quantity(1.0, "")
    >>> x = Quantity(1.0, "")
    >>> lax.igamma(a, x)
    Quantity['dimensionless'](Array(0.6321202, dtype=float32, ...), unit='')

    """
    return replace(x, value=lax.igamma(ustrip(one, a), ustrip(one, x)))


# ==============================================================================


@register(lax.imag_p)
def _imag_p(x: AbstractQuantity) -> AbstractQuantity:
    return replace(x, value=lax.imag(x.value))


# ==============================================================================


@register(lax.integer_pow_p)
def _integer_pow_p(x: AbstractQuantity, *, y: Any) -> AbstractQuantity:
    """Integer power of a quantity.

    Examples
    --------
    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(2, "m")
    >>> q ** 3
    UncheckedQuantity(Array(8, dtype=int32, ...), unit='m3')

    >>> from unxt import Quantity
    >>> q = Quantity(2, "m")
    >>> q ** 3
    Quantity['volume'](Array(8, dtype=int32, ...), unit='m3')

    """
    return type_np(x)(value=lax.integer_pow(x.value, y), unit=x.unit**y)


@register(lax.integer_pow_p)
def _integer_pow_p_d(x: AbstractDistance, *, y: Any) -> Quantity:
    """Integer power of a Distance.

    Examples
    --------
    >>> from unxt import Distance
    >>> q = Distance(2, "m")
    >>> q ** 3
    Quantity['volume'](Array(8, dtype=int32, ...), unit='m3')

    """
    return Quantity(value=lax.integer_pow(x.value, y), unit=x.unit**y)


# ==============================================================================


@register(lax.is_finite_p)
def _is_finite_p(x: AbstractQuantity) -> ArrayLike:
    """Check if a quantity is finite.

    Examples
    --------
    >>> import quaxed.numpy as jnp

    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(1, "m")
    >>> jnp.isfinite(q)
    array(True)
    >>> q = UncheckedQuantity(float('inf'), "m")
    >>> jnp.isfinite(q)
    Array(False, dtype=bool, ...)

    >>> from unxt import Quantity
    >>> q = Quantity(1, "m")
    >>> jnp.isfinite(q)
    array(True)
    >>> q = Quantity(float('inf'), "m")
    >>> jnp.isfinite(q)
    Array(False, dtype=bool, ...)

    """
    return lax.is_finite(x.value)


# ==============================================================================


@register(lax.le_p)
def _le_p_qq(x: AbstractQuantity, y: AbstractQuantity, /) -> ArrayLike:
    """Less than or equal to of two quantities.

    Examples
    --------
    >>> import quaxed.numpy as jnp

    >>> from unxt import UncheckedQuantity
    >>> q1 = UncheckedQuantity(1_001., "m")
    >>> q2 = UncheckedQuantity(1., "km")
    >>> jnp.less_equal(q1, q2)
    Array(False, dtype=bool, ...)

    >>> from unxt import Quantity
    >>> q1 = Quantity(1_001., "m")
    >>> q2 = Quantity(1., "km")
    >>> jnp.less_equal(q1, q2)
    Array(False, dtype=bool, ...)

    """
    try:
        yv = ustrip(x.unit, y)
    except UnitConversionError:
        return jnp.full(_bshape((x, y)), fill_value=False, dtype=bool)

    # re-dispatch on the values
    return qlax.le(x.value, yv)


@register(lax.le_p)
def _le_p_vq(x: ArrayLike, y: AbstractQuantity, /) -> ArrayLike:
    """Less than or equal to of an array and a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp

    >>> x1 = jnp.asarray(1.001)

    >>> from unxt import UncheckedQuantity
    >>> q2 = UncheckedQuantity(1., "")
    >>> jnp.less_equal(x1, q2)
    Array(False, dtype=bool, ...)

    >>> from unxt import Quantity
    >>> q2 = Quantity(1., "")
    >>> jnp.less_equal(x1, q2)
    Array(False, dtype=bool, ...)

    >>> q2 = Quantity(1., "m")
    >>> jnp.less_equal(x1, q2)
    Array(False, dtype=bool)

    """
    try:
        yv = ustrip(one, y)
    except UnitConversionError:
        return jnp.full(_bshape((x, y)), fill_value=False, dtype=bool)

    # re-dispatch on the values
    return qlax.le(x, yv)


@register(lax.le_p)
def _le_p_qv(x: AbstractQuantity, y: ArrayLike, /) -> ArrayLike:
    """Less than or equal to of a quantity and an array.

    Examples
    --------
    >>> import quaxed.numpy as jnp

    >>> y1 = jnp.asarray(0.9)

    >>> from unxt import UncheckedQuantity
    >>> q1 = UncheckedQuantity(1., "")
    >>> jnp.less_equal(q1, y1)
    Array(False, dtype=bool, ...)

    >>> from unxt import Quantity
    >>> q1 = Quantity(1., "")
    >>> jnp.less_equal(q1, y1)
    Array(False, dtype=bool, ...)

    >>> q1 = Quantity(1., "m")
    >>> jnp.less_equal(q1, y1)
    Array(False, dtype=bool)

    """
    try:
        xv = ustrip(one, x)
    except UnitConversionError:
        return jnp.full(_bshape((x, y)), fill_value=False, dtype=bool)

    # re-dispatch on the values
    return qlax.le(xv, y)


# ==============================================================================


@register(lax.lgamma_p)
def _lgamma_p(x: AbstractQuantity) -> AbstractQuantity:
    """Log-gamma function of a quantity.

    Examples
    --------
    >>> import quaxed.scipy as jsp

    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(3, "")
    >>> jsp.special.gammaln(q)
    UncheckedQuantity(Array(0.6931474, dtype=float32, ...), unit='')

    >>> from unxt import Quantity
    >>> q = Quantity(3, "")
    >>> jsp.special.gammaln(q)
    Quantity['dimensionless'](Array(0.6931474, dtype=float32, ...), unit='')

    """
    # TODO: are there any units that this can support?
    return replace(x, value=lax.lgamma(ustrip(one, x)))


# ==============================================================================


@register(lax.log1p_p)
def _log1p_p(x: AbstractQuantity) -> AbstractQuantity:
    return replace(x, value=lax.log1p(ustrip(one, x)))


# ==============================================================================


@register(lax.log_p)
def _log_p(x: AbstractQuantity) -> AbstractQuantity:
    return replace(x, value=lax.log(ustrip(one, x)))


# ==============================================================================


@register(lax.logistic_p)
def _logistic_p(x: AbstractQuantity) -> AbstractQuantity:
    return replace(x, value=lax.logistic(ustrip(one, x)))


# ==============================================================================


@register(lax.lt_p)
def _lt_p_qq(x: AbstractQuantity, y: AbstractQuantity, /) -> ArrayLike:
    """Less than of two quantities.

    Examples
    --------
    >>> import quaxed.numpy as jnp

    `UncheckedQuantity`:

    >>> from unxt import UncheckedQuantity

    >>> x = UncheckedQuantity(1., "km")
    >>> y = UncheckedQuantity(2000., "m")
    >>> x < y
    Array(True, dtype=bool, ...)

    >>> jnp.less(x, y)
    Array(True, dtype=bool, ...)

    >>> x = UncheckedQuantity([1., 2, 3], "km")
    >>> x < y
    Array([ True, False, False], dtype=bool)

    >>> jnp.less(x, y)
    Array([ True, False, False], dtype=bool)

    `Quantity`:

    >>> from unxt import Quantity

    >>> x = Quantity(1., "km")
    >>> y = Quantity(2000., "m")
    >>> x < y
    Array(True, dtype=bool, ...)

    >>> jnp.less(x, y)
    Array(True, dtype=bool, ...)

    >>> x = Quantity([1., 2, 3], "km")
    >>> x < y
    Array([ True, False, False], dtype=bool)

    >>> jnp.less(x, y)
    Array([ True, False, False], dtype=bool)

    """
    try:
        yv = ustrip(x.unit, y)
    except UnitConversionError:
        return jnp.full(_bshape((x, y)), fill_value=False, dtype=bool)

    # re-dispatch on the values
    return qlax.lt(x.value, yv)


@register(lax.lt_p)
def _lt_p_vq(x: ArrayLike, y: AbstractQuantity, /) -> ArrayLike:
    """Less than of an array and a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp

    `UncheckedQuantity`:

    >>> from unxt import UncheckedQuantity

    >>> x = jnp.asarray([1.])
    >>> y = UncheckedQuantity(2., "")

    Note that `JAX` does support passing the comparison to
    a different class.

    >>> x < y
    Array([ True], dtype=bool)

    But we can always use the `jnp.less` function.

    >>> jnp.less(x, y)
    Array([ True], dtype=bool)

    >>> x = jnp.asarray([1., 2, 3])
    >>> jnp.less(x, y)
    Array([ True, False, False], dtype=bool)

    `Quantity`:

    >>> from unxt import Quantity

    >>> y = Quantity(2., "")
    >>> jnp.less(x, y)
    Array([ True, False, False], dtype=bool)

    >>> x = jnp.asarray([1., 2, 3])
    >>> jnp.less(x, y)
    Array([ True, False, False], dtype=bool)

    >>> y = Quantity(2., "m")
    >>> jnp.less(x, y)
    Array([False, False, False], dtype=bool)

    """
    try:
        yv = ustrip(one, y)
    except UnitConversionError:
        return jnp.full(_bshape((x, y)), fill_value=False, dtype=bool)

    # re-dispatch on the values
    return qlax.lt(x, yv)


@register(lax.lt_p)
def _lt_p_qv(x: AbstractQuantity, y: ArrayLike, /) -> ArrayLike:
    """Compare a unitless Quantity to a value.

    Examples
    --------
    >>> import quaxed.numpy as jnp

    `UncheckedQuantity`:

    >>> from unxt import UncheckedQuantity

    >>> x = UncheckedQuantity(1, "")
    >>> y = 2
    >>> x < y
    Array(True, dtype=bool, ...)

    >>> jnp.less(x, y)
    Array(True, dtype=bool, ...)

    >>> x = UncheckedQuantity([1, 2, 3], "")
    >>> x < y
    Array([ True, False, False], dtype=bool)

    >>> jnp.less(x, y)
    Array([ True, False, False], dtype=bool)

    `Quantity`:

    >>> from unxt import Quantity

    >>> x = Quantity(1, "")
    >>> y = 2
    >>> x < y
    Array(True, dtype=bool, ...)

    >>> jnp.less(x, y)
    Array(True, dtype=bool, ...)

    >>> x = Quantity([1, 2, 3], "")
    >>> x < y
    Array([ True, False, False], dtype=bool)

    >>> jnp.less(x, y)
    Array([ True, False, False], dtype=bool)

    >>> x = Quantity([1, 2], "m")
    >>> jnp.less(x, y)
    Array([False, False], dtype=bool)

    """
    try:
        xv = ustrip(one, x)
    except UnitConversionError:
        return jnp.full(_bshape((x, y)), fill_value=False, dtype=bool)

    # re-dispatch on the values
    return qlax.lt(xv, y)


# ==============================================================================


@register(lax.max_p)
def _max_p_qq(x: AbstractQuantity, y: AbstractQuantity, /) -> AbstractQuantity:
    """Maximum of two quantities.

    Examples
    --------
    >>> import quaxed.numpy as jnp

    >>> from unxt import UncheckedQuantity
    >>> q1 = UncheckedQuantity(1, "m")
    >>> q2 = UncheckedQuantity(2, "m")
    >>> jnp.maximum(q1, q2)
    UncheckedQuantity(Array(2, dtype=int32, ...), unit='m')

    >>> from unxt import Quantity
    >>> q1 = Quantity(1, "m")
    >>> q2 = Quantity(2, "m")
    >>> jnp.maximum(q1, q2)
    Quantity['length'](Array(2, dtype=int32, ...), unit='m')

    """
    yv = ustrip(x.unit, y)
    return replace(x, value=qlax.max(x.value, yv))


@register(lax.max_p)
def _max_p_vq(x: ArrayLike, y: AbstractQuantity) -> AbstractQuantity:
    """Maximum of an array and quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp

    >>> from unxt import UncheckedQuantity
    >>> x = jnp.array([1.])
    >>> q2 = UncheckedQuantity(2, "")
    >>> jnp.maximum(x, q2)
    UncheckedQuantity(Array([2.], dtype=float32), unit='')

    >>> from unxt import Quantity
    >>> q2 = Quantity(2, "")
    >>> jnp.maximum(x, q2)
    Quantity['dimensionless'](Array([2.], dtype=float32), unit='')

    """
    yv = ustrip(one, y)
    return replace(y, value=lax.max(x, yv))


@register(lax.max_p)
def _max_p_qv(x: AbstractQuantity, y: ArrayLike) -> AbstractQuantity:
    """Maximum of an array and quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp

    >>> from unxt import UncheckedQuantity
    >>> q1 = UncheckedQuantity(2, "")
    >>> y = jnp.array([1.])
    >>> jnp.maximum(q1, y)
    UncheckedQuantity(Array([2.], dtype=float32), unit='')

    >>> from unxt import Quantity
    >>> q1 = Quantity(2, "")
    >>> jnp.maximum(q1, y)
    Quantity['dimensionless'](Array([2.], dtype=float32), unit='')

    """
    xv = ustrip(one, x)
    return replace(x, value=lax.max(xv, y))


# ==============================================================================


@register(lax.min_p)
def _min_p_qq(x: AbstractQuantity, y: AbstractQuantity) -> AbstractQuantity:
    return replace(x, value=lax.min(x.value, ustrip(x.unit, y)))


@register(lax.min_p)
def _min_p_vq(x: ArrayLike, y: AbstractQuantity) -> AbstractQuantity:
    return replace(y, value=lax.min(x, ustrip(one, y)))


@register(lax.min_p)
def _min_p_qv(x: AbstractQuantity, y: ArrayLike) -> AbstractQuantity:
    return replace(x, value=lax.min(ustrip(one, x), y))


# ==============================================================================
# Multiplication


@register(lax.mul_p)
def _mul_p_qq(x: AbstractQuantity, y: AbstractQuantity) -> AbstractQuantity:
    unit = units(x.unit * y.unit)
    return type_np(x)(lax.mul(x.value, y.value), unit=unit)


@register(lax.mul_p)
def _mul_p_vq(x: ArrayLike, y: AbstractQuantity) -> AbstractQuantity:
    return replace(y, value=lax.mul(x, y.value))


@register(lax.mul_p)
def _mul_p_qv(x: AbstractQuantity, y: ArrayLike) -> AbstractQuantity:
    return replace(x, value=lax.mul(x.value, y))


# ==============================================================================


@register(lax.ne_p)
def _ne_p_qq(x: AbstractQuantity, y: AbstractQuantity) -> ArrayLike:
    """Inequality of two quantities.

    Examples
    --------
    >>> import quaxed.numpy as jnp

    >>> from unxt import UncheckedQuantity
    >>> q1 = UncheckedQuantity(1, "m")
    >>> q2 = UncheckedQuantity(2, "m")
    >>> jnp.not_equal(q1, q2)
    Array(True, dtype=bool, ...)
    >>> q1 != q2
    Array(True, dtype=bool, ...)

    >>> q2 = UncheckedQuantity(1, "m")
    >>> jnp.not_equal(q1, q2)
    Array(False, dtype=bool, ...)
    >>> q1 != q2
    Array(False, dtype=bool, ...)

    >>> from unxt import Quantity
    >>> q1 = Quantity(1, "m")
    >>> q2 = Quantity(2, "m")
    >>> jnp.not_equal(q1, q2)
    Array(True, dtype=bool, ...)
    >>> q1 != q2
    Array(True, dtype=bool, ...)

    >>> q2 = Quantity(1, "m")
    >>> jnp.not_equal(q1, q2)
    Array(False, dtype=bool, ...)
    >>> q1 != q2
    Array(False, dtype=bool, ...)

    """
    try:
        yv = ustrip(x.unit, y)
    except UnitConversionError:
        return jnp.full(_bshape((x, y)), fill_value=True, dtype=bool)

    # re-dispatch on the value
    return qlax.ne(x.value, yv)


@register(lax.ne_p)
def _ne_p_vq(x: ArrayLike, y: AbstractQuantity) -> ArrayLike:
    """Inequality of an array and a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp

    >>> from unxt import UncheckedQuantity
    >>> x = 1
    >>> q2 = UncheckedQuantity(2, "")
    >>> jnp.not_equal(x, q2)
    Array(True, dtype=bool, ...)
    >>> x != q2
    Array(True, dtype=bool, ...)

    >>> q2 = UncheckedQuantity(1, "")
    >>> jnp.not_equal(x, q2)
    Array(False, dtype=bool, ...)
    >>> x != q2
    Array(False, dtype=bool, ...)

    >>> from unxt import Quantity
    >>> x = Quantity(1, "")
    >>> q2 = Quantity(2, "")
    >>> jnp.not_equal(x, q2)
    Array(True, dtype=bool, ...)
    >>> x != q2
    Array(True, dtype=bool, ...)

    >>> q2 = Quantity(1, "")
    >>> jnp.not_equal(x, q2)
    Array(False, dtype=bool, ...)
    >>> x != q2
    Array(False, dtype=bool, ...)

    """
    try:
        yv = ustrip(one, y)
    except UnitConversionError:
        return jnp.full(_bshape((x, y)), fill_value=True, dtype=bool)

    # re-dispatch on the value
    return qlax.ne(x, yv)


@register(lax.ne_p)
def _ne_p_qv(x: AbstractQuantity, y: ArrayLike) -> ArrayLike:
    # special-case for scalar value=0, unit=one
    if jnp.shape(y) == () and y == 0:  # TODO: proper jax
        return lax.ne(x.value, y)

    try:
        xv = ustrip(one, x)
    except UnitConversionError:
        return jnp.full(_bshape((x, y)), fill_value=True, dtype=bool)

    return lax.ne(xv, y)


# @register(lax.ne_p)
# def _ne_p_qv(x: AbstractParametricQuantity, y: ArrayLike) -> ArrayLike:
#     return lax.


# ==============================================================================


@register(lax.neg_p)
def _neg_p(x: AbstractQuantity) -> AbstractQuantity:
    return replace(x, value=lax.neg(x.value))


# ==============================================================================


@register(lax.pow_p)
def _pow_p_qq(
    x: AbstractQuantity, y: AbstractParametricQuantity["dimensionless"]
) -> AbstractQuantity:
    y_: Array = ustrip(one, y)
    y0 = y_[(0,) * y_.ndim]
    y_ = eqx.error_if(y_, any(y_ != y0), "power must be a scalar")
    return type_np(x)(value=lax.pow(x.value, y0), unit=x.unit**y0)


@register(lax.pow_p)
def _pow_p_qf(x: AbstractQuantity, y: ArrayLike) -> AbstractQuantity:
    return type_np(x)(value=lax.pow(x.value, y), unit=x.unit**y)


@register(lax.pow_p)
def _pow_p_vq(
    x: ArrayLike, y: AbstractParametricQuantity["dimensionless"]
) -> AbstractQuantity:
    return replace(y, value=lax.pow(x, y.value))


@register(lax.pow_p)
def _pow_p_d(x: AbstractDistance, y: ArrayLike) -> Quantity:
    """Power of a Distance by redispatching to Quantity.

    Examples
    --------
    >>> import math
    >>> from unxt import Distance

    >>> q1 = Distance(10.0, "m")
    >>> y = 3.0
    >>> q1 ** y
    Quantity['volume'](Array(1000., dtype=float32, ...), unit='m3')

    """
    return Quantity(x.value, x.unit) ** y  # TODO: better call to power


# ==============================================================================


@register(lax.real_p)
def _real_p(x: AbstractQuantity) -> AbstractQuantity:
    return replace(x, value=lax.real(x.value))


# ==============================================================================


@register(lax.reduce_and_p)
def _reduce_and_p(operand: AbstractQuantity, *, axes: Sequence[int]) -> Any:
    return lax.reduce_and_p.bind(operand.value, axes=tuple(axes))


# ==============================================================================


@register(lax.reduce_max_p)
def _reduce_max_p(operand: AbstractQuantity, *, axes: Axes) -> AbstractQuantity:
    return replace(operand, value=lax.reduce_max_p.bind(operand.value, axes=axes))


# ==============================================================================


@register(lax.reduce_min_p)
def _reduce_min_p(operand: AbstractQuantity, *, axes: Axes) -> AbstractQuantity:
    return replace(operand, value=lax.reduce_min_p.bind(operand.value, axes=axes))


# ==============================================================================


@register(lax.reduce_or_p)
def _reduce_or_p(operand: AbstractQuantity, *, axes: Axes) -> AbstractQuantity:
    return type_np(operand)(lax.reduce_or_p.bind(operand.value, axes=axes), unit=one)


# ==============================================================================


@register(lax.reduce_prod_p)
def _reduce_prod_p(operand: AbstractQuantity, *, axes: Axes) -> AbstractQuantity:
    return type_np(operand)(
        lax.reduce_prod_p.bind(operand.value, axes=axes),
        unit=operand.unit ** prod(operand.shape[ax] for ax in axes),
    )


# ==============================================================================


@register(lax.reduce_sum_p)
def _reduce_sum_p(operand: AbstractQuantity, *, axes: Axes) -> AbstractQuantity:
    return replace(operand, value=lax.reduce_sum_p.bind(operand.value, axes=axes))


# ==============================================================================


@register(lax.rem_p)
def _rem_p_qq(x: AbstractQuantity, y: AbstractQuantity) -> AbstractQuantity:
    """Remainder of two quantities.

    Examples
    --------
    >>> from unxt import UncheckedQuantity

    >>> q1 = UncheckedQuantity(10, "m")
    >>> q2 = UncheckedQuantity(3, "m")
    >>> q1 % q2
    UncheckedQuantity(Array(1, dtype=int32, ...), unit='m')

    >>> from unxt import Quantity
    >>> q1 = Quantity(10, "m")
    >>> q2 = Quantity(3, "m")
    >>> q1 % q2
    Quantity['length'](Array(1, dtype=int32, ...), unit='m')

    >>> from unxt import Distance
    >>> q1 = Distance(10, "m")
    >>> q2 = Quantity(3, "m")
    >>> q1 % q2
    Distance(Array(1, dtype=int32, ...), unit='m')

    """
    return replace(x, value=lax.rem(x.value, ustrip(x.unit, y)))


@register(lax.rem_p)
def _rem_p_uqv(x: Quantity["dimensionless"], y: ArrayLike) -> Quantity["dimensionless"]:
    """Remainder of two quantities.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from unxt import Quantity

    >>> q1 = Quantity(10, "")
    >>> q2 = jnp.array(3)
    >>> q1 % q2
    Quantity['dimensionless'](Array(1, dtype=int32, ...), unit='')

    """
    return replace(x, value=lax.rem(x.value, y))


# ==============================================================================


@register(lax.reshape_p)
def _reshape_p(
    operand: AbstractQuantity, *, new_sizes: Any, dimensions: Any
) -> AbstractQuantity:
    return replace(operand, value=lax.reshape(operand.value, new_sizes, dimensions))


# ==============================================================================


@register(lax.rev_p)
def _rev_p(operand: AbstractQuantity, *, dimensions: Any) -> AbstractQuantity:
    return replace(operand, value=lax.rev(operand.value, dimensions))


# ==============================================================================


@register(lax.round_p)
def _round_p(x: AbstractQuantity, *, rounding_method: Any) -> AbstractQuantity:
    return replace(x, value=lax.round(x.value, rounding_method))


# ==============================================================================


@register(lax.rsqrt_p)
def _rsqrt_p(x: AbstractQuantity) -> AbstractQuantity:
    return type_np(x)(lax.rsqrt(x.value), unit=x.unit ** (-1 / 2))


# ==============================================================================


@register(lax.scatter_add_p)
def _scatter_add_p_qvq(
    operand: AbstractQuantity,
    scatter_indices: ArrayLike,
    updates: AbstractQuantity,
    **kwargs: Any,
) -> AbstractQuantity:
    """Scatter-add operator."""
    return replace(
        operand,
        value=lax.scatter_add_p.bind(
            operand.value, scatter_indices, ustrip(operand.units, updates), **kwargs
        ),
    )


@register(lax.scatter_add_p)
def _scatter_add_p_vvq(
    operand: ArrayLike,
    scatter_indices: ArrayLike,
    updates: AbstractQuantity,
    **kwargs: Any,
) -> AbstractQuantity:
    """Scatter-add operator between an Array and a Quantity.

    This is an interesting case where the Quantity is the `updates` and the Array
    is the `operand`. For some reason when doing a ``scatter_add`` between two
    Quantity objects an intermediate Array operand is created. Therefore we
    need to pretend that the Array has the same units as the `updates`.
    """
    return replace(
        updates,
        value=lax.scatter_add_p.bind(operand, scatter_indices, updates.value, **kwargs),
    )


# ==============================================================================


@register(lax.select_n_p)
def _select_n_p(which: AbstractQuantity, *cases: AbstractQuantity) -> AbstractQuantity:
    unit = cases[0].unit
    cases_ = (ustrip(unit, case) for case in cases)
    return type_np(which)(lax.select_n(ustrip(one, which), *cases_), unit=unit)


@register(lax.select_n_p)
def _select_n_p_vq(
    which: AbstractQuantity, case0: AbstractQuantity, case1: ArrayLike
) -> AbstractQuantity:
    # encountered from jnp.hypot
    unit = case0.unit
    return type_np(which)(
        lax.select_n(ustrip(one, which), ustrip(unit, case0), case1), unit=unit
    )


@register(lax.select_n_p)
def _select_n_p_jjq(
    which: ArrayLike, case0: ArrayLike, case1: AbstractQuantity
) -> AbstractQuantity:
    # Used by a `jnp.linalg.trace`
    return replace(case1, value=lax.select_n(which, case0, case1.value))


@register(lax.select_n_p)
def _select_n_p_jqj(
    which: ArrayLike, case0: AbstractQuantity, case1: ArrayLike
) -> AbstractQuantity:
    # Used by a `triu`
    return replace(case0, value=lax.select_n(which, case0.value, case1))


@register(lax.select_n_p)
def _select_n_p_jqq(
    which: ArrayLike, case0: AbstractQuantity, case1: AbstractQuantity
) -> AbstractQuantity:
    # used by `jnp.hypot`
    unit = case0.unit
    return replace(case0, value=lax.select_n(which, case0.value, ustrip(unit, case1)))


# ==============================================================================


@register(lax.sign_p)
def _sign_p(x: AbstractQuantity) -> ArrayLike:
    """Sign of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp

    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(10, "m")
    >>> jnp.sign(q)
    Array(1, dtype=int32, ...)

    >>> from unxt import Quantity
    >>> q = Quantity(10, "m")
    >>> jnp.sign(q)
    Array(1, dtype=int32, ...)

    """
    return lax.sign(x.value)


# ==============================================================================


@register(lax.sin_p)
def _sin_p(x: AbstractQuantity) -> AbstractQuantity:
    return type_np(x)(lax.sin(_to_value_rad_or_one(x)), unit=one)


# ==============================================================================


@register(lax.sinh_p)
def _sinh_p(x: AbstractQuantity) -> AbstractQuantity:
    return type_np(x)(lax.sinh(_to_value_rad_or_one(x)), unit=one)


# ==============================================================================


@register(lax.slice_p)
def _slice_p(
    operand: AbstractQuantity, *, start_indices: Any, limit_indices: Any, strides: Any
) -> AbstractQuantity:
    return replace(
        operand,
        value=lax.slice_p.bind(
            operand.value,
            start_indices=start_indices,
            limit_indices=limit_indices,
            strides=strides,
        ),
    )


# ==============================================================================


# Called by `argsort`
@register(lax.sort_p)
def _sort_p_two_operands(
    operand0: AbstractQuantity,
    operand1: ArrayLike,
    *,
    dimension: int,
    is_stable: bool,
    num_keys: int,
) -> tuple[AbstractQuantity, ArrayLike]:
    out0, out1 = lax.sort_p.bind(
        operand0.value,
        operand1,
        dimension=dimension,
        is_stable=is_stable,
        num_keys=num_keys,
    )
    return (replace(operand0, value=out0), out1)


# Called by `sort`
@register(lax.sort_p)
def _sort_p_one_operand(
    operand: AbstractQuantity, *, dimension: int, is_stable: bool, num_keys: int
) -> tuple[AbstractQuantity]:
    (out,) = lax.sort_p.bind(
        operand.value, dimension=dimension, is_stable=is_stable, num_keys=num_keys
    )
    return (type_np(operand)(out, unit=operand.unit),)


# ==============================================================================


@register(lax.sqrt_p)
def _sqrt_p_q(x: AbstractQuantity) -> AbstractQuantity:
    """Square root of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp

    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(9, "m")
    >>> jnp.sqrt(q)
    UncheckedQuantity(Array(3., dtype=float32, ...), unit='m(1/2)')

    >>> from unxt import Quantity
    >>> q = Quantity(9, "m")
    >>> jnp.sqrt(q)
    Quantity['m0.5'](Array(3., dtype=float32, ...), unit='m(1/2)')

    """
    # Apply sqrt to the value and adjust the unit
    return type_np(x)(lax.sqrt(x.value), unit=x.unit ** (1 / 2))


@register(lax.sqrt_p)
def _sqrt_p_d(x: AbstractDistance) -> Quantity:
    """Square root of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp

    >>> from unxt import Distance
    >>> q = Distance(9, "m")
    >>> jnp.sqrt(q)
    Quantity['m0.5'](Array(3., dtype=float32, ...), unit='m(1/2)')

    >>> from unxt import Parallax
    >>> q = Parallax(9, "mas")
    >>> jnp.sqrt(q)
    Quantity['rad0.5'](Array(3., dtype=float32, ...), unit='mas(1/2)')

    """
    # Promote to something that supports sqrt units.
    return Quantity(lax.sqrt(x.value), unit=x.unit ** (1 / 2))


# ==============================================================================


@register(lax.squeeze_p)
def _squeeze_p(x: AbstractQuantity, *, dimensions: Any) -> AbstractQuantity:
    return type_np(x)(lax.squeeze(x.value, dimensions), unit=x.unit)


# ==============================================================================


@register(lax.stop_gradient_p)
def _stop_gradient_p(x: AbstractQuantity) -> AbstractQuantity:
    return replace(x, value=lax.stop_gradient(x.value))


# ==============================================================================
# Subtraction


@register(lax.sub_p)
def _sub_p_qq(x: AbstractQuantity, y: AbstractQuantity) -> AbstractQuantity:
    """Subtract two quantities.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from unxt import UncheckedQuantity

    >>> q1 = UncheckedQuantity(1.0, "km")
    >>> q2 = UncheckedQuantity(500.0, "m")
    >>> jnp.subtract(q1, q2)
    UncheckedQuantity(Array(0.5, dtype=float32, ...), unit='km')
    >>> q1 - q2
    UncheckedQuantity(Array(0.5, dtype=float32, ...), unit='km')

    >>> from unxt import Quantity
    >>> q1 = Quantity(1.0, "km")
    >>> q2 = Quantity(500.0, "m")
    >>> jnp.subtract(q1, q2)
    Quantity['length'](Array(0.5, dtype=float32, ...), unit='km')
    >>> q1 - q2
    Quantity['length'](Array(0.5, dtype=float32, ...), unit='km')

    >>> from unxt import Distance
    >>> d1 = Distance(1.0, "km")
    >>> d2 = Distance(500.0, "m")
    >>> jnp.subtract(d1, d2)
    Distance(Array(0.5, dtype=float32, ...), unit='km')

    >>> from unxt import Parallax
    >>> p1 = Parallax(1.0, "mas")
    >>> p2 = Parallax(500.0, "uas")
    >>> jnp.subtract(p1, p2)
    Parallax(Array(0.5, dtype=float32, ...), unit='mas')

    >>> from unxt import DistanceModulus
    >>> dm1 = DistanceModulus(1.0, "mag")
    >>> dm2 = DistanceModulus(500.0, "mag")
    >>> jnp.subtract(dm1, dm2)
    DistanceModulus(Array(-499., dtype=float32, ...), unit='mag')

    """
    return replace(x, value=lax.sub(ustrip(x.unit, x), ustrip(x.unit, y)))


@register(lax.sub_p)
def _sub_p_vq(x: ArrayLike, y: AbstractQuantity) -> AbstractQuantity:
    """Subtract a quantity from an array.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from unxt import UncheckedQuantity

    >>> x = 1_000
    >>> q = UncheckedQuantity(500.0, "")
    >>> jnp.subtract(x, q)
    UncheckedQuantity(Array(500., dtype=float32, ...), unit='')

    >>> x - q
    UncheckedQuantity(Array(500., dtype=float32, ...), unit='')

    >>> from unxt import Quantity
    >>> q = Quantity(500.0, "")
    >>> jnp.subtract(x, q)
    Quantity['dimensionless'](Array(500., dtype=float32, ...), unit='')

    >>> x - q
    Quantity['dimensionless'](Array(500., dtype=float32, ...), unit='')

    """
    y = uconvert(one, y)
    return replace(y, value=qlax.sub(x, y.value))


@register(lax.sub_p)
def _sub_p_qv(x: AbstractQuantity, y: ArrayLike) -> AbstractQuantity:
    """Subtract an array from a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from unxt import UncheckedQuantity

    >>> q = UncheckedQuantity(500.0, "")
    >>> y = 1_000
    >>> jnp.subtract(q, y)
    UncheckedQuantity(Array(-500., dtype=float32, ...), unit='')

    >>> q - y
    UncheckedQuantity(Array(-500., dtype=float32, ...), unit='')

    >>> from unxt import Quantity
    >>> q = Quantity(500.0, "")
    >>> jnp.subtract(q, y)
    Quantity['dimensionless'](Array(-500., dtype=float32, ...), unit='')

    >>> q - y
    Quantity['dimensionless'](Array(-500., dtype=float32, ...), unit='')

    """
    x = uconvert(one, x)
    return replace(x, value=qlax.sub(x.value, y))


# ==============================================================================


@register(lax.tan_p)
def _tan_p(x: AbstractQuantity) -> AbstractQuantity:
    return type_np(x)(lax.tan(_to_value_rad_or_one(x)), unit=one)


# TODO: figure out a promotion alternative that works in general
@register(lax.tan_p)
def _tan_p_d(x: AbstractDistance) -> Quantity["dimensionless"]:
    return Quantity(lax.tan(_to_value_rad_or_one(x)), unit=one)


# ==============================================================================


@register(lax.tanh_p)
def _tanh_p(x: AbstractQuantity) -> AbstractQuantity:
    return type_np(x)(lax.tanh(_to_value_rad_or_one(x)), unit=one)


# ==============================================================================


@register(lax.transpose_p)
def _transpose_p(operand: AbstractQuantity, *, permutation: Any) -> AbstractQuantity:
    return replace(
        operand, value=lax.transpose_p.bind(operand.value, permutation=permutation)
    )
