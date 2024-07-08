# pylint: disable=import-error, too-many-lines

__all__: list[str] = []

from collections.abc import Callable, Sequence
from dataclasses import replace
from math import prod
from typing import Any, TypeAlias, TypeVar

import equinox as eqx
import jax
from astropy.units import (  # pylint: disable=no-name-in-module
    Unit,
    UnitBase,
    dimensionless_unscaled as one,
    radian,
)
from jax import lax, numpy as jnp
from jax._src.ad_util import add_any_p
from jax._src.lax.lax import DotDimensionNumbers, DTypeLike, PrecisionLike
from jax._src.lax.slicing import GatherDimensionNumbers, GatherScatterMode
from jax._src.typing import Shape
from jax.core import Primitive
from jaxtyping import Array, ArrayLike
from plum import promote
from quax import register as register_

from .base import AbstractQuantity, can_convert_unit
from .base_parametric import AbstractParametricQuantity
from .core import Quantity
from .distance import AbstractDistance
from .utils import type_unparametrized as type_np

T = TypeVar("T")

Axes: TypeAlias = tuple[int, ...]
UnitClasses: TypeAlias = UnitBase


def register(primitive: Primitive) -> Callable[[T], T]:
    """:func`quax.register`, but makes mypy happy."""
    return register_(primitive)


def _to_value_rad_or_one(q: AbstractQuantity) -> ArrayLike:
    return (
        q.to_units_value(radian)
        if can_convert_unit(q.unit, radian)
        else q.to_units_value(one)
    )


################################################################################
# Registering Primitives

# ==============================================================================


@register(lax.abs_p)
def _abs_p(x: AbstractQuantity) -> AbstractQuantity:
    """Absolute value of a quantity.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> from unxt import Quantity
    >>> q = Quantity(-1, "m")
    >>> xp.abs(q)
    Quantity['length'](Array(1, dtype=int32, ...), unit='m')
    >>> abs(q)
    Quantity['length'](Array(1, dtype=int32, ...), unit='m')

    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(-1, "m")
    >>> xp.abs(q)
    UncheckedQuantity(Array(1, dtype=int32, ...), unit='m')
    >>> abs(q)
    UncheckedQuantity(Array(1, dtype=int32, ...), unit='m')

    >>> from unxt import Distance
    >>> d = Distance(-1, "m")
    >>> xp.abs(d)
    Distance(Array(1, dtype=int32, ...), unit='m')

    >>> from unxt import Parallax
    >>> p = Parallax(-1, "mas", check_negative=False)
    >>> xp.abs(p)
    Parallax(Array(1, dtype=int32, ...), unit='mas')

    >>> from unxt import DistanceModulus
    >>> dm = DistanceModulus(-1, "mag")
    >>> xp.abs(dm)
    DistanceModulus(Array(1, dtype=int32, weak_type=True), unit='mag')

    """
    return replace(x, value=lax.abs(x.value))


# ==============================================================================


@register(lax.acos_p)
def _acos_p_aq(x: AbstractQuantity) -> AbstractQuantity:
    """Inverse cosine of a quantity.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(-1, "")
    >>> xp.acos(q)
    UncheckedQuantity(Array(3.1415927, dtype=float32), unit='rad')

    >>> from unxt import Quantity
    >>> q = Quantity(-1, "")
    >>> xp.acos(q)
    Quantity['angle'](Array(3.1415927, dtype=float32), unit='rad')

    """
    x_ = x.to_units_value(one)
    return type_np(x)(value=lax.acos(x_), unit=radian)


# ==============================================================================


@register(lax.acosh_p)
def _acosh_p_aq(x: AbstractQuantity) -> AbstractQuantity:
    """Inverse hyperbolic cosine of a quantity.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(2.0, "")
    >>> xp.acosh(q)
    UncheckedQuantity(Array(1.316958, dtype=float32, ...), unit='rad')

    >>> from unxt import Quantity
    >>> q = Quantity(2.0, "")
    >>> xp.acosh(q)
    Quantity['angle'](Array(1.316958, dtype=float32, ...), unit='rad')

    """
    x_ = x.to_units_value(one)
    return type_np(x)(value=lax.acosh(x_), unit=radian)


# ==============================================================================
# Addition


@register(lax.add_p)
def _add_p_aqaq(x: AbstractQuantity, y: AbstractQuantity) -> AbstractQuantity:
    """Add two quantities.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> from unxt import UncheckedQuantity

    >>> q1 = UncheckedQuantity(1.0, "km")
    >>> q2 = UncheckedQuantity(500.0, "m")
    >>> xp.add(q1, q2)
    UncheckedQuantity(Array(1.5, dtype=float32, ...), unit='km')
    >>> q1 + q2
    UncheckedQuantity(Array(1.5, dtype=float32, ...), unit='km')

    >>> from unxt import Quantity
    >>> q1 = Quantity(1.0, "km")
    >>> q2 = Quantity(500.0, "m")
    >>> xp.add(q1, q2)
    Quantity['length'](Array(1.5, dtype=float32, ...), unit='km')
    >>> q1 + q2
    Quantity['length'](Array(1.5, dtype=float32, ...), unit='km')

    >>> from unxt import Distance
    >>> d1 = Distance(1.0, "km")
    >>> d2 = Distance(500.0, "m")
    >>> xp.add(d1, d2)
    Distance(Array(1.5, dtype=float32, ...), unit='km')

    >>> from unxt import Parallax
    >>> p1 = Parallax(1.0, "mas")
    >>> p2 = Parallax(500.0, "uas")
    >>> xp.add(p1, p2)
    Parallax(Array(1.5, dtype=float32, ...), unit='mas')

    >>> from unxt import DistanceModulus
    >>> dm1 = DistanceModulus(1.0, "mag")
    >>> dm2 = DistanceModulus(500.0, "mag")
    >>> xp.add(dm1, dm2)
    DistanceModulus(Array(501., dtype=float32, ...), unit='mag')

    """
    return replace(x, value=lax.add(x.value, y.to_units_value(x.unit)))


@register(lax.add_p)
def _add_p_vaq(x: ArrayLike, y: AbstractQuantity) -> AbstractQuantity:
    """Add a value and a quantity.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> x1 = xp.asarray(500.0)

    >>> from unxt import UncheckedQuantity
    >>> q2 = UncheckedQuantity(1.0, "km")
    >>> try: xp.add(x1, q2)
    ... except Exception as e: print(e)
    Cannot add a non-quantity and quantity.
    >>> try: x1 + q2
    ... except Exception as e: print(e)
    Cannot add a non-quantity and quantity.
    >>> q2 = UncheckedQuantity(100.0, "")
    >>> xp.add(x1, q2)
    UncheckedQuantity(Array(600., dtype=float32, ...), unit='')
    >>> x1 + q2
    UncheckedQuantity(Array(600., dtype=float32, ...), unit='')

    >>> from unxt import Quantity
    >>> x1 = xp.asarray(500.0)
    >>> q2 = Quantity(1.0, "km")
    >>> try: x1 + q2
    ... except Exception as e: print(e)
    Cannot add a non-quantity and quantity.
    >>> q2 = Quantity(100.0, "")
    >>> xp.add(x1, q2)
    Quantity['dimensionless'](Array(600., dtype=float32, ...), unit='')
    >>> x1 + q2
    Quantity['dimensionless'](Array(600., dtype=float32, ...), unit='')

    """
    y = eqx.error_if(y, y.unit != one, "Cannot add a non-quantity and quantity.")
    return replace(y, value=lax.add(x, y.to_units_value(one)))


@register(lax.add_p)
def _add_p_aqv(x: AbstractQuantity, y: ArrayLike) -> AbstractQuantity:
    """Add a quantity and a value.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> y = xp.asarray(500.0)

    >>> from unxt import UncheckedQuantity
    >>> q1 = UncheckedQuantity(1.0, "km")
    >>> try: xp.add(q1, y)
    ... except Exception as e: print(e)
    Cannot add a quantity and a non-quantity.
    >>> try: q1 + y
    ... except Exception as e: print(e)
    Cannot add a quantity and a non-quantity.

    >>> q1 = UncheckedQuantity(100.0, "")
    >>> xp.add(q1, y)
    UncheckedQuantity(Array(600., dtype=float32, ...), unit='')
    >>> q1 + y
    UncheckedQuantity(Array(600., dtype=float32, ...), unit='')

    >>> from unxt import Quantity
    >>> q1 = Quantity(1.0, "km")
    >>> try: xp.add(q1, y)
    ... except Exception as e: print(e)
    Cannot add a quantity and a non-quantity.
    >>> try: q1 + y
    ... except Exception as e: print(e)
    Cannot add a quantity and a non-quantity.

    >>> q1 = Quantity(100.0, "")
    >>> xp.add(q1, y)
    Quantity[...](Array(600., dtype=float32, ...), unit='')
    >>> q1 + y
    Quantity[...](Array(600., dtype=float32, ...), unit='')

    """
    x = eqx.error_if(x, x.unit != one, "Cannot add a quantity and a non-quantity.")
    return replace(x, value=lax.add(x.to_units_value(one), y))


# ==============================================================================


@register(add_any_p)
def _add_any_p(
    x: AbstractParametricQuantity, y: AbstractParametricQuantity
) -> AbstractParametricQuantity:
    """Add two quantities using the ``jax._src.ad_util.add_any_p``."""
    return replace(x, value=add_any_p.bind(x.value, y.value))


# ==============================================================================


@register(lax.after_all_p)
def _after_all_p() -> AbstractQuantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.all_gather_p)
def _all_gather_p() -> AbstractQuantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.all_to_all_p)
def _all_to_all_p() -> AbstractQuantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.and_p)
def _and_p_aq(x1: AbstractQuantity, x2: AbstractQuantity, /) -> ArrayLike:
    """Bitwise AND of two quantities.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> from unxt import UncheckedQuantity
    >>> x1 = UncheckedQuantity(1, "")
    >>> x2 = UncheckedQuantity(2, "")
    >>> xp.bitwise_and(x1, x2)
    Array(0, dtype=int32, ...)

    >>> from unxt import Quantity
    >>> x1 = Quantity(1, "")
    >>> x2 = Quantity(2, "")
    >>> xp.bitwise_and(x1, x2)
    Array(0, dtype=int32, ...)

    """
    return lax.and_p.bind(x1.to_units_value(one), x2.to_units_value(one))


# ==============================================================================


@register(lax.approx_top_k_p)
def _approx_top_k_p() -> AbstractQuantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.argmax_p)
def _argmax_p(
    operand: AbstractQuantity, *, axes: Any, index_dtype: Any
) -> AbstractQuantity:
    """Argmax of a quantity.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> from unxt import Quantity
    >>> x = Quantity([1, 2, 3], "m")
    >>> xp.argmax(x)
    Quantity['length'](Array(2, dtype=int32), unit='m')

    >>> from unxt import UncheckedQuantity
    >>> x = UncheckedQuantity([1, 2, 3], "m")
    >>> xp.argmax(x)
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
    >>> import quaxed.array_api as xp
    >>> from unxt import Quantity
    >>> x = Quantity([1, 2, 3], "m")
    >>> xp.argmin(x)
    Quantity['length'](Array(0, dtype=int32), unit='m')

    >>> from unxt import UncheckedQuantity
    >>> x = UncheckedQuantity([1, 2, 3], "m")
    >>> xp.argmin(x)
    UncheckedQuantity(Array(0, dtype=int32), unit='m')

    """
    return replace(operand, value=lax.argmin(operand.value, axes[0], index_dtype))


# ==============================================================================


@register(lax.asin_p)
def _asin_p_aq(x: AbstractQuantity) -> AbstractQuantity:
    """Inverse sine of a quantity.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(1, "")
    >>> xp.asin(q)
    UncheckedQuantity(Array(1.5707964, dtype=float32), unit='rad')

    """
    return type_np(x)(lax.asin(x.to_units_value(one)), unit=radian)


@register(lax.asin_p)
def _asin_p_q(
    x: AbstractParametricQuantity["dimensionless"],
) -> AbstractParametricQuantity["angle"]:
    """Inverse sine of a quantity.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> from unxt import Quantity
    >>> q = Quantity(1, "")
    >>> xp.asin(q)
    Quantity['angle'](Array(1.5707964, dtype=float32), unit='rad')

    """
    return type_np(x)(lax.asin(x.to_units_value(one)), unit=radian)


# ==============================================================================


@register(lax.asinh_p)
def _asinh_p_aq(x: AbstractQuantity) -> AbstractQuantity:
    """Inverse hyperbolic sine of a quantity.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(2, "")
    >>> xp.asinh(q)
    UncheckedQuantity(Array(1.4436355, dtype=float32), unit='rad')

    """
    return type_np(x)(lax.asinh(x.to_units_value(one)), unit=radian)


@register(lax.asinh_p)
def _asinh_p_q(
    x: AbstractParametricQuantity["dimensionless"],
) -> AbstractParametricQuantity["angle"]:
    """Inverse hyperbolic sine of a quantity.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> from unxt import Quantity
    >>> q = Quantity(2, "")
    >>> xp.asinh(q)
    Quantity['angle'](Array(1.4436355, dtype=float32), unit='rad')

    """
    return type_np(x)(lax.asinh(x.to_units_value(one)), unit=radian)


# ==============================================================================


@register(lax.atan2_p)
def _atan2_p_aqaq(x: AbstractQuantity, y: AbstractQuantity) -> AbstractQuantity:
    """Arctangent2 of two abstract quantities.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> from unxt import UncheckedQuantity
    >>> q1 = UncheckedQuantity(1, "m")
    >>> q2 = UncheckedQuantity(3, "m")
    >>> xp.atan2(q1, q2)
    UncheckedQuantity(Array(0.32175055, dtype=float32), unit='rad')

    """
    x, y = promote(x, y)  # e.g. Distance -> Quantity
    y_ = y.to_units_value(x.unit)
    return type_np(x)(lax.atan2(x.value, y_), unit=radian)


@register(lax.atan2_p)
def _atan2_p_qq(
    x: AbstractParametricQuantity, y: AbstractParametricQuantity
) -> AbstractParametricQuantity["radian"]:
    """Arctangent2 of two quantities.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> from unxt import Quantity
    >>> q1 = Quantity(1, "m")
    >>> q2 = Quantity(3, "m")
    >>> xp.atan2(q1, q2)
    Quantity['angle'](Array(0.32175055, dtype=float32), unit='rad')

    """
    x, y = promote(x, y)  # e.g. Distance -> Quantity
    y_ = y.to_units_value(x.unit)
    return type_np(x)(lax.atan2(x.value, y_), unit=radian)


# ---------------------------


@register(lax.atan2_p)
def _atan2_p_vaq(x: ArrayLike, y: AbstractQuantity) -> AbstractQuantity:
    """Arctangent2 of a value and a quantity.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> from unxt import UncheckedQuantity
    >>> x1 = xp.asarray(1.0)
    >>> q2 = UncheckedQuantity(3.0, "")
    >>> xp.atan2(x1, q2)
    UncheckedQuantity(Array(0.32175055, dtype=float32, ...), unit='rad')

    """
    y_ = y.to_units_value(one)
    return type_np(y)(lax.atan2(x, y_), unit=radian)


@register(lax.atan2_p)
def _atan2_p_vq(
    x: ArrayLike, y: AbstractParametricQuantity["dimensionless"]
) -> AbstractParametricQuantity["angle"]:
    """Arctangent2 of a value and a quantity.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> from unxt import Quantity
    >>> x1 = xp.asarray(1.0)
    >>> q2 = Quantity(3.0, "")
    >>> xp.atan2(x1, q2)
    Quantity['angle'](Array(0.32175055, dtype=float32, ...), unit='rad')

    """
    y_ = y.to_units_value(one)
    return Quantity(lax.atan2(x, y_), unit=radian)


# ---------------------------


@register(lax.atan2_p)
def _atan2_p_aqv(x: AbstractQuantity, y: ArrayLike) -> AbstractQuantity:
    """Arctangent2 of a quantity and a value.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> from unxt import UncheckedQuantity
    >>> q1 = UncheckedQuantity(1.0, "")
    >>> x2 = xp.asarray(3.0)
    >>> xp.atan2(q1, x2)
    UncheckedQuantity(Array(0.32175055, dtype=float32, ...), unit='rad')

    """
    x_ = x.to_units_value(one)
    return type_np(x)(lax.atan2(x_, y), unit=radian)


@register(lax.atan2_p)
def _atan2_p_qv(
    x: AbstractParametricQuantity["dimensionless"], y: ArrayLike
) -> AbstractParametricQuantity["angle"]:
    """Arctangent2 of a quantity and a value.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> from unxt import Quantity
    >>> q1 = Quantity(1.0, "")
    >>> x2 = xp.asarray(3.0)
    >>> xp.atan2(q1, x2)
    Quantity['angle'](Array(0.32175055, dtype=float32, ...), unit='rad')

    """
    x_ = x.to_units_value(one)
    return type_np(x)(lax.atan2(x_, y), unit=radian)


# ==============================================================================


@register(lax.atan_p)
def _atan_p_aq(x: AbstractQuantity) -> AbstractQuantity:
    """Arctangent of a quantity.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(1, "")
    >>> xp.atan(q)
    UncheckedQuantity(Array(0.7853982, dtype=float32), unit='rad')

    """
    return type_np(x)(lax.atan(x.to_units_value(one)), unit=radian)


@register(lax.atan_p)
def _atan_p_q(
    x: AbstractParametricQuantity["dimensionless"],
) -> AbstractParametricQuantity["angle"]:
    """Arctangent of a quantity.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> from unxt import Quantity
    >>> q = Quantity(1, "")
    >>> xp.atan(q)
    Quantity['angle'](Array(0.7853982, dtype=float32), unit='rad')

    """
    return Quantity(lax.atan(x.to_units_value(one)), unit=radian)


# ==============================================================================


@register(lax.atanh_p)
def _atanh_p_aq(x: AbstractQuantity) -> AbstractQuantity:
    """Inverse hyperbolic tangent of a quantity.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(2, "")
    >>> xp.atanh(q)
    UncheckedQuantity(Array(nan, dtype=float32), unit='rad')

    """
    return type_np(x)(lax.atanh(x.to_units_value(one)), unit=radian)


@register(lax.atanh_p)
def _atanh_p_q(
    x: AbstractParametricQuantity["dimensionless"],
) -> AbstractParametricQuantity["angle"]:
    """Inverse hyperbolic tangent of a quantity.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> from unxt import Quantity
    >>> q = Quantity(2, "")
    >>> xp.atanh(q)
    Quantity['angle'](Array(nan, dtype=float32), unit='rad')

    """
    return type_np(x)(lax.atanh(x.to_units_value(one)), unit=radian)


# ==============================================================================


@register(lax.axis_index_p)
def _axis_index_p() -> AbstractQuantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.bessel_i0e_p)
def _bessel_i0e_p() -> AbstractQuantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.bessel_i1e_p)
def _bessel_i1e_p() -> AbstractQuantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.bitcast_convert_type_p)
def _bitcast_convert_type_p() -> AbstractQuantity:
    raise NotImplementedError


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
    UncheckedQuantity(Array(2., dtype=float32), unit='m')

    >>> from unxt import Quantity
    >>> q = Quantity(8, "m3")
    >>> jnp.cbrt(q)
    Quantity['length'](Array(2., dtype=float32), unit='m')

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
    Quantity['m1/3'](Array(2., dtype=float32), unit='m(1/3)')

    """
    return Quantity(lax.cbrt(x.value), unit=x.unit ** (1 / 3))


# ==============================================================================


@register(lax.ceil_p)
def _ceil_p(x: AbstractQuantity) -> AbstractQuantity:
    """Ceiling of a quantity.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(1.5, "m")
    >>> xp.ceil(q)
    UncheckedQuantity(Array(2., dtype=float32, ...), unit='m')

    >>> from unxt import Quantity
    >>> q = Quantity(1.5, "m")
    >>> xp.ceil(q)
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
    >>> import quaxed.array_api as xp
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
        x,
        value=lax.clamp(
            min.to_units_value(x.unit),
            x.value,
            max.to_units_value(x.unit),
        ),
    )


# ---------------------------


@register(lax.clamp_p)
def _clamp_p_vaqaq(
    min: ArrayLike, x: AbstractQuantity, max: AbstractQuantity
) -> AbstractQuantity:
    """Clamp a quantity between a value and another quantity.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> import quaxed.lax as lax

    >>> from unxt import UncheckedQuantity
    >>> min = xp.asarray(0)
    >>> max = UncheckedQuantity(2, "")
    >>> q = UncheckedQuantity([-1, 1, 3], "")
    >>> lax.clamp(min, q, max)
    UncheckedQuantity(Array([0, 1, 2], dtype=int32), unit='')

    >>> from unxt import Quantity
    >>> min = xp.asarray(0)
    >>> max = Quantity(2, "")
    >>> q = Quantity([-1, 1, 3], "")
    >>> lax.clamp(min, q, max)
    Quantity['dimensionless'](Array([0, 1, 2], dtype=int32), unit='')

    """
    return replace(
        x, value=lax.clamp(min, x.to_units_value(one), max.to_units_value(one))
    )


# ---------------------------


@register(lax.clamp_p)
def _clamp_p_aqvaq(
    min: AbstractQuantity, x: ArrayLike, max: AbstractQuantity
) -> ArrayLike:
    """Clamp a value between two quantities.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> import quaxed.lax as lax

    >>> from unxt import UncheckedQuantity
    >>> min = UncheckedQuantity(0, "")
    >>> max = UncheckedQuantity(2, "")
    >>> x = xp.asarray([-1, 1, 3])
    >>> lax.clamp(min, x, max)
    Array([0, 1, 2], dtype=int32)

    """
    return lax.clamp(min.to_units_value(one), x, max.to_units_value(one))


@register(lax.clamp_p)
def _clamp_p_qvq(
    min: AbstractParametricQuantity["dimensionless"],
    x: ArrayLike,
    max: AbstractParametricQuantity["dimensionless"],
) -> ArrayLike:
    """Clamp a value between two quantities.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> import quaxed.lax as lax

    >>> from unxt import Quantity
    >>> min = Quantity(0, "")
    >>> max = Quantity(2, "")
    >>> x = xp.asarray([-1, 1, 3])
    >>> lax.clamp(min, x, max)
    Array([0, 1, 2], dtype=int32)

    """
    return lax.clamp(min.to_units_value(one), x, max.to_units_value(one))


# ---------------------------


@register(lax.clamp_p)
def _clamp_p_aqaqv(
    min: AbstractQuantity, x: AbstractQuantity, max: ArrayLike
) -> AbstractQuantity:
    """Clamp a quantity between a quantity and a value.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> import quaxed.lax as lax

    >>> from unxt import UncheckedQuantity
    >>> min = UncheckedQuantity(0, "")
    >>> max = xp.asarray(2)
    >>> q = UncheckedQuantity([-1, 1, 3], "")
    >>> lax.clamp(min, q, max)
    UncheckedQuantity(Array([0, 1, 2], dtype=int32), unit='')

    """
    return replace(
        x, value=lax.clamp(min.to_units_value(one), x.to_units_value(one), max)
    )


@register(lax.clamp_p)
def _clamp_p_qqv(
    min: AbstractParametricQuantity["dimensionless"],
    x: AbstractParametricQuantity["dimensionless"],
    max: ArrayLike,
) -> AbstractParametricQuantity["dimensionless"]:
    """Clamp a quantity between a quantity and a value.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> import quaxed.lax as lax

    >>> from unxt import Quantity
    >>> min = Quantity(0, "")
    >>> max = xp.asarray(2)
    >>> q = Quantity([-1, 1, 3], "")
    >>> lax.clamp(min, q, max)
    Quantity['dimensionless'](Array([0, 1, 2], dtype=int32), unit='')

    """
    return replace(
        x, value=lax.clamp(min.to_units_value(one), x.to_units_value(one), max)
    )


# ==============================================================================


@register(lax.clz_p)
def _clz_p() -> AbstractQuantity:
    raise NotImplementedError


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
    y_ = y.to_units_value(x.unit)
    return replace(x, value=lax.complex(x.value, y_))


# ==============================================================================
# Concatenation


@register(lax.concatenate_p)
def _concatenate_p_aq(*operands: AbstractQuantity, dimension: Any) -> AbstractQuantity:
    """Concatenate quantities.

    Examples
    --------
    >>> import quaxed.array_api as xp

    >>> from unxt import UncheckedQuantity
    >>> q1 = UncheckedQuantity([1.0], "km")
    >>> q2 = UncheckedQuantity([2_000.0], "m")
    >>> xp.concat([q1, q2])
    UncheckedQuantity(Array([1., 2.], dtype=float32), unit='km')

    >>> from unxt import Quantity
    >>> q1 = Quantity([1.0], "km")
    >>> q2 = Quantity([2_000.0], "m")
    >>> xp.concat([q1, q2])
    Quantity['length'](Array([1., 2.], dtype=float32), unit='km')

    """
    operand0 = operands[0]
    units = operand0.unit
    return replace(
        operand0,
        value=lax.concatenate(
            [op.to_units_value(units) for op in operands], dimension=dimension
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
    >>> import quaxed.array_api as xp
    >>> from unxt import Quantity
    >>> theta = Quantity(45, "deg")
    >>> Rz = xp.asarray([[xp.cos(theta), -xp.sin(theta), 0],
    ...                  [xp.sin(theta), xp.cos(theta),  0],
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
                (op.to_units_value(one) if hasattr(op, "unit") else op)
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
    >>> import quaxed.array_api as xp
    >>> from unxt import Quantity
    >>> theta = Quantity(45, "deg")
    >>> Rx = xp.asarray([[1.0, 0.0,           0.0           ],
    ...                  [0.0, xp.cos(theta), -xp.sin(theta)],
    ...                  [0.0, xp.sin(theta), xp.cos(theta) ]])
    >>> Rx
    Quantity[...](Array([[ 1.        ,  0.        ,  0.        ],
                         [ 0.        ,  0.70710677, -0.70710677],
                         [ 0.        ,  0.70710677,  0.70710677]], dtype=float32),
                  unit='')

    """
    return Quantity(
        lax.concatenate(
            [
                (op.to_units_value(one) if hasattr(op, "unit") else op)
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


# ==============================================================================


@register(lax.conj_p)
def _conj_p(x: AbstractQuantity, *, input_dtype: Any) -> AbstractQuantity:
    """Conjugate of a quantity.

    Examples
    --------
    >>> import quaxed.array_api as xp

    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(1 + 2j, "m")
    >>> xp.conj(q)
    UncheckedQuantity(Array(1.-2.j, dtype=complex64, ...), unit='m')

    >>> from unxt import Quantity
    >>> q = Quantity(1 + 2j, "m")
    >>> xp.conj(q)
    Quantity['length'](Array(1.-2.j, dtype=complex64, ...), unit='m')

    """
    del input_dtype  # TODO: use this?
    return replace(x, value=lax.conj(x.value))


# ==============================================================================


@register(lax.conv_general_dilated_p)
def _conv_general_dilated_p() -> AbstractQuantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.convert_element_type_p)
def _convert_element_type_p(
    operand: AbstractQuantity,
    *,
    new_dtype: Any,
    weak_type: Any,
) -> AbstractQuantity:
    """Convert the element type of a quantity."""
    # TODO: examples
    del weak_type
    return replace(operand, value=lax.convert_element_type(operand.value, new_dtype))


# ==============================================================================


@register(lax.copy_p)
def _copy_p(x: AbstractQuantity) -> AbstractQuantity:
    """Copy a quantity.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> import quaxed.numpy as jnp

    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(1, "m")
    >>> jnp.copy(q)
    UncheckedQuantity(Array(1, dtype=int32), unit='m')

    >>> from unxt import Quantity
    >>> q = Quantity(1, "m")
    >>> jnp.copy(q)
    Quantity['length'](Array(1, dtype=int32), unit='m')

    """
    return replace(x, value=lax.copy_p.bind(x.value))


# ==============================================================================


@register(lax.cos_p)
def _cos_p_aq(x: AbstractQuantity) -> AbstractQuantity:
    """Cosine of a quantity.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(1, "rad")
    >>> xp.cos(q)
    UncheckedQuantity(Array(0.5403023, dtype=float32), unit='')

    >>> q = UncheckedQuantity(1, "")
    >>> xp.cos(q)
    UncheckedQuantity(Array(0.5403023, dtype=float32), unit='')

    """
    return type_np(x)(lax.cos(_to_value_rad_or_one(x)), unit=one)


@register(lax.cos_p)
def _cos_p_q(
    x: AbstractParametricQuantity["angle"] | Quantity["dimensionless"],
) -> AbstractParametricQuantity["dimensionless"]:
    """Cosine of a quantity.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> from unxt import Quantity
    >>> q = Quantity(1, "rad")
    >>> xp.cos(q)
    Quantity['dimensionless'](Array(0.5403023, dtype=float32), unit='')

    >>> q = Quantity(1, "")
    >>> xp.cos(q)
    Quantity['dimensionless'](Array(0.5403023, dtype=float32), unit='')

    """
    return Quantity(lax.cos(_to_value_rad_or_one(x)), unit=one)


# ==============================================================================


@register(lax.cosh_p)
def _cosh_p_aq(x: AbstractQuantity) -> AbstractQuantity:
    """Cosine of a quantity.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(1, "rad")
    >>> xp.cosh(q)
    UncheckedQuantity(Array(1.5430806, dtype=float32), unit='')

    >>> q = UncheckedQuantity(1, "")
    >>> xp.cosh(q)
    UncheckedQuantity(Array(1.5430806, dtype=float32), unit='')

    """
    return type_np(x)(lax.cosh(_to_value_rad_or_one(x)), unit=one)


@register(lax.cosh_p)
def _cosh_p_q(
    x: AbstractParametricQuantity["angle"] | Quantity["dimensionless"],
) -> AbstractParametricQuantity["dimensionless"]:
    """Cosine of a quantity.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> from unxt import Quantity
    >>> q = Quantity(1, "rad")
    >>> xp.cosh(q)
    Quantity['dimensionless'](Array(1.5430806, dtype=float32), unit='')

    >>> q = Quantity(1, "")
    >>> xp.cosh(q)
    Quantity['dimensionless'](Array(1.5430806, dtype=float32), unit='')

    """
    return type_np(x)(lax.cosh(_to_value_rad_or_one(x)), unit=one)


# ==============================================================================


@register(lax.create_token_p)
def _create_token_p() -> AbstractQuantity:
    raise NotImplementedError


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
        operand,
        value=lax.cumprod(operand.to_units_value(one), axis=axis, reverse=reverse),
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
    return replace(x, value=jax.device_put(x.value, **kwargs))


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
    return replace(x, value=lax.digamma(x.to_units_value(one)))


# ==============================================================================
# Division


@register(lax.div_p)
def _div_p_qq(x: AbstractQuantity, y: AbstractQuantity) -> AbstractQuantity:
    """Division of two quantities.

    Examples
    --------
    >>> import quaxed.array_api as xp

    >>> from unxt import UncheckedQuantity
    >>> q1 = UncheckedQuantity(1, "m")
    >>> q2 = UncheckedQuantity(2, "s")
    >>> xp.divide(q1, q2)
    UncheckedQuantity(Array(0.5, dtype=float32), unit='m / s')
    >>> q1 / q2
    UncheckedQuantity(Array(0.5, dtype=float32), unit='m / s')

    >>> from unxt import Quantity
    >>> q1 = Quantity(1, "m")
    >>> q2 = Quantity(2, "s")
    >>> xp.divide(q1, q2)
    Quantity['speed'](Array(0.5, dtype=float32), unit='m / s')
    >>> q1 / q2
    Quantity['speed'](Array(0.5, dtype=float32), unit='m / s')

    """
    x, y = promote(x, y)
    unit = Unit(x.unit / y.unit)
    return type_np(x)(lax.div(x.value, y.value), unit=unit)


@register(lax.div_p)
def _div_p_vq(x: ArrayLike, y: AbstractQuantity) -> AbstractQuantity:
    """Division of an array by a quantity.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> x = xp.asarray([1.0, 2, 3])

    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(2.0, "m")
    >>> xp.divide(x, q)
    UncheckedQuantity(Array([0.5, 1. , 1.5], dtype=float32), unit='1 / m')
    >>> x / q
    UncheckedQuantity(Array([0.5, 1. , 1.5], dtype=float32), unit='1 / m')

    >>> from unxt import Quantity
    >>> q = Quantity(2.0, "m")
    >>> xp.divide(x, q)
    Quantity['wavenumber'](Array([0.5, 1. , 1.5], dtype=float32), unit='1 / m')
    >>> x / q
    Quantity['wavenumber'](Array([0.5, 1. , 1.5], dtype=float32), unit='1 / m')

    """
    return type_np(y)(lax.div(x, y.value), unit=1 / y.unit)


@register(lax.div_p)
def _div_p_qv(x: AbstractQuantity, y: ArrayLike) -> AbstractQuantity:
    """Division of a quantity by an array.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> y = xp.asarray([1.0, 2, 3])

    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(6.0, "m")
    >>> xp.divide(q, y)
    UncheckedQuantity(Array([6., 3., 2.], dtype=float32), unit='m')
    >>> q / y
    UncheckedQuantity(Array([6., 3., 2.], dtype=float32), unit='m')

    >>> from unxt import Quantity
    >>> q = Quantity(6.0, "m")
    >>> xp.divide(q, y)
    Quantity['length'](Array([6., 3., 2.], dtype=float32), unit='m')
    >>> q / y
    Quantity['length'](Array([6., 3., 2.], dtype=float32), unit='m')

    """
    return replace(x, value=lax.div(x.value, y))


# ==============================================================================


@register(lax.dot_general_p)
def _dot_general_jq(
    lhs: ArrayLike,
    rhs: AbstractQuantity,
    *,
    dimension_numbers: DotDimensionNumbers,
    precision: PrecisionLike,
    preferred_element_type: DTypeLike | None = None,
) -> AbstractQuantity:
    """Dot product of an array and a quantity.

    >>> import jax.numpy as jnp
    >>> import quaxed.array_api as xp
    >>> from unxt import Quantity, UncheckedQuantity

    >>> theta = jnp.pi / 4  # 45 degrees
    >>> Rz = jnp.asarray([[jnp.cos(theta), -jnp.sin(theta), 0],
    ...                   [jnp.sin(theta), jnp.cos(theta),  0],
    ...                   [0,              0,               1]])

    >>> q = UncheckedQuantity([1, 0, 0], "m")
    >>> xp.linalg.matmul(Rz, q)
    UncheckedQuantity(Array([0.70710677, 0.70710677, 0. ], dtype=float32), unit='m')
    >>> Rz @ q
    UncheckedQuantity(Array([0.70710677, 0.70710677, 0. ], dtype=float32), unit='m')

    >>> q = Quantity([1, 0, 0], "m")
    >>> xp.linalg.matmul(Rz, q)
    Quantity['length'](Array([0.70710677, 0.70710677, 0. ], dtype=float32), unit='m')
    >>> Rz @ q
    Quantity['length'](Array([0.70710677, 0.70710677, 0. ], dtype=float32), unit='m')
    """
    return type_np(rhs)(
        lax.dot_general_p.bind(
            lhs,
            rhs.value,
            dimension_numbers=dimension_numbers,
            precision=precision,
            preferred_element_type=preferred_element_type,
        ),
        unit=rhs.unit,
    )


@register(lax.dot_general_p)
def _dot_general_qq(
    lhs: AbstractQuantity,
    rhs: AbstractQuantity,
    *,
    dimension_numbers: DotDimensionNumbers,
    precision: PrecisionLike,
    preferred_element_type: DTypeLike | None = None,
) -> AbstractQuantity:
    """Dot product of two quantities.

    Examples
    --------
    This is a dot product of two quantities.

    >>> import quaxed.array_api as xp
    >>> from unxt import UncheckedQuantity

    >>> q1 = UncheckedQuantity([1, 2, 3], "m")
    >>> q2 = UncheckedQuantity([4, 5, 6], "m")
    >>> xp.vecdot(q1, q2)
    UncheckedQuantity(Array(32, dtype=int32), unit='m2')
    >>> q1 @ q2
    UncheckedQuantity(Array(32, dtype=int32), unit='m2')

    >>> from unxt import Quantity

    >>> q1 = Quantity([1, 2, 3], "m")
    >>> q2 = Quantity([4, 5, 6], "m")
    >>> xp.vecdot(q1, q2)
    Quantity['area'](Array(32, dtype=int32), unit='m2')
    >>> q1 @ q2
    Quantity['area'](Array(32, dtype=int32), unit='m2')

    >>> from unxt import Distance

    >>> q1 = Distance([1, 2, 3], "m")
    >>> q2 = Quantity([4, 5, 6], "m")
    >>> xp.vecdot(q1, q2)
    Quantity['area'](Array(32, dtype=int32), unit='m2')
    >>> q1 @ q2
    Quantity['area'](Array(32, dtype=int32), unit='m2')

    This rule is also used by `jnp.matmul` for quantities.

    >>> Rz = xp.asarray([[0, -1,  0],
    ...                  [1,  0,  0],
    ...                  [0,  0,  1]])
    >>> q = Quantity([1, 0, 0], "m")
    >>> Rz @ q
    Quantity['length'](Array([0, 1, 0], dtype=int32), unit='m')

    This uses `matmul` for quantities.

    >>> xp.linalg.matmul(Rz, q)
    Quantity['length'](Array([0, 1, 0], dtype=int32), unit='m')

    """
    lhs, rhs = promote(lhs, rhs)
    return type_np(lhs)(
        lax.dot_general_p.bind(
            lhs.value,
            rhs.value,
            dimension_numbers=dimension_numbers,
            precision=precision,
            preferred_element_type=preferred_element_type,
        ),
        unit=lhs.unit * rhs.unit,
    )


@register(lax.dot_general_p)
def _dot_general_dd(
    lhs: AbstractDistance,
    rhs: AbstractDistance,
    *,
    dimension_numbers: DotDimensionNumbers,
    precision: PrecisionLike,
    preferred_element_type: DTypeLike | None = None,
) -> Quantity:
    """Dot product of two Distances.

    Examples
    --------
    This is a dot product of two Distances.

    >>> import quaxed.array_api as xp
    >>> from unxt import Distance

    >>> q1 = Distance([1, 2, 3], "m")
    >>> q2 = Distance([4, 5, 6], "m")
    >>> xp.vecdot(q1, q2)
    Quantity['area'](Array(32, dtype=int32), unit='m2')
    >>> q1 @ q2
    Quantity['area'](Array(32, dtype=int32), unit='m2')

    This rule is also used by `jnp.matmul` for quantities.

    >>> Rz = xp.asarray([[0, -1,  0],
    ...                  [1,  0,  0],
    ...                  [0,  0,  1]])
    >>> q = Quantity([1, 0, 0], "m")
    >>> Rz @ q
    Quantity['length'](Array([0, 1, 0], dtype=int32), unit='m')

    This uses `matmul` for quantities.

    >>> xp.linalg.matmul(Rz, q)
    Quantity['length'](Array([0, 1, 0], dtype=int32), unit='m')

    """
    return Quantity(
        lax.dot_general_p.bind(
            lhs.value,
            rhs.value,
            dimension_numbers=dimension_numbers,
            precision=precision,
            preferred_element_type=preferred_element_type,
        ),
        unit=lhs.unit * rhs.unit,
    )


# ==============================================================================


@register(lax.dynamic_slice_p)
def _dynamic_slice_p(
    operand: AbstractQuantity,
    start_indices: ArrayLike,
    dynamic_sizes: ArrayLike,
    *,
    slice_sizes: Any,
) -> AbstractQuantity:
    raise NotImplementedError  # TODO: implement


# ==============================================================================


@register(lax.dynamic_update_slice_p)
def _dynamic_update_slice_p() -> AbstractQuantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.eq_p)
def _eq_p_qq(x: AbstractQuantity, y: AbstractQuantity) -> ArrayLike:
    """Equality of two quantities.

    Examples
    --------
    >>> import quaxed.array_api as xp

    >>> from unxt import UncheckedQuantity
    >>> q1 = UncheckedQuantity(1, "m")
    >>> q2 = UncheckedQuantity(1, "m")
    >>> xp.equal(q1, q2)
    Array(True, dtype=bool, ...)
    >>> q1 == q2
    Array(True, dtype=bool, ...)

    >>> from unxt import Quantity
    >>> q1 = Quantity(1, "m")
    >>> q2 = Quantity(1, "m")
    >>> xp.equal(q1, q2)
    Array(True, dtype=bool, ...)
    >>> q1 == q2
    Array(True, dtype=bool, ...)

    """
    return lax.eq(x.value, y.to_units_value(x.unit))


@register(lax.eq_p)
def _eq_p_vq(x: ArrayLike, y: AbstractQuantity) -> ArrayLike:
    """Equality of an array and a quantity.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> x = xp.asarray([1.0, 2, 3])

    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(2.0, "")
    >>> xp.equal(x, q)
    Array([False,  True, False], dtype=bool)

    >>> from unxt import Quantity
    >>> q = Quantity(2.0, "")
    >>> xp.equal(x, q)
    Array([False,  True, False], dtype=bool)

    """
    return lax.eq(x, y.to_units_value(one))


@register(lax.eq_p)
def _eq_p_aqv(x: AbstractQuantity, y: ArrayLike) -> ArrayLike:
    """Equality of an array and a quantity.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> y = xp.asarray([1.0, 2, 3])

    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(2.0, "")
    >>> xp.equal(q, y)
    Array([False,  True, False], dtype=bool)

    >>> q = UncheckedQuantity([3., 2, 1], "")
    >>> xp.equal(q, y)
    Array([False,  True, False], dtype=bool)

    >>> q = UncheckedQuantity([3., 2, 1], "m")
    >>> try: xp.equal(q, y)
    ... except Exception as e: print(e)
    'm' (length) and '' (dimensionless) are not convertible

    >>> from unxt import Quantity
    >>> q = Quantity(2.0, "")
    >>> xp.equal(q, y)
    Array([False,  True, False], dtype=bool)

    >>> q = Quantity([3., 2, 1], "")
    >>> xp.equal(q, y)
    Array([False,  True, False], dtype=bool)

    >>> q = Quantity([3., 2, 1], "m")
    >>> try: xp.equal(q, y)
    ... except Exception as e: print(e)
    'm' (length) and '' (dimensionless) are not convertible

    Check against the special cases:

    >>> q == 0
    False

    >>> q == xp.inf
    False

    """
    is_special = jnp.isscalar(y) and (jnp.isinf(y) | (y == 0))

    def special_case(_: Any) -> Array:
        return lax.eq(x.value, y)

    def regular_case(_: Any) -> Array:
        return lax.eq(x.to_units_value(one), y)

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


@register(lax.eq_to_p)
def _eq_to_p() -> AbstractQuantity:
    raise NotImplementedError


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
    return replace(x, value=lax.erf_inv(x.to_units_value(one)))


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
    return replace(x, value=lax.erf(x.to_units_value(one)))


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
    return replace(x, value=lax.erfc(x.to_units_value(one)))


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
    UncheckedQuantity(Array(8., dtype=float32), unit='')

    >>> from unxt import Quantity
    >>> q = Quantity(3, "")
    >>> jnp.exp2(q)
    Quantity['dimensionless'](Array(8., dtype=float32), unit='')

    """
    return replace(x, value=lax.exp2(x.to_units_value(one)))


# ==============================================================================


@register(lax.exp_p)
def _exp_p(x: AbstractQuantity) -> AbstractQuantity:
    """Exponential of a quantity.

    Examples
    --------
    >>> import quaxed.array_api as xp

    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(1, "")
    >>> xp.exp(q)
    UncheckedQuantity(Array(2.7182817, dtype=float32), unit='')

    >>> from unxt import Quantity
    >>> q = Quantity(1, "")
    >>> xp.exp(q)
    Quantity['dimensionless'](Array(2.7182817, dtype=float32), unit='')

    Euler's crown jewel:

    >>> xp.exp(Quantity(xp.pi * 1j, "")) + 1
    Quantity['dimensionless'](Array(0.-8.742278e-08j, dtype=complex64, ...), unit='')

    Pretty close to zero!

    """
    # TODO: more meaningful error message.
    return replace(x, value=lax.exp(x.to_units_value(one)))


# ==============================================================================


@register(lax.expm1_p)
def _expm1_p(x: AbstractQuantity) -> AbstractQuantity:
    """Exponential of a quantity minus 1.

    Examples
    --------
    >>> import quaxed.array_api as xp

    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(0, "")
    >>> xp.expm1(q)
    UncheckedQuantity(Array(0., dtype=float32), unit='')

    >>> from unxt import Quantity
    >>> q = Quantity(0, "")
    >>> xp.expm1(q)
    Quantity['dimensionless'](Array(0., dtype=float32), unit='')

    """
    return replace(x, value=lax.expm1(x.to_units_value(one)))


# ==============================================================================


@register(lax.fft_p)
def _fft_p(x: AbstractQuantity, *, fft_type: Any, fft_lengths: Any) -> AbstractQuantity:
    """Fast Fourier transform of a quantity.

    Examples
    --------
    >>> import quaxed.array_api as xp

    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity([1, 2, 3], "")
    >>> xp.fft.fft(q)
    UncheckedQuantity(Array([ 6. +0.j       , -1.5+0.8660254j, -1.5-0.8660254j],
                       dtype=complex64), unit='')

    >>> from unxt import Quantity
    >>> q = Quantity([1, 2, 3], "")
    >>> xp.fft.fft(q)
    Quantity['dimensionless'](Array([ 6. +0.j       , -1.5+0.8660254j, -1.5-0.8660254j],
                                    dtype=complex64), unit='')

    """
    # TODO: what units can this support?
    return replace(
        x,
        value=lax.fft(x.to_units_value(one), fft_type, fft_lengths),
    )


# ==============================================================================


@register(lax.floor_p)
def _floor_p(x: AbstractQuantity) -> AbstractQuantity:
    """Floor of a quantity.

    Examples
    --------
    >>> import quaxed.array_api as xp

    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(1.5, "")
    >>> xp.floor(q)
    UncheckedQuantity(Array(1., dtype=float32, ...), unit='')

    >>> from unxt import Quantity
    >>> q = Quantity(1.5, "")
    >>> xp.floor(q)
    Quantity['dimensionless'](Array(1., dtype=float32, ...), unit='')

    """
    return replace(x, value=lax.floor(x.value))


# ==============================================================================


# used in `jnp.cross`
@register(lax.gather_p)
def _gather_p(
    operand: AbstractQuantity,
    start_indices: ArrayLike,
    *,
    dimension_numbers: GatherDimensionNumbers,
    slice_sizes: Shape,
    unique_indices: bool,
    indices_are_sorted: bool,
    mode: str | GatherScatterMode | None,
    fill_value: Any,
) -> AbstractQuantity:
    # TODO: examples
    return replace(
        operand,
        value=lax.gather_p.bind(
            operand.value,
            start_indices,
            dimension_numbers=dimension_numbers,
            slice_sizes=slice_sizes,
            unique_indices=unique_indices,
            indices_are_sorted=indices_are_sorted,
            mode=mode,
            fill_value=fill_value,
        ),
    )


# ==============================================================================


@register(lax.ge_p)
def _ge_p_qq(x: AbstractQuantity, y: AbstractQuantity) -> ArrayLike:
    """Greater than or equal to of two quantities.

    Examples
    --------
    >>> import quaxed.array_api as xp

    >>> from unxt import UncheckedQuantity
    >>> q1 = UncheckedQuantity(1_001., "m")
    >>> q2 = UncheckedQuantity(1., "km")
    >>> xp.greater_equal(q1, q2)
    Array(True, dtype=bool, ...)
    >>> q1 >= q2
    Array(True, dtype=bool, ...)

    >>> from unxt import Quantity
    >>> q1 = Quantity(1_001., "m")
    >>> q2 = Quantity(1., "km")
    >>> xp.greater_equal(q1, q2)
    Array(True, dtype=bool, ...)
    >>> q1 >= q2
    Array(True, dtype=bool, ...)

    """
    return lax.ge(x.value, y.to_units_value(x.unit))


@register(lax.ge_p)
def _ge_p_vq(x: ArrayLike, y: AbstractQuantity) -> ArrayLike:
    """Greater than or equal to of an array and a quantity.

    Examples
    --------
    >>> import quaxed.array_api as xp

    >>> x = xp.asarray(1_001.0)

    >>> from unxt import UncheckedQuantity
    >>> q2 = UncheckedQuantity(1., "")
    >>> xp.greater_equal(x, q2)
    Array(True, dtype=bool, ...)

    >>> from unxt import Quantity
    >>> q2 = Quantity(1., "")
    >>> xp.greater_equal(x, q2)
    Array(True, dtype=bool, ...)

    """
    return lax.ge(x, y.to_units_value(one))


@register(lax.ge_p)
def _ge_p_qv(x: AbstractQuantity, y: ArrayLike) -> ArrayLike:
    """Greater than or equal to of a quantity and an array.

    Examples
    --------
    >>> import quaxed.array_api as xp

    >>> y = xp.asarray(0.9)

    >>> from unxt import UncheckedQuantity
    >>> q1 = UncheckedQuantity(1., "")
    >>> xp.greater_equal(q1, y)
    Array(True, dtype=bool, ...)

    >>> from unxt import Quantity
    >>> q1 = Quantity(1., "")
    >>> xp.greater_equal(q1, y)
    Array(True, dtype=bool, ...)

    """
    # if jnp.array_equal(y, 0):
    #     return lax.ge(x.value, y)
    return lax.ge(x.to_units_value(one), y)


# ==============================================================================


@register(lax.gt_p)
def _gt_p_qq(x: AbstractQuantity, y: AbstractQuantity) -> ArrayLike:
    """Greater than of two quantities.

    Examples
    --------
    >>> import quaxed.array_api as xp

    >>> from unxt import UncheckedQuantity
    >>> q1 = UncheckedQuantity(1_001., "m")
    >>> q2 = UncheckedQuantity(1., "km")
    >>> xp.greater_equal(q1, q2)
    Array(True, dtype=bool, ...)

    >>> from unxt import Quantity
    >>> q1 = Quantity(1_001., "m")
    >>> q2 = Quantity(1., "km")
    >>> xp.greater_equal(q1, q2)
    Array(True, dtype=bool, ...)

    """
    return lax.gt(x.value, y.to_units_value(x.unit))


@register(lax.gt_p)
def _gt_p_vq(x: ArrayLike, y: AbstractQuantity) -> ArrayLike:
    """Greater than of an array and a quantity.

    Examples
    --------
    >>> import quaxed.array_api as xp

    >>> x = xp.asarray(1_001.0)

    >>> from unxt import UncheckedQuantity
    >>> q2 = UncheckedQuantity(1., "")
    >>> xp.greater_equal(x, q2)
    Array(True, dtype=bool, ...)

    >>> from unxt import Quantity
    >>> q2 = Quantity(1., "")
    >>> xp.greater_equal(x, q2)
    Array(True, dtype=bool, ...)

    """
    return lax.gt(x, y.to_units_value(one))


@register(lax.gt_p)
def _gt_p_qv(x: AbstractQuantity, y: ArrayLike) -> ArrayLike:
    """Greater than or equal to of a quantity and an array.

    Examples
    --------
    >>> import quaxed.array_api as xp

    >>> y = xp.asarray(0.9)

    >>> from unxt import UncheckedQuantity
    >>> q1 = UncheckedQuantity(1., "")
    >>> xp.greater_equal(q1, y)
    Array(True, dtype=bool, ...)

    >>> from unxt import Quantity
    >>> q1 = Quantity(1., "")
    >>> xp.greater_equal(q1, y)
    Array(True, dtype=bool, ...)

    """
    return lax.gt(x.to_units_value(one), y)


@register(lax.gt_p)
def _gt_p_qi(x: AbstractQuantity, y: int) -> ArrayLike:
    """Greater than or equal to of a quantity and an array.

    Examples
    --------
    >>> import quaxed.numpy as jnp

    >>> y = 0

    >>> from unxt import UncheckedQuantity
    >>> q1 = UncheckedQuantity(1., "")
    >>> jnp.greater(q1, y)
    Array(True, dtype=bool, ...)

    >>> from unxt import Quantity
    >>> q1 = Quantity(1., "")
    >>> jnp.greater(q1, y)
    Array(True, dtype=bool, ...)

    """
    return lax.gt(x.to_units_value(one), y)


# ==============================================================================


@register(lax.igamma_grad_a_p)
def _igamma_grad_a_p() -> AbstractQuantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.igamma_p)
def _igamma_p(a: AbstractQuantity, x: AbstractQuantity) -> AbstractQuantity:
    """Regularized incomplete gamma function of a and x."""
    return lax.igamma(a.to_units_value(one), x.to_units_value(one))


# ==============================================================================


@register(lax.igammac_p)
def _igammac_p() -> AbstractQuantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.imag_p)
def _imag_p(x: AbstractQuantity) -> AbstractQuantity:
    return replace(x, value=lax.imag(x.value))


# ==============================================================================


@register(lax.infeed_p)
def _infeed_p() -> AbstractQuantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.integer_pow_p)
def _integer_pow_p(x: AbstractQuantity, *, y: Any) -> AbstractQuantity:
    """Integer power of a quantity.

    Examples
    --------
    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(2, "m")
    >>> q ** 3
    UncheckedQuantity(Array(8, dtype=int32), unit='m3')

    >>> from unxt import Quantity
    >>> q = Quantity(2, "m")
    >>> q ** 3
    Quantity['volume'](Array(8, dtype=int32), unit='m3')

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
    Quantity['volume'](Array(8, dtype=int32), unit='m3')

    """
    return Quantity(value=lax.integer_pow(x.value, y), unit=x.unit**y)


# ==============================================================================


# @register(lax.iota_p)
# def _iota_p(dtype: AbstractParametricQuantity) -> AbstractParametricQuantity:
#     raise NotImplementedError


# ==============================================================================


@register(lax.is_finite_p)
def _is_finite_p(x: AbstractQuantity) -> ArrayLike:
    """Check if a quantity is finite.

    Examples
    --------
    >>> import quaxed.array_api as xp

    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(1, "m")
    >>> xp.isfinite(q)
    array(True)
    >>> q = UncheckedQuantity(float('inf'), "m")
    >>> xp.isfinite(q)
    Array(False, dtype=bool, ...)

    >>> from unxt import Quantity
    >>> q = Quantity(1, "m")
    >>> xp.isfinite(q)
    array(True)
    >>> q = Quantity(float('inf'), "m")
    >>> xp.isfinite(q)
    Array(False, dtype=bool, ...)

    """
    return lax.is_finite(x.value)


# ==============================================================================


@register(lax.le_p)
def _le_p_qq(x: AbstractQuantity, y: AbstractQuantity) -> ArrayLike:
    """Less than or equal to of two quantities.

    Examples
    --------
    >>> import quaxed.array_api as xp

    >>> from unxt import UncheckedQuantity
    >>> q1 = UncheckedQuantity(1_001., "m")
    >>> q2 = UncheckedQuantity(1., "km")
    >>> xp.less_equal(q1, q2)
    Array(False, dtype=bool, ...)

    >>> from unxt import Quantity
    >>> q1 = Quantity(1_001., "m")
    >>> q2 = Quantity(1., "km")
    >>> xp.less_equal(q1, q2)
    Array(False, dtype=bool, ...)

    """
    return lax.le(x.value, y.to_units_value(x.unit))


@register(lax.le_p)
def _le_p_vq(x: ArrayLike, y: AbstractQuantity) -> ArrayLike:
    """Less than or equal to of an array and a quantity.

    Examples
    --------
    >>> import quaxed.array_api as xp

    >>> x1 = xp.asarray(1.001)

    >>> from unxt import UncheckedQuantity
    >>> q2 = UncheckedQuantity(1., "")
    >>> xp.less_equal(x1, q2)
    Array(False, dtype=bool, ...)

    >>> from unxt import Quantity
    >>> q2 = Quantity(1., "")
    >>> xp.less_equal(x1, q2)
    Array(False, dtype=bool, ...)

    """
    return lax.le(x, y.to_units_value(one))


@register(lax.le_p)
def _le_p_qv(x: AbstractQuantity, y: ArrayLike) -> ArrayLike:
    """Less than or equal to of a quantity and an array.

    Examples
    --------
    >>> import quaxed.array_api as xp

    >>> y1 = xp.asarray(0.9)

    >>> from unxt import UncheckedQuantity
    >>> q1 = UncheckedQuantity(1., "")
    >>> xp.less_equal(q1, y1)
    Array(False, dtype=bool, ...)

    >>> from unxt import Quantity
    >>> q1 = Quantity(1., "")
    >>> xp.less_equal(q1, y1)
    Array(False, dtype=bool, ...)

    """
    return lax.le(x.to_units_value(one), y)


# ==============================================================================


@register(lax.le_to_p)
def _le_to_p() -> AbstractParametricQuantity:
    raise NotImplementedError


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
    UncheckedQuantity(Array(0.6931474, dtype=float32), unit='')

    >>> from unxt import Quantity
    >>> q = Quantity(3, "")
    >>> jsp.special.gammaln(q)
    Quantity['dimensionless'](Array(0.6931474, dtype=float32), unit='')

    """
    # TODO: are there any units that this can support?
    return replace(x, value=lax.lgamma(x.to_units_value(one)))


# ==============================================================================


@register(lax.linear_solve_p)
def _linear_solve_p() -> AbstractQuantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.log1p_p)
def _log1p_p(x: AbstractQuantity) -> AbstractQuantity:
    return replace(x, value=lax.log1p(x.to_units_value(one)))


# ==============================================================================


@register(lax.log_p)
def _log_p(x: AbstractQuantity) -> AbstractQuantity:
    return replace(x, value=lax.log(x.to_units_value(one)))


# ==============================================================================


@register(lax.logistic_p)
def _logistic_p(x: AbstractQuantity) -> AbstractQuantity:
    return replace(x, value=lax.logistic(x.to_units_value(one)))


# ==============================================================================


@register(lax.lt_p)
def _lt_p_qq(x: AbstractQuantity, y: AbstractQuantity) -> ArrayLike:
    return lax.lt(x.value, y.to_units_value(x.unit))


@register(lax.lt_p)
def _lt_p_vq(x: ArrayLike, y: AbstractQuantity) -> ArrayLike:
    return lax.lt(x, y.to_units_value(one))


@register(lax.lt_p)
def _lt_p_qv(x: AbstractQuantity, y: ArrayLike) -> ArrayLike:
    return lax.lt(x.to_units_value(one), y)


# ==============================================================================


@register(lax.lt_to_p)
def _lt_to_p() -> ArrayLike:
    raise NotImplementedError


# ==============================================================================


@register(lax.max_p)
def _max_p_qq(x: AbstractQuantity, y: AbstractQuantity) -> AbstractQuantity:
    return replace(x, value=lax.max(x.value, y.to_units_value(x.unit)))


@register(lax.max_p)
def _max_p_vq(x: ArrayLike, y: AbstractQuantity) -> AbstractQuantity:
    return replace(y, value=lax.max(x, y.to_units_value(one)))


@register(lax.max_p)
def _max_p_qv(x: AbstractQuantity, y: ArrayLike) -> AbstractQuantity:
    return replace(x, value=lax.max(x.to_units_value(one), y))


# ==============================================================================


@register(lax.min_p)
def _min_p_qq(x: AbstractQuantity, y: AbstractQuantity) -> AbstractQuantity:
    return replace(x, value=lax.min(x.value, y.to_units_value(x.unit)))


@register(lax.min_p)
def _min_p_vq(x: ArrayLike, y: AbstractQuantity) -> AbstractQuantity:
    return replace(y, value=lax.min(x, y.to_units_value(one)))


@register(lax.min_p)
def _min_p_qv(x: AbstractQuantity, y: ArrayLike) -> AbstractQuantity:
    return replace(x, value=lax.min(x.to_units_value(one), y))


# ==============================================================================
# Multiplication


@register(lax.mul_p)
def _mul_p_qq(x: AbstractQuantity, y: AbstractQuantity) -> AbstractQuantity:
    unit = Unit(x.unit * y.unit)
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
    return lax.ne(x.value, y.to_units_value(x.unit))


@register(lax.ne_p)
def _ne_p_vq(x: ArrayLike, y: AbstractQuantity) -> ArrayLike:
    return lax.ne(x, y.to_units_value(one))


@register(lax.ne_p)
def _ne_p_qv(x: AbstractQuantity, y: ArrayLike) -> ArrayLike:
    # special-case for scalar value=0, unit=one
    if y.shape == () and y == 0:  # TODO: proper jax
        return lax.ne(x.value, y)
    return lax.ne(x.to_units_value(one), y)


# @register(lax.ne_p)
# def _ne_p_qv(x: AbstractParametricQuantity, y: ArrayLike) -> ArrayLike:
#     return lax.


# ==============================================================================


@register(lax.neg_p)
def _neg_p(x: AbstractQuantity) -> AbstractQuantity:
    return replace(x, value=lax.neg(x.value))


# ==============================================================================


@register(lax.nextafter_p)
def _nextafter_p() -> AbstractQuantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.not_p)
def _not_p() -> AbstractQuantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.or_p)
def _or_p() -> AbstractQuantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.outfeed_p)
def _outfeed_p() -> AbstractQuantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.pad_p)
def _pad_p() -> AbstractQuantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.pmax_p)
def _pmax_p() -> AbstractQuantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.pmin_p)
def _pmin_p() -> AbstractQuantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.polygamma_p)
def _polygamma_p() -> AbstractQuantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.population_count_p)
def _population_count_p() -> AbstractQuantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.pow_p)
def _pow_p_qq(
    x: AbstractQuantity, y: AbstractParametricQuantity["dimensionless"]
) -> AbstractQuantity:
    y_: Array = y.to_units_value(one)
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


@register(lax.ppermute_p)
def _ppermute_p() -> AbstractQuantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.psum_p)
def _psum_p() -> AbstractQuantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.random_gamma_grad_p)
def _random_gamma_grad_p() -> AbstractQuantity:
    raise NotImplementedError


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


@register(lax.reduce_p)
def _reduce_p() -> AbstractQuantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.reduce_precision_p)
def _reduce_precision_p() -> AbstractQuantity:
    raise NotImplementedError


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


@register(lax.reduce_window_max_p)
def _reduce_window_max_p() -> AbstractQuantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.reduce_window_min_p)
def _reduce_window_min_p() -> AbstractQuantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.reduce_window_p)
def _reduce_window_p() -> AbstractQuantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.reduce_window_sum_p)
def _reduce_window_sum_p() -> AbstractQuantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.reduce_xor_p)
def _reduce_xor_p() -> AbstractQuantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.regularized_incomplete_beta_p)
def _regularized_incomplete_beta_p() -> AbstractQuantity:
    raise NotImplementedError


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
    return replace(x, value=lax.rem(x.value, y.to_units_value(x.unit)))


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


@register(lax.rng_bit_generator_p)
def _rng_bit_generator_p() -> AbstractQuantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.rng_uniform_p)
def _rng_uniform_p() -> AbstractQuantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.round_p)
def _round_p(x: AbstractQuantity, *, rounding_method: Any) -> AbstractQuantity:
    return replace(x, value=lax.round(x.value, rounding_method))


# ==============================================================================


@register(lax.rsqrt_p)
def _rsqrt_p(x: AbstractQuantity) -> AbstractQuantity:
    return type_np(x)(lax.rsqrt(x.value), unit=x.unit ** (-1 / 2))


# ==============================================================================


@register(lax.scan_p)
def _scan_p() -> AbstractQuantity:
    raise NotImplementedError


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
            operand.value,
            scatter_indices,
            updates.to_units_value(operand.units),
            **kwargs,
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


@register(lax.scatter_max_p)
def _scatter_max_p() -> AbstractQuantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.scatter_min_p)
def _scatter_min_p() -> AbstractQuantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.scatter_mul_p)
def _scatter_mul_p() -> AbstractQuantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.scatter_p)
def _scatter_p() -> AbstractQuantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.select_and_gather_add_p)
def _select_and_gather_add_p() -> AbstractQuantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.select_and_scatter_add_p)
def _select_and_scatter_add_p() -> AbstractQuantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.select_and_scatter_p)
def _select_and_scatter_p() -> AbstractQuantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.select_n_p)
def _select_n_p(which: AbstractQuantity, *cases: AbstractQuantity) -> AbstractQuantity:
    unit = cases[0].unit
    cases_ = (case.to_units_value(unit) for case in cases)
    return type_np(which)(lax.select_n(which.to_units_value(one), *cases_), unit=unit)


@register(lax.select_n_p)
def _select_n_p_vq(
    which: AbstractQuantity, case0: AbstractQuantity, case1: ArrayLike
) -> AbstractQuantity:
    # encountered from jnp.hypot
    unit = case0.unit
    return type_np(which)(
        lax.select_n(which.to_units_value(one), case0.to_units_value(unit), case1),
        unit=unit,
    )


@register(lax.select_n_p)
def _select_n_p_jjq(
    which: ArrayLike, case0: ArrayLike, case1: AbstractQuantity
) -> AbstractQuantity:
    # Used by a `xp.linalg.trace`
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
    return replace(
        case0, value=lax.select_n(which, case0.value, case1.to_units_value(unit))
    )


# ==============================================================================


@register(lax.sharding_constraint_p)
def _sharding_constraint_p() -> AbstractQuantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.shift_left_p)
def _shift_left_p() -> AbstractQuantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.shift_right_arithmetic_p)
def _shift_right_arithmetic_p() -> AbstractQuantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.shift_right_logical_p)
def _shift_right_logical_p() -> AbstractQuantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.sign_p)
def _sign_p(x: AbstractQuantity) -> ArrayLike:
    """Sign of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as qnp

    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(10, "m")
    >>> qnp.sign(q)
    Array(1, dtype=int32, ...)

    >>> from unxt import Quantity
    >>> q = Quantity(10, "m")
    >>> qnp.sign(q)
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
    >>> import quaxed.numpy as qnp

    >>> from unxt import UncheckedQuantity
    >>> q = UncheckedQuantity(9, "m")
    >>> qnp.sqrt(q)
    UncheckedQuantity(Array(3., dtype=float32), unit='m(1/2)')

    >>> from unxt import Quantity
    >>> q = Quantity(9, "m")
    >>> qnp.sqrt(q)
    Quantity['m0.5'](Array(3., dtype=float32), unit='m(1/2)')

    """
    # Apply sqrt to the value and adjust the unit
    return type_np(x)(lax.sqrt(x.value), unit=x.unit ** (1 / 2))


@register(lax.sqrt_p)
def _sqrt_p_d(x: AbstractDistance) -> Quantity:
    """Square root of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as qnp

    >>> from unxt import Distance
    >>> q = Distance(9, "m")
    >>> qnp.sqrt(q)
    Quantity['m0.5'](Array(3., dtype=float32), unit='m(1/2)')

    >>> from unxt import Parallax
    >>> q = Parallax(9, "mas")
    >>> qnp.sqrt(q)
    Quantity['rad0.5'](Array(3., dtype=float32), unit='mas(1/2)')

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
    return replace(x, value=lax.sub(x.to_units_value(x.unit), y.to_units_value(x.unit)))


@register(lax.sub_p)
def _sub_p_vq(x: ArrayLike, y: AbstractQuantity) -> AbstractQuantity:
    return replace(y, value=lax.sub(x, y.value))


@register(lax.sub_p)
def _sub_p_qv(x: AbstractQuantity, y: ArrayLike) -> AbstractQuantity:
    return replace(x, value=lax.sub(x.value, y))


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


@register(lax.top_k_p)
def _top_k_p() -> AbstractQuantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.transpose_p)
def _transpose_p(operand: AbstractQuantity, *, permutation: Any) -> AbstractQuantity:
    return replace(operand, value=lax.transpose(operand.value, permutation))


# ==============================================================================


@register(lax.while_p)
def _while_p() -> AbstractQuantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.xor_p)
def _xor_p() -> AbstractQuantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.zeta_p)
def _zeta_p() -> AbstractQuantity:
    raise NotImplementedError
