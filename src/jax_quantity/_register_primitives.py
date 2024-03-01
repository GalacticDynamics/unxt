# pylint: disable=import-error, too-many-lines

__all__: list[str] = []

from collections.abc import Callable, Sequence
from dataclasses import replace
from math import prod
from typing import Any, TypeAlias, TypeVar

import jax
import jax.core
import jax.numpy as jnp
from astropy.units import (  # pylint: disable=no-name-in-module
    Unit,
    UnitBase,
    UnitTypeError,
    dimensionless_unscaled as dimensionless,
    radian,
)
from jax import lax
from jax._src.lax.lax import DotDimensionNumbers, DTypeLike, PrecisionLike
from jax._src.lax.slicing import GatherDimensionNumbers, GatherScatterMode
from jax._src.typing import Shape
from jaxtyping import ArrayLike
from quax import register as register_

from ._core import Quantity, can_convert_unit

T = TypeVar("T")

Axes: TypeAlias = tuple[int, ...]
UnitClasses: TypeAlias = UnitBase


def register(primitive: jax.core.Primitive) -> Callable[[T], T]:
    """:func`quax.register`, but makes mypy happy."""
    return register_(primitive)


def _to_value_rad_or_one(q: Quantity) -> ArrayLike:
    return (
        q.to_value(radian)
        if can_convert_unit(q.unit, radian)
        else q.to_value(dimensionless)
    )


################################################################################
# Registering Primitives

# ==============================================================================


@register(lax.abs_p)
def _abs_p(x: Quantity) -> Quantity:
    return replace(x, value=lax.abs(x.value))


# ==============================================================================


@register(lax.acos_p)
def _acos_p(x: Quantity) -> Quantity:
    v = x.to_value(dimensionless)
    return Quantity(value=lax.acos(v), unit=radian)


# ==============================================================================


@register(lax.acosh_p)
def _acosh_p(x: Quantity) -> Quantity:
    v = x.to_value(dimensionless)
    return Quantity(value=lax.acosh(v), unit=radian)


# ==============================================================================
# Addition


@register(lax.add_p)
def _add_p_qq(x: Quantity, y: Quantity) -> Quantity:
    return Quantity(lax.add(x.to_value(x.unit), y.to_value(x.unit)), unit=x.unit)


@register(lax.add_p)
def _add_p_vq1(x: ArrayLike, y: Quantity["dimensionless"]) -> Quantity:  # type: ignore[type-arg]
    return Quantity(lax.add(x, y.to_value(dimensionless)), unit=dimensionless)


@register(lax.add_p)
def _add_p_vq2(x: ArrayLike, y: Quantity) -> Quantity:
    msg = "Cannot add a non-quantity and quantity."
    raise ValueError(msg)


@register(lax.add_p)
def _add_p_qv1(x: Quantity["dimensionless"], y: ArrayLike) -> Quantity:  # type: ignore[type-arg]
    return Quantity(lax.add(x.to_value(dimensionless), y), unit=dimensionless)


@register(lax.add_p)
def _add_p_qv2(x: Quantity, y: ArrayLike) -> Quantity:
    msg = "Cannot add a quantity and a non-quantity."
    raise ValueError(msg)


# ==============================================================================


@register(lax.after_all_p)
def _after_all_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.all_gather_p)
def _all_gather_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.all_to_all_p)
def _all_to_all_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.and_p)
def _and_p(
    x1: Quantity["dimensionless"],  # type: ignore[type-arg]
    x2: Quantity["dimensionless"],  # type: ignore[type-arg]
    /,
) -> ArrayLike:
    return x1.value & x2.value


# ==============================================================================


@register(lax.approx_top_k_p)
def _approx_top_k_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.argmax_p)
def _argmax_p(operand: Quantity, *, axes: Any, index_dtype: Any) -> Quantity:
    return Quantity(lax.argmax(operand.value, axes[0], index_dtype), unit=operand.unit)


# ==============================================================================


@register(lax.argmin_p)
def _argmin_p(operand: Quantity, *, axes: Any, index_dtype: Any) -> Quantity:
    return Quantity(lax.argmin(operand.value, axes[0], index_dtype), unit=operand.unit)


# ==============================================================================


@register(lax.asin_p)
def _asin_p(x: Quantity) -> Quantity:
    return Quantity(lax.asin(x.to_value(dimensionless)), unit=radian)


# ==============================================================================


@register(lax.asinh_p)
def _asinh_p(x: Quantity) -> Quantity:
    return Quantity(lax.asinh(x.to_value(dimensionless)), unit=radian)


# ==============================================================================


@register(lax.atan2_p)
def _atan2_p(x: Quantity, y: Quantity) -> Quantity:
    y_ = y.to_value(x.unit)
    return Quantity(lax.atan2(x.value, y_), unit=radian)


@register(lax.atan2_p)
def _atan2_p_vq(x: ArrayLike, y: Quantity) -> Quantity:
    y_ = y.to_value(dimensionless)
    return Quantity(lax.atan2(x, y_), unit=radian)


@register(lax.atan2_p)
def _atan2_p_qv(x: Quantity, y: ArrayLike) -> Quantity:
    x_ = x.to_value(dimensionless)
    return Quantity(lax.atan2(x_, y), unit=radian)


# ==============================================================================


@register(lax.atan_p)
def _atan_p(x: Quantity) -> Quantity:
    return Quantity(lax.atan(x.to_value(dimensionless)), unit=radian)


# ==============================================================================


@register(lax.atanh_p)
def _atanh_p(x: Quantity) -> Quantity:
    return Quantity(lax.atanh(x.to_value(dimensionless)), unit=radian)


# ==============================================================================


@register(lax.axis_index_p)
def _axis_index_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.bessel_i0e_p)
def _bessel_i0e_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.bessel_i1e_p)
def _bessel_i1e_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.bitcast_convert_type_p)
def _bitcast_convert_type_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.broadcast_in_dim_p)
def _broadcast_in_dim_p(
    operand: Quantity,
    *,
    shape: Any,
    broadcast_dimensions: Any,
) -> Quantity:
    return Quantity(
        value=lax.broadcast_in_dim(operand.value, shape, broadcast_dimensions),
        unit=operand.unit,
    )


# ==============================================================================


@register(lax.cbrt_p)
def _cbrt_p(x: Quantity) -> Quantity:
    return Quantity(lax.cbrt(x.value), unit=x.unit ** (1 / 3))


# ==============================================================================


@register(lax.ceil_p)
def _ceil_p(x: Quantity) -> Quantity:
    return replace(x, value=lax.ceil(x.value))


# ==============================================================================


@register(lax.clamp_p)
def _clamp_p(min: Quantity, x: Quantity, max: Quantity) -> Quantity:
    return replace(
        x,
        value=lax.clamp(
            min.to_value(x.unit),
            x.value,
            max.to_value(x.unit),
        ),
    )


@register(lax.clamp_p)
def _clamp_p_vqq(min: ArrayLike, x: Quantity, max: Quantity) -> Quantity:
    v = x.to_value(dimensionless)
    maxv = max.to_value(dimensionless)
    return replace(x, value=lax.clamp(min, v, maxv))


@register(lax.clamp_p)
def _clamp_p_qvq(min: Quantity, x: ArrayLike, max: Quantity) -> ArrayLike:
    minv = min.to_value(dimensionless)
    maxv = max.to_value(dimensionless)
    return lax.clamp(minv, x, maxv)


@register(lax.clamp_p)
def _clamp_p_qqv(min: Quantity, x: Quantity, max: ArrayLike) -> Quantity:
    minv = min.to_value(dimensionless)
    v = x.to_value(dimensionless)
    return replace(x, value=lax.clamp(minv, v, max))


# ==============================================================================


@register(lax.clz_p)
def _clz_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.complex_p)
def _complex_p(x: Quantity, y: Quantity) -> Quantity:
    y_ = y.to_value(x.unit)
    return Quantity(lax.complex(x.value, y_), unit=x.unit)


# ==============================================================================


@register(lax.concatenate_p)
def _concatenate_p(*operands: Quantity, dimension: Any) -> Quantity:
    units = operands[0].unit
    return Quantity(
        lax.concatenate(
            [op.to_value(units) for op in operands],
            dimension=dimension,
        ),
        unit=units,
    )


@register(lax.concatenate_p)
def _concatenate_p_jqnd(
    operand0: Quantity["dimensionless"],  # type: ignore[type-arg]
    *operands: Quantity["dimensionless"] | ArrayLike,  # type: ignore[type-arg]
    dimension: Any,
) -> Quantity["dimensionless"]:  # type: ignore[type-arg]
    """Concatenate quantities and arrays with dimensionless units.

    Examples
    --------
    >>> import array_api_jax_compat as xp
    >>> from jax_quantity import Quantity
    >>> theta = Quantity(45, "deg")
    >>> Rz = xp.asarray([[xp.cos(theta), -xp.sin(theta), 0],
    ...                  [xp.sin(theta), xp.cos(theta),  0],
    ...                  [0,             0,              1]])
    >>> Rz
    Quantity['dimensionless'](Array([[ 0.70710678, -0.70710678,  0.        ],
           [ 0.70710678,  0.70710678,  0.        ],
           [ 0.        ,  0.        ,  1.        ]], dtype=float64), unit='')

    """
    return Quantity(
        lax.concatenate(
            [
                (op.to_value(dimensionless) if hasattr(op, "unit") else op)
                for op in (operand0, *operands)
            ],
            dimension=dimension,
        ),
        unit=dimensionless,
    )


# ==============================================================================


# @register(lax.cond_p)  # TODO: implement
# def _implemen(index, consts) -> Quantity:
#     raise NotImplementedError


# ==============================================================================


@register(lax.conj_p)
def _conj_p(x: Quantity, *, input_dtype: Any) -> Quantity:
    del input_dtype  # TODO: use this?
    return replace(x, value=lax.conj(x.value))


# ==============================================================================


@register(lax.conv_general_dilated_p)
def _conv_general_dilated_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.convert_element_type_p)
def _convert_element_type_p(
    operand: Quantity,
    *,
    new_dtype: Any,
    weak_type: Any,
) -> Quantity:
    del weak_type
    return replace(operand, value=lax.convert_element_type(operand.value, new_dtype))


# ==============================================================================


@register(lax.copy_p)
def _copy_p(x: Quantity) -> Quantity:
    return replace(x, value=lax.copy_p.bind(x.value))


# ==============================================================================


@register(lax.cos_p)
def _cos_p(x: Quantity) -> Quantity:
    return Quantity(lax.cos(_to_value_rad_or_one(x)), unit=dimensionless)


# ==============================================================================


@register(lax.cosh_p)
def _cosh_p(x: Quantity) -> Quantity:
    return Quantity(lax.cosh(_to_value_rad_or_one(x)), unit=dimensionless)


# ==============================================================================


@register(lax.create_token_p)
def _create_token_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.cumlogsumexp_p)
def _cumlogsumexp_p(operand: Quantity, *, axis: Any, reverse: Any) -> Quantity:
    # TODO: double check units make sense here.
    return replace(
        operand,
        value=lax.cumlogsumexp(operand.value, axis=axis, reverse=reverse),
    )


# ==============================================================================


@register(lax.cummax_p)
def _cummax_p(operand: Quantity, *, axis: Any, reverse: Any) -> Quantity:
    return replace(operand, value=lax.cummax(operand.value, axis=axis, reverse=reverse))


# ==============================================================================


@register(lax.cummin_p)
def _cummin_p(operand: Quantity, *, axis: Any, reverse: Any) -> Quantity:
    return replace(operand, value=lax.cummin(operand.value, axis=axis, reverse=reverse))


# ==============================================================================


@register(lax.cumprod_p)
def _cumprod_p(operand: Quantity, *, axis: Any, reverse: Any) -> Quantity:
    return replace(
        operand,
        value=lax.cumprod(operand.value, axis=axis, reverse=reverse),
    )


# ==============================================================================


@register(lax.cumsum_p)
def _cumsum_p(operand: Quantity, *, axis: Any, reverse: Any) -> Quantity:
    return replace(operand, value=lax.cumsum(operand.value, axis=axis, reverse=reverse))


# ==============================================================================


@register(lax.device_put_p)
def _device_put_p(x: Quantity, *, device: Any, src: Any) -> Quantity:
    return replace(x, value=jax.device_put(x.value, device=device, src=src))


# ==============================================================================


@register(lax.digamma_p)
def _digamma_p(x: Quantity) -> Quantity:
    if x.unit != dimensionless:
        msg = "TODO: implement the result units for `digamma`."
        raise NotImplementedError(msg)

    return Quantity(lax.digamma(x.value), unit=dimensionless)


# ==============================================================================
# Division


@register(lax.div_p)
def _div_p_qq(x: Quantity, y: Quantity) -> Quantity:
    unit = Unit(x.unit / y.unit)
    return Quantity(lax.div(x.value, y.value), unit=unit)


@register(lax.div_p)
def _div_p_vq(x: ArrayLike, y: Quantity) -> Quantity:
    return Quantity(lax.div(x, y.value), unit=1 / y.unit)


@register(lax.div_p)
def _div_p_qv(x: Quantity, y: ArrayLike) -> Quantity:
    return Quantity(lax.div(x.value, y), unit=x.unit)


# ==============================================================================


@register(lax.dot_general_p)
def _dot_general_jq(
    lhs: ArrayLike,
    rhs: Quantity,
    *,
    dimension_numbers: DotDimensionNumbers,
    precision: PrecisionLike,
    preferred_element_type: DTypeLike | None = None,
) -> Quantity:
    """Dot product of an array and a quantity.

    >>> import jax.numpy as jnp
    >>> from jax_quantity import Quantity

    >>> theta = jnp.pi / 4  # 45 degrees
    >>> Rz = jnp.asarray([[jnp.cos(theta), -jnp.sin(theta), 0],
    ...                   [jnp.sin(theta), jnp.cos(theta),  0],
    ...                   [0,              0,               1]])
    >>> q = Quantity([1, 0, 0], "m")
    >>> Rz @ q
    Quantity['length'](Array([0.70710678, 0.70710678, 0. ], dtype=float64), unit='m')
    """
    return Quantity(
        lax.dot_general_p.bind(
            lhs,
            rhs.value,
            dimension_numbers=dimension_numbers,
            precision=precision,
            preferred_element_type=preferred_element_type,
        ),
        unit=rhs.unit,
    )


# ==============================================================================


@register(lax.dynamic_slice_p)
def _dynamic_slice_p(
    operand: Quantity,
    start_indices: ArrayLike,
    dynamic_sizes: ArrayLike,
    *,
    slice_sizes: Any,
) -> Quantity:
    raise NotImplementedError  # TODO: implement


# ==============================================================================


@register(lax.dynamic_update_slice_p)
def _dynamic_update_slice_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.eq_p)
def _eq_p_qq(x: Quantity, y: Quantity) -> ArrayLike:
    return lax.eq(x.value, y.to_value(x.unit))


@register(lax.eq_p)
def _eq_p_vq(x: ArrayLike, y: Quantity) -> ArrayLike:
    return lax.eq(x, y.to_value(dimensionless))


@register(lax.eq_p)
def _eq_p_qv(x: Quantity, y: ArrayLike) -> ArrayLike:
    # special-case for all-0 values
    return lax.eq(x.value, y)


# ==============================================================================


@register(lax.eq_to_p)
def _eq_to_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.erf_inv_p)
def _erf_inv_p(x: Quantity) -> Quantity:
    # TODO: can this support non-dimensionless quantities?
    return Quantity(lax.erf_inv(x.to_value(dimensionless)), unit=dimensionless)


# ==============================================================================


@register(lax.erf_p)
def _erf_p(x: Quantity) -> Quantity:
    # TODO: can this support non-dimensionless quantities?
    return Quantity(lax.erf(x.to_value(dimensionless)), unit=dimensionless)


# ==============================================================================


@register(lax.erfc_p)
def _erfc_p(x: Quantity) -> Quantity:
    # TODO: can this support non-dimensionless quantities?
    return Quantity(lax.erfc(x.to_value(dimensionless)), unit=dimensionless)


# ==============================================================================


@register(lax.exp2_p)
def _exp2_p(x: Quantity) -> Quantity:
    return Quantity(lax.exp2(x.to_value(dimensionless)), unit=dimensionless)


# ==============================================================================


@register(lax.exp_p)
def _exp_p(x: Quantity) -> Quantity:
    # TODO: more meaningful error message.
    return Quantity(lax.exp(x.to_value(dimensionless)), unit=dimensionless)


# ==============================================================================


@register(lax.expm1_p)
def _expm1_p(x: Quantity) -> Quantity:
    return Quantity(lax.expm1(x.to_value(dimensionless)), unit=dimensionless)


# ==============================================================================


@register(lax.fft_p)
def _fft_p(x: Quantity, *, fft_type: Any, fft_lengths: Any) -> Quantity:
    # TODO: what units can this support?
    return Quantity(
        lax.fft(x.to_value(dimensionless), fft_type, fft_lengths),
        unit=dimensionless,
    )


# ==============================================================================


@register(lax.floor_p)
def _floor_p(x: Quantity) -> Quantity:
    return replace(x, value=lax.floor(x.value))


# ==============================================================================


# used in `jnp.cross`
@register(lax.gather_p)
def _gather_p(
    operand: Quantity,
    start_indices: ArrayLike,
    *,
    dimension_numbers: GatherDimensionNumbers,
    slice_sizes: Shape,
    unique_indices: bool,
    indices_are_sorted: bool,
    mode: str | GatherScatterMode | None,
    fill_value: Any,
) -> Quantity:
    return Quantity(
        lax.gather_p.bind(
            operand.value,
            start_indices,
            dimension_numbers=dimension_numbers,
            slice_sizes=slice_sizes,
            unique_indices=unique_indices,
            indices_are_sorted=indices_are_sorted,
            mode=mode,
            fill_value=fill_value,
        ),
        unit=operand.unit,
    )


# ==============================================================================


@register(lax.ge_p)
def _ge_p_qq(x: Quantity, y: Quantity) -> ArrayLike:
    return lax.ge(x.value, y.to_value(x.unit))


@register(lax.ge_p)
def _ge_p_vq(x: ArrayLike, y: Quantity) -> ArrayLike:
    return lax.ge(x, y.to_value(dimensionless))


@register(lax.ge_p)
def _ge_p_qv(x: Quantity, y: ArrayLike) -> ArrayLike:
    if jnp.array_equal(y, 0):
        return lax.ge(x.value, y)
    return lax.ge(x.to_value(dimensionless), y)


# ==============================================================================


@register(lax.gt_p)
def _gt_p_qq(x: Quantity, y: Quantity) -> ArrayLike:
    return lax.gt(x.value, y.to_value(x.unit))


@register(lax.gt_p)
def _gt_p_vq(x: ArrayLike, y: Quantity) -> ArrayLike:
    return lax.gt(x, y.to_value(dimensionless))


@register(lax.gt_p)
def _gt_p_qv(x: Quantity, y: ArrayLike) -> ArrayLike:
    return lax.gt(x.to_value(dimensionless), y)


@register(lax.gt_p)
def _gt_p_qi(x: Quantity, y: int) -> ArrayLike:
    return lax.gt(x.to_value(dimensionless), y)


# ==============================================================================


@register(lax.igamma_grad_a_p)
def _igamma_grad_a_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.igamma_p)
def _igamma_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.igammac_p)
def _igammac_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.imag_p)
def _imag_p(x: Quantity) -> Quantity:
    return replace(x, value=lax.imag(x.value))


# ==============================================================================


@register(lax.infeed_p)
def _infeed_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.integer_pow_p)
def _integer_pow_p(x: Quantity, *, y: Any) -> Quantity:
    return Quantity(value=lax.integer_pow(x.value, y), unit=x.unit**y)


# ==============================================================================


# @register(lax.iota_p)
# def _iota_p(dtype: Quantity) -> Quantity:
#     raise NotImplementedError


# ==============================================================================


@register(lax.is_finite_p)
def _is_finite_p(x: Quantity) -> ArrayLike:
    return lax.is_finite(x.value)


# ==============================================================================


@register(lax.le_p)
def _le_p_qq(x: Quantity, y: Quantity) -> ArrayLike:
    return lax.le(x.value, y.to_value(x.unit))


@register(lax.le_p)
def _le_p_vq(x: ArrayLike, y: Quantity) -> ArrayLike:
    return lax.le(x, y.to_value(dimensionless))


@register(lax.le_p)
def _le_p_qv(x: Quantity, y: ArrayLike) -> ArrayLike:
    return lax.le(x.to_value(dimensionless), y)


# ==============================================================================


@register(lax.le_to_p)
def _le_to_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.lgamma_p)
def _lgamma_p(x: Quantity) -> Quantity:
    # TODO: handle non-dimensionless quantities.
    return Quantity(lax.lgamma(x.to_value(dimensionless)), unit=dimensionless)


# ==============================================================================


@register(lax.linear_solve_p)
def _linear_solve_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.log1p_p)
def _log1p_p(x: Quantity) -> Quantity:
    return Quantity(lax.log1p(x.to_value(dimensionless)), unit=dimensionless)


# ==============================================================================


@register(lax.log_p)
def _log_p(x: Quantity) -> Quantity:
    return Quantity(lax.log(x.to_value(dimensionless)), unit=dimensionless)


# ==============================================================================


@register(lax.logistic_p)
def _logistic_p(x: Quantity) -> Quantity:
    return Quantity(lax.logistic(x.to_value(dimensionless)), unit=dimensionless)


# ==============================================================================


@register(lax.lt_p)
def _lt_p_qq(x: Quantity, y: Quantity) -> ArrayLike:
    return lax.lt(x.value, y.to_value(x.unit))


@register(lax.lt_p)
def _lt_p_vq(x: ArrayLike, y: Quantity) -> ArrayLike:
    return lax.lt(x, y.to_value(dimensionless))


@register(lax.lt_p)
def _lt_p_qv(x: Quantity, y: ArrayLike) -> ArrayLike:
    return lax.lt(x.to_value(dimensionless), y)


# ==============================================================================


@register(lax.lt_to_p)
def _lt_to_p() -> ArrayLike:
    raise NotImplementedError


# ==============================================================================


@register(lax.max_p)
def _max_p_qq(x: Quantity, y: Quantity) -> Quantity:
    return Quantity(lax.max(x.value, y.to_value(x.unit)), unit=x.unit)


@register(lax.max_p)
def _max_p_vq(x: ArrayLike, y: Quantity) -> Quantity:
    return Quantity(lax.max(x, y.to_value(dimensionless)), unit=dimensionless)


@register(lax.max_p)
def _max_p_qv(x: Quantity, y: ArrayLike) -> Quantity:
    return Quantity(lax.max(x.to_value(dimensionless), y), unit=dimensionless)


# ==============================================================================


@register(lax.min_p)
def _min_p_qq(x: Quantity, y: Quantity) -> Quantity:
    return Quantity(lax.min(x.value, y.to_value(x.unit)), unit=x.unit)


@register(lax.min_p)
def _min_p_vq(x: ArrayLike, y: Quantity) -> Quantity:
    return Quantity(lax.min(x, y.to_value(dimensionless)), unit=dimensionless)


@register(lax.min_p)
def _min_p_qv(x: Quantity, y: ArrayLike) -> Quantity:
    return Quantity(lax.min(x.to_value(dimensionless), y), unit=dimensionless)


# ==============================================================================
# Multiplication


@register(lax.mul_p)
def _mul_p_qq(x: Quantity, y: Quantity) -> Quantity:
    unit = Unit(x.unit * y.unit)
    return Quantity(lax.mul(x.value, y.value), unit=unit)


@register(lax.mul_p)
def _mul_p_vq(x: ArrayLike, y: Quantity) -> Quantity:
    return Quantity(lax.mul(x, y.value), unit=y.unit)


@register(lax.mul_p)
def _mul_p_qv(x: Quantity, y: ArrayLike) -> Quantity:
    return Quantity(lax.mul(x.value, y), unit=x.unit)


# ==============================================================================


@register(lax.ne_p)
def _ne_p_qq(x: Quantity, y: Quantity) -> ArrayLike:
    return lax.ne(x.value, y.to_value(x.unit))


@register(lax.ne_p)
def _ne_p_vq(x: ArrayLike, y: Quantity) -> ArrayLike:
    return lax.ne(x, y.to_value(dimensionless))


@register(lax.ne_p)
def _ne_p_qv(x: Quantity, y: ArrayLike) -> ArrayLike:
    # special-case for scalar value=0, unit=dimensionless
    if y.shape == () and y == 0:
        return lax.ne(x.value, y)
    return lax.ne(x.to_value(dimensionless), y)


# ==============================================================================


@register(lax.neg_p)
def _neg_p(x: Quantity) -> Quantity:
    return replace(x, value=lax.neg(x.value))


# ==============================================================================


@register(lax.nextafter_p)
def _nextafter_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.not_p)
def _not_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.or_p)
def _or_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.outfeed_p)
def _outfeed_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.pad_p)
def _pad_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.pmax_p)
def _pmax_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.pmin_p)
def _pmin_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.polygamma_p)
def _polygamma_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.population_count_p)
def _population_count_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.pow_p)
def _pow_p_qq(x: Quantity, y: Quantity) -> Quantity:
    if y.unit != dimensionless:
        msg = f"power must be dimensionless, got {y.unit}"
        raise UnitTypeError(msg)

    y0 = y.value.flatten()[0]
    if not all(y.value == y0):
        msg = "power must be a scalar"
        raise ValueError(msg)

    return Quantity(value=lax.pow(x.value, y0), unit=x.unit**y0)


@register(lax.pow_p)
def _pow_p_qf(x: Quantity, y: int | float) -> Quantity:
    return Quantity(value=lax.pow(x.value, y), unit=x.unit**y)


# ==============================================================================


@register(lax.ppermute_p)
def _ppermute_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.psum_p)
def _psum_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.random_gamma_grad_p)
def _random_gamma_grad_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.real_p)
def _real_p(x: Quantity) -> Quantity:
    return replace(x, value=lax.real(x.value))


# ==============================================================================


@register(lax.reduce_and_p)
def _reduce_and_p(
    operand: Quantity,
    *,
    axes: Sequence[int],
) -> Any:
    return lax.reduce_and_p.bind(operand.value, axes=tuple(axes))


# ==============================================================================


@register(lax.reduce_max_p)
def _reduce_max_p(operand: Quantity, *, axes: Axes) -> Quantity:
    return Quantity(
        value=lax.reduce_max_p.bind(operand.value, axes=axes), unit=operand.unit
    )


# ==============================================================================


@register(lax.reduce_min_p)
def _reduce_min_p(operand: Quantity, *, axes: Axes) -> Quantity:
    return Quantity(lax.reduce_min_p.bind(operand.value, axes=axes), unit=operand.unit)


# ==============================================================================


@register(lax.reduce_or_p)
def _reduce_or_p(operand: Quantity, *, axes: Axes) -> Quantity:
    return Quantity(lax.reduce_or_p.bind(operand.value, axes=axes), unit=dimensionless)


# ==============================================================================


@register(lax.reduce_p)
def _reduce_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.reduce_precision_p)
def _reduce_precision_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.reduce_prod_p)
def _reduce_prod_p(operand: Quantity, *, axes: Axes) -> Quantity:
    return Quantity(
        lax.reduce_prod_p.bind(operand.value, axes=axes),
        unit=operand.unit ** prod(operand.shape[ax] for ax in axes),
    )


# ==============================================================================


@register(lax.reduce_sum_p)
def _reduce_sum_p(operand: Quantity, *, axes: Axes) -> Quantity:
    return Quantity(lax.reduce_sum_p.bind(operand.value, axes=axes), unit=operand.unit)


# ==============================================================================


@register(lax.reduce_window_max_p)
def _reduce_window_max_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.reduce_window_min_p)
def _reduce_window_min_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.reduce_window_p)
def _reduce_window_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.reduce_window_sum_p)
def _reduce_window_sum_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.reduce_xor_p)
def _reduce_xor_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.regularized_incomplete_beta_p)
def _regularized_incomplete_beta_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.rem_p)
def _rem_p(x: Quantity, y: Quantity) -> Quantity:
    return Quantity(lax.rem(x.value, y.to_value(x.unit)), unit=x.unit)


# ==============================================================================


@register(lax.reshape_p)
def _reshape_p(operand: Quantity, *, new_sizes: Any, dimensions: Any) -> Quantity:
    return Quantity(
        lax.reshape(operand.value, new_sizes, dimensions), unit=operand.unit
    )


# ==============================================================================


@register(lax.rev_p)
def _rev_p(operand: Quantity, *, dimensions: Any) -> Quantity:
    return replace(operand, value=lax.rev(operand.value, dimensions))


# ==============================================================================


@register(lax.rng_bit_generator_p)
def _rng_bit_generator_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.rng_uniform_p)
def _rng_uniform_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.round_p)
def _round_p(x: Quantity, *, rounding_method: Any) -> Quantity:
    return replace(x, value=lax.round(x.value, rounding_method))


# ==============================================================================


@register(lax.rsqrt_p)
def _rsqrt_p(x: Quantity) -> Quantity:
    return Quantity(lax.rsqrt(x.value), unit=x.unit ** (-1 / 2))


# ==============================================================================


@register(lax.scan_p)
def _scan_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.scatter_add_p)
def _scatter_add_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.scatter_max_p)
def _scatter_max_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.scatter_min_p)
def _scatter_min_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.scatter_mul_p)
def _scatter_mul_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.scatter_p)
def _scatter_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.select_and_gather_add_p)
def _select_and_gather_add_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.select_and_scatter_add_p)
def _select_and_scatter_add_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.select_and_scatter_p)
def _select_and_scatter_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.select_n_p)
def _select_n_p(which: Quantity, *cases: Quantity) -> Quantity:
    unit = cases[0].unit
    cases_ = (case.to_value(unit) for case in cases)
    return Quantity(lax.select_n(which.to_value(dimensionless), *cases_), unit=unit)


@register(lax.select_n_p)
def _select_n_p_vq(which: Quantity, case0: Quantity, case1: ArrayLike) -> Quantity:
    # encountered from jnp.hypot
    unit = case0.unit
    return Quantity(
        lax.select_n(which.to_value(dimensionless), case0.to_value(unit), case1),
        unit=unit,
    )


@register(lax.select_n_p)
def _select_n_p_jjq(which: ArrayLike, case0: ArrayLike, case1: Quantity) -> Quantity:
    # Used by a `xp.linalg.trace`
    unit = case1.unit
    return Quantity(lax.select_n(which, case0, case1.to_value(unit)), unit=unit)


@register(lax.select_n_p)
def _select_n_p_jqj(which: ArrayLike, case0: Quantity, case1: ArrayLike) -> Quantity:
    # Used by a `triu`
    unit = case0.unit
    return Quantity(lax.select_n(which, case0.to_value(unit), case1), unit=unit)


@register(lax.select_n_p)
def _select_n_p_jqq(which: ArrayLike, case0: Quantity, case1: Quantity) -> Quantity:
    # used by `jnp.hypot`
    unit = case0.unit
    return Quantity(
        lax.select_n(which, case0.to_value(unit), case1.to_value(unit)), unit=unit
    )


# ==============================================================================


@register(lax.sharding_constraint_p)
def _sharding_constraint_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.shift_left_p)
def _shift_left_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.shift_right_arithmetic_p)
def _shift_right_arithmetic_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.shift_right_logical_p)
def _shift_right_logical_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.sign_p)
def _sign_p(x: Quantity) -> ArrayLike:
    return lax.sign(x.value)


# ==============================================================================


@register(lax.sin_p)
def _sin_p(x: Quantity) -> Quantity:
    return Quantity(lax.sin(_to_value_rad_or_one(x)), unit=dimensionless)


# ==============================================================================


@register(lax.sinh_p)
def _sinh_p(x: Quantity) -> Quantity:
    return Quantity(lax.sinh(_to_value_rad_or_one(x)), unit=dimensionless)


# ==============================================================================


@register(lax.slice_p)
def _slice_p(
    operand: Quantity,
    *,
    start_indices: Any,
    limit_indices: Any,
    strides: Any,
) -> Quantity:
    return Quantity(
        lax.slice_p.bind(
            operand.value,
            start_indices=start_indices,
            limit_indices=limit_indices,
            strides=strides,
        ),
        unit=operand.unit,
    )


# ==============================================================================


# Called by `argsort`
@register(lax.sort_p)
def _sort_p_two_operands(
    operand0: Quantity,
    operand1: ArrayLike,
    *,
    dimension: int,
    is_stable: bool,
    num_keys: int,
) -> tuple[Quantity, Quantity]:
    out0, out1 = lax.sort_p.bind(
        operand0.value,
        operand1,
        dimension=dimension,
        is_stable=is_stable,
        num_keys=num_keys,
    )
    return Quantity(out0, unit=operand0.unit), Quantity(out1, unit=dimensionless)


# Called by `sort`
@register(lax.sort_p)
def _sort_p_one_operand(
    operand: Quantity, *, dimension: int, is_stable: bool, num_keys: int
) -> tuple[Quantity]:
    (out,) = lax.sort_p.bind(
        operand.value, dimension=dimension, is_stable=is_stable, num_keys=num_keys
    )
    return (Quantity(out, unit=operand.unit),)


# ==============================================================================


@register(lax.sqrt_p)
def _sqrt_p(x: Quantity) -> Quantity:
    return Quantity(lax.sqrt(x.value), unit=x.unit ** (1 / 2))


# ==============================================================================


@register(lax.squeeze_p)
def _squeeze_p(x: Quantity, *, dimensions: Any) -> Quantity:
    return Quantity(lax.squeeze(x.value, dimensions), unit=x.unit)


# ==============================================================================


@register(lax.stop_gradient_p)
def _stop_gradient_p(x: Quantity) -> Quantity:
    return replace(x, value=lax.stop_gradient(x.value))


# ==============================================================================
# Subtraction


@register(lax.sub_p)
def _sub_p_qq(x: Quantity, y: Quantity) -> Quantity:
    return Quantity(
        lax.sub(x.to_value(x.unit), y.to_value(x.unit)),
        unit=x.unit,
    )


@register(lax.sub_p)
def _sub_p_vq(x: ArrayLike, y: Quantity) -> Quantity:
    return Quantity(lax.sub(x, y.value), unit=y.unit)


@register(lax.sub_p)
def _sub_p_qv(x: Quantity, y: ArrayLike) -> Quantity:
    return Quantity(lax.sub(x.value, y), unit=x.unit)


# ==============================================================================


@register(lax.tan_p)
def _tan_p(x: Quantity) -> Quantity:
    return Quantity(lax.tan(_to_value_rad_or_one(x)), unit=dimensionless)


# ==============================================================================


@register(lax.tanh_p)
def _tanh_p(x: Quantity) -> Quantity:
    return Quantity(lax.tanh(_to_value_rad_or_one(x)), unit=dimensionless)


# ==============================================================================


@register(lax.top_k_p)
def _top_k_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.transpose_p)
def _transpose_p(operand: Quantity, *, permutation: Any) -> Quantity:
    return Quantity(lax.transpose(operand.value, permutation), unit=operand.unit)


# ==============================================================================


@register(lax.while_p)
def _while_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.xor_p)
def _xor_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.zeta_p)
def _zeta_p() -> Quantity:
    raise NotImplementedError
