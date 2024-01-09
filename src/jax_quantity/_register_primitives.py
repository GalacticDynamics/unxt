# pylint: disable=import-error, too-many-lines

__all__: list[str] = []

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any, TypeAlias, TypeVar

import jax
import jax.core
import jax.numpy as jnp
from astropy.units import (  # pylint: disable=no-name-in-module
    Unit,
    UnitBase,
    UnitTypeError,
    radian,
)
from astropy.units import dimensionless_unscaled as dimensionless
from jax import lax
from jaxtyping import ArrayLike
from quax import DenseArrayValue
from quax import register as register_
from quax._core import _QuaxTracer
from quax.zero import Zero

from ._core import Quantity, can_convert

T = TypeVar("T")

UnitClasses: TypeAlias = UnitBase


def register(primitive: jax.core.Primitive) -> Callable[[T], T]:
    """:func`quax.register`, but makes mypy happy."""
    return register_(primitive)


def _to_value_rad_or_one(q: Quantity) -> ArrayLike:
    return (
        q.to_value(radian) if can_convert(q.unit, radian) else q.to_value(dimensionless)
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
    unit = x.unit
    out = Quantity(lax.add(x.to_value(unit), y.to_value(unit)), unit=unit)
    jax.debug.print(
        "add_p_qq: {} + {} -> {}",
        str(x).replace("f64[]", str(x.value)),
        str(y).replace("f64[]", str(y.value)),
        str(out).replace("f64[]", str(out.value)),
    )
    return out


@register(lax.add_p)
def _add_p_vq(x: DenseArrayValue, y: Quantity) -> Quantity:
    # x = 0 is a special case
    jax.debug.print(
        "add_p_vq: {}, {}, {}, {}", type(x), x.array.value, x.array.value.value, type(y)
    )
    if jnp.array_equal(x.array, 0):
        return y

    # otherwise we can't add a quantity to a normal value
    msg = "Cannot add a non-quantity and quantity."
    raise ValueError(msg)


@register(lax.add_p)
def _add_p_qv(x: Quantity, y: DenseArrayValue) -> Quantity:
    jax.debug.print("add_p_qv: {}, {}", type(x), type(y.array))
    if isinstance(y.array, _QuaxTracer) and isinstance(y.array.value, Quantity):
        return x + y.array.value

    if isinstance(y.array, jax._src.interpreters.ad.JVPTracer):
        print(f"\t{x.unit} + {y.array.primal.value.unit}")
        return x + y.array.primal.value

    # y = 0 is a special case
    if jnp.array_equal(y.array, 0):
        return x

    # otherwise we can't add a normal value to a quantity
    msg = "Cannot add a quantity and non-quantity."
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


# TODO: return jax.Array. But `quax` is raising an error.
@register(lax.and_p)
def _and_p(x1: Quantity, x2: Quantity, /) -> Quantity:
    # IDK what to do about non-dimensionless quantities.
    if x1.unit != dimensionless or x2.unit != dimensionless:
        raise NotImplementedError
    return Quantity(x1.value & x2.value, unit=dimensionless)


# ==============================================================================


@register(lax.approx_top_k_p)
def _approx_top_k_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.argmax_p)
def _argmax_p(operand: Quantity, *, axes: Any, index_dtype: Any) -> Quantity:
    return replace(operand, value=lax.argmax(operand.value, axes[0], index_dtype))


# ==============================================================================


@register(lax.argmin_p)
def _argmin_p(operand: Quantity, *, axes: Any, index_dtype: Any) -> Quantity:
    return replace(operand, value=lax.argmin(operand.value, axes[0], index_dtype))


# ==============================================================================


@register(lax.asin_p)
def _asin_p(x: Quantity) -> Quantity:
    return replace(x, value=lax.asin(x.to_value(dimensionless)), unit=radian)


# ==============================================================================


@register(lax.asinh_p)
def _asinh_p(x: Quantity) -> Quantity:
    return replace(x, value=lax.asinh(x.to_value(dimensionless)), unit=radian)


# ==============================================================================


@register(lax.atan2_p)
def _atan2_p(x: Quantity, y: Quantity) -> Quantity:
    y_ = y.to_value(x.unit)
    return Quantity(lax.atan2(x.value, y_), unit=radian)


@register(lax.atan2_p)
def _atan2_p_vq(x: DenseArrayValue, y: Quantity) -> Quantity:
    y_ = y.to_value(dimensionless)
    return Quantity(lax.atan2(x, y_), unit=radian)


@register(lax.atan2_p)
def _atan2_p_qv(x: Quantity, y: DenseArrayValue) -> Quantity:
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
    return replace(
        operand,
        value=lax.broadcast_in_dim(operand.value, shape, broadcast_dimensions),
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
def _clamp_p_vqq(min: DenseArrayValue, x: Quantity, max: Quantity) -> Quantity:
    v = x.to_value(dimensionless)
    maxv = max.to_value(dimensionless)
    return replace(x, value=lax.clamp(min, v, maxv))


@register(lax.clamp_p)
def _clamp_p_qvq(min: Quantity, x: DenseArrayValue, max: Quantity) -> DenseArrayValue:
    minv = min.to_value(dimensionless)
    maxv = max.to_value(dimensionless)
    return DenseArrayValue(lax.clamp(minv, x, maxv))


@register(lax.clamp_p)
def _clamp_p_qqv(min: Quantity, x: Quantity, max: DenseArrayValue) -> Quantity:
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
    return Quantity(lax.div(x.value, y.to_value(x.unit)), unit=unit)


@register(lax.div_p)
def _div_p_vq(x: DenseArrayValue, y: Quantity) -> Quantity:
    return Quantity(lax.div(x, y.value), unit=1 / y.unit)


@register(lax.div_p)
def _div_p_qv(x: Quantity, y: DenseArrayValue) -> Quantity:
    return Quantity(lax.div(x.value, y), unit=x.unit)


# ==============================================================================


@register(lax.dot_general_p)  # TODO: implement
def _implemen() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.dynamic_slice_p)
def _dynamic_slice_p(
    operand: Quantity,
    start_indices: DenseArrayValue,
    dynamic_sizes: DenseArrayValue,
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
def _eq_p_qq(x: Quantity, y: Quantity) -> Quantity:
    return Quantity(lax.eq(x.value, y.to_value(x.unit)), unit=dimensionless)


@register(lax.eq_p)
def _eq_p_vq(x: DenseArrayValue, y: Quantity) -> Quantity:
    return Quantity(lax.eq(x.array, y.to_value(dimensionless)), unit=dimensionless)


@register(lax.eq_p)
def _eq_p_qv(x: Quantity, y: DenseArrayValue) -> Quantity:
    # special-case for all-0 values
    if jnp.all(y.array == 0) or jnp.all(jnp.isinf(y.array)):
        return Quantity(lax.eq(x.value, y.array), unit=dimensionless)
    return Quantity(lax.eq(x.to_value(dimensionless), y.array), unit=dimensionless)


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


@register(lax.gather_p)
def _gather_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.ge_p)
def _ge_p_qq(x: Quantity, y: Quantity) -> Quantity:
    return Quantity(lax.ge(x.value, y.to_value(x.unit)), unit=dimensionless)


@register(lax.ge_p)
def _ge_p_vq(x: DenseArrayValue, y: Quantity) -> Quantity:
    return Quantity(lax.ge(x, y.to_value(dimensionless)), unit=dimensionless)


@register(lax.ge_p)
def _ge_p_qv(x: Quantity, y: DenseArrayValue) -> Quantity:
    return Quantity(lax.ge(x.to_value(dimensionless), y), unit=dimensionless)


@register(lax.ge_p)
def _ge_p_qi(x: Quantity, y: int) -> Quantity:
    return Quantity(lax.ge(x.to_value(dimensionless), y), unit=dimensionless)


# ==============================================================================


@register(lax.gt_p)
def _gt_p_qq(x: Quantity, y: Quantity) -> Quantity:
    return Quantity(lax.gt(x.value, y.to_value(x.unit)), unit=dimensionless)


@register(lax.gt_p)
def _gt_p_vq(x: DenseArrayValue, y: Quantity) -> Quantity:
    return Quantity(lax.gt(x, y.to_value(dimensionless)), unit=dimensionless)


@register(lax.gt_p)
def _gt_p_qv(x: Quantity, y: DenseArrayValue) -> Quantity:
    return Quantity(lax.gt(x.to_value(dimensionless), y), unit=dimensionless)


@register(lax.gt_p)
def _gt_p_qi(x: Quantity, y: int) -> Quantity:
    return Quantity(lax.gt(x.to_value(dimensionless), y), unit=dimensionless)


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
    return replace(x, value=lax.integer_pow(x.value, y), unit=x.unit**y)


# ==============================================================================


# @register(lax.iota_p)
# def _iota_p(dtype: Quantity) -> Quantity:
#     raise NotImplementedError


# ==============================================================================


@register(lax.is_finite_p)
def _is_finite_p(x: Quantity) -> Quantity:
    return Quantity(value=lax.is_finite(x.value), unit=dimensionless)


# ==============================================================================


@register(lax.le_p)
def _le_p_qq(x: Quantity, y: Quantity) -> Quantity:
    return Quantity(lax.le(x.value, y.to_value(x.unit)), unit=dimensionless)


@register(lax.le_p)
def _le_p_vq(x: DenseArrayValue, y: Quantity) -> Quantity:
    return Quantity(lax.le(x, y.to_value(dimensionless)), unit=dimensionless)


@register(lax.le_p)
def _le_p_qv(x: Quantity, y: DenseArrayValue) -> Quantity:
    return Quantity(lax.le(x.to_value(dimensionless), y), unit=dimensionless)


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
def _lt_p_qq(x: Quantity, y: Quantity) -> Quantity:
    return Quantity(lax.lt(x.value, y.to_value(x.unit)), unit=dimensionless)


@register(lax.lt_p)
def _lt_p_vq(x: DenseArrayValue, y: Quantity) -> Quantity:
    return Quantity(lax.lt(x, y.to_value(dimensionless)), unit=dimensionless)


@register(lax.lt_p)
def _lt_p_qv(x: Quantity, y: DenseArrayValue) -> Quantity:
    return Quantity(lax.lt(x.to_value(dimensionless), y), unit=dimensionless)


# ==============================================================================


@register(lax.lt_to_p)
def _lt_to_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.max_p)
def _max_p_qq(x: Quantity, y: Quantity) -> Quantity:
    return Quantity(lax.max(x.value, y.to_value(x.unit)), unit=x.unit)


@register(lax.max_p)
def _max_p_vq(x: DenseArrayValue, y: Quantity) -> Quantity:
    return Quantity(lax.max(x, y.to_value(dimensionless)), unit=dimensionless)


@register(lax.max_p)
def _max_p_qv(x: Quantity, y: DenseArrayValue) -> Quantity:
    return Quantity(lax.max(x.to_value(dimensionless), y), unit=dimensionless)


# ==============================================================================


@register(lax.min_p)
def _min_p_qq(x: Quantity, y: Quantity) -> Quantity:
    return Quantity(lax.min(x.value, y.to_value(x.unit)), unit=x.unit)


@register(lax.min_p)
def _min_p_vq(x: DenseArrayValue, y: Quantity) -> Quantity:
    return Quantity(lax.min(x, y.to_value(dimensionless)), unit=dimensionless)


@register(lax.min_p)
def _min_p_qv(x: Quantity, y: DenseArrayValue) -> Quantity:
    return Quantity(lax.min(x.to_value(dimensionless), y), unit=dimensionless)


# ==============================================================================
# Multiplication


@register(lax.mul_p)
def _mul_p_qq(x: Quantity, y: Quantity) -> Quantity:
    unit = Unit(x.unit * y.unit)
    out = Quantity(lax.mul(x.value, y.value), unit=unit)
    jax.debug.print(
        "mul_p_qq: {}, {} -> {}",
        str(x).replace("f64[]", str(x.value)),
        str(y).replace("f64[]", str(y.value)),
        str(out).replace("f64[]", str(out.value)),
    )
    return out


@register(lax.mul_p)
def _mul_p_vq(x: DenseArrayValue, y: Quantity) -> Quantity:
    out = Quantity(lax.mul(x.array, y.value), unit=y.unit)
    jax.debug.print(
        "mul_p_vq: {}, {} -> {}",
        x.array,
        str(y).replace("f64[]", str(y.value)),
        str(out).replace("f64[]", str(out.value)),
    )
    return out


@register(lax.mul_p)
def _mul_p_qv(x: Quantity, y: DenseArrayValue) -> Quantity:
    # FIXME: this is a weird hack. See
    # https://github.com/patrick-kidger/quax/issues/5

    if isinstance(y.array, _QuaxTracer) and isinstance(y.array.value, Quantity):
        jax.debug.print("mul_p_qt: {}, {}", type(x), type(y.array.value))
        out = x * y.array.value

    elif isinstance(y.array, jax.Array):
        out = Quantity(lax.mul(x.value, y.array), unit=x.unit)
        jax.debug.print(
            "mul_p_qv: {}, {}",
            str(x).replace("f64[]", str(x.value)),
            type(y.array),
            # str(out).replace("f64[]", str(out.value)),
        )
    else:
        raise NotImplementedError
    return out


# ==============================================================================


@register(lax.ne_p)
def _ne_p_qq(x: Quantity, y: Quantity) -> Quantity:
    return Quantity(lax.ne(x.value, y.to_value(x.unit)), unit=dimensionless)


@register(lax.ne_p)
def _ne_p_vq(x: DenseArrayValue, y: Quantity) -> Quantity:
    return Quantity(lax.ne(x, y.to_value(dimensionless)), unit=dimensionless)


@register(lax.ne_p)
def _ne_p_qv(x: Quantity, y: DenseArrayValue) -> Quantity:
    # special-case for scalar value=0, unit=dimensionless
    if y.shape == () and y.array == 0:
        return Quantity(lax.ne(x.value, y), unit=dimensionless)
    return Quantity(lax.ne(x.to_value(dimensionless), y), unit=dimensionless)


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
def _reduce_max_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.reduce_min_p)
def _reduce_min_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.reduce_or_p)
def _reduce_or_p() -> Quantity:
    raise NotImplementedError


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
def _reduce_prod_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.reduce_sum_p)
def _reduce_sum_p() -> Quantity:
    raise NotImplementedError


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
    return replace(operand, value=lax.reshape(operand.value, new_sizes, dimensions))


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
    return Quantity(
        lax.select_n(
            which.to_value(dimensionless), *(case.to_value(unit) for case in cases)
        ),
        unit=unit,
    )


@register(lax.select_n_p)
def _select_n_p_jzq(which: DenseArrayValue, case0: Zero, case1: Quantity) -> Quantity:
    unit = case1.unit
    return Quantity(
        lax.select_n(which, case0.materialise(), case1.to_value(unit)), unit=unit
    )


@register(lax.select_n_p)
def _select_n_p_jqz(which: DenseArrayValue, case0: Quantity, case1: Zero) -> Quantity:
    unit = case0.unit
    return Quantity(
        lax.select_n(which, case0.to_value(unit), case1.materialise()), unit=unit
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
def _sign_p(x: Quantity) -> Quantity:
    return Quantity(lax.sign(x.value), unit=dimensionless)


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


@register(lax.sort_p)
def _sort_p() -> Quantity:
    raise NotImplementedError


# ==============================================================================


@register(lax.sqrt_p)
def _sqrt_p(x: Quantity) -> Quantity:
    return Quantity(lax.sqrt(x.value), unit=x.unit ** (1 / 2))


# ==============================================================================


@register(lax.squeeze_p)
def _squeeze_p(x: Quantity, *, dimensions: Any) -> Quantity:
    return replace(x, value=lax.squeeze(x.value, dimensions))


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
def _sub_p_vq(x: DenseArrayValue, y: Quantity) -> Quantity:
    return Quantity(lax.sub(x, y.value), unit=y.unit)


@register(lax.sub_p)
def _sub_p_qv(x: Quantity, y: DenseArrayValue) -> Quantity:
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
    return replace(operand, value=lax.transpose(operand.value, permutation))


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
