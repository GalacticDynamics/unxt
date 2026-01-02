"""Test the Array API."""
# pylint: disable=import-error, too-many-lines

import re

import jax.dlpack
import pytest
from jax import Array
from jax._src.numpy.setops import (
    _UniqueAllResult,
    _UniqueCountsResult,
    _UniqueInverseResult,
)
from jax.numpy import iinfo as IInfo  # noqa: N812
from numpy import finfo as FInfo  # noqa: N812

import quaxed.numpy as jnp

import unxt as u

# =============================================================================
# Constants


def test_e():
    """Test `e`."""
    assert not isinstance(jnp.e, u.Q)


def test_inf():
    """Test `inf`."""
    assert not isinstance(jnp.inf, u.Q)


def test_nan():
    """Test `nan`."""
    assert not isinstance(jnp.nan, u.Q)


def test_newaxis():
    """Test `newaxis`."""
    assert not isinstance(jnp.newaxis, u.Q)


def test_pi():
    """Test `pi`."""
    assert not isinstance(jnp.pi, u.Q)


# =============================================================================
# Creation functions

# -----------------------------------------------
# arange


def test_arange_start():
    """Test `arange`."""
    # -----------------------
    got = jnp.arange(u.Q(10, "m"))
    exp = u.Q(jnp.arange(10), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)

    # -----------------------
    got = jnp.arange(start=u.Q(10, "m"))

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_arange_stop():
    """Test `arange`."""
    start = u.Q(2, "m")
    stop = u.Q(10, "km")
    got = jnp.arange(start, stop)
    exp = u.Q(jnp.arange(start.value, u.ustrip(start.unit, stop)), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_arange_step():
    """Test `arange`."""
    start = u.Q(2, "m")
    stop = u.Q(10, "km")
    step = u.Q(2, "m")
    got = jnp.arange(start, stop, step)
    exp = u.Q(
        jnp.arange(start.value, u.ustrip(start.unit, stop), u.ustrip(start.unit, step)),
        "m",
    )

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


# -----------------------------------------------


def test_asarray():
    """Test `asarray`."""
    # TODO: test the dtype, device, copy arguments
    x = [1, 2, 3]
    got = jnp.asarray(u.Q(x, "m"))
    exp = u.Q(jnp.asarray(x), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


# -----------------------------------------------


@pytest.mark.xfail(reason="returns a jax.Array")
def test_empty():
    """Test `empty`."""
    # TODO: test the dtype, device arguments
    got = jnp.empty((2, 3))
    assert isinstance(got, u.Q)


# -----------------------------------------------


def test_empty_like():
    """Test `empty_like`."""
    x = jnp.asarray([1, 2, 3], dtype=float)
    q = u.Q(x, "m")
    got = jnp.empty_like(q)
    exp = u.Q(jnp.empty_like(x), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


# -----------------------------------------------


@pytest.mark.xfail(reason="returns a jax.Array")
def test_eye():
    """Test `eye`."""
    got = jnp.eye(3)

    assert isinstance(got, u.Q)


# -----------------------------------------------


def test_from_dlpack():
    """Test `from_dlpack`."""
    # Test that from_dlpack works with quantity values
    # from_dlpack expects an object with __dlpack__ method, not a capsule
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "m")

    # Convert using the __dlpack__ protocol - pass the array object itself
    got_array = jax.dlpack.from_dlpack(x.value)

    # Result should be a plain array with same data
    assert isinstance(got_array, Array)
    assert jnp.array_equal(got_array, x.value)


# -----------------------------------------------


@pytest.mark.xfail(reason="returns a jax.Array")
def test_full():
    """Test `full`."""
    got = jnp.full((2, 3), 1.0)

    assert isinstance(got, u.Q)


# -----------------------------------------------


def test_full_like_single_arg():
    """Test `full_like`."""
    x = jnp.asarray([1, 2, 3], dtype=float)
    q = u.Q(x, "m")
    got = jnp.full_like(q, fill_value=1.0)
    exp = u.Q(jnp.full_like(x, fill_value=1.0), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_full_like_double_arg():
    """Test `full_like`."""
    x = jnp.asarray([1, 2, 3], dtype=float)
    q = u.Q(x, "m")
    got = jnp.full_like(q, 1.0)
    exp = u.Q(jnp.full_like(x, 1.0), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_full_like_dtype():
    """Test `full_like`."""
    x = jnp.asarray([1, 2, 3], dtype=float)
    q = u.Q(x, "m")
    got = jnp.full_like(q, 1.0, dtype=int)
    exp = u.Q(jnp.full_like(q.value, 1.0, dtype=int), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)
    assert got.value.dtype == exp.value.dtype


# -----------------------------------------------


def test_linspace():
    """Test `linspace`."""
    # TODO: test the dtype, device, endpoint arguments
    got = jnp.linspace(u.Q(0.0, "m"), u.Q(10.0, "m"), 11)
    exp = u.Q(jnp.linspace(0.0, 10.0, 11), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


# -----------------------------------------------


def test_meshgrid():
    """Test `meshgrid`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "m")
    y = u.Q(jnp.asarray([4, 5, 6], dtype=float), "m")

    got1, got2 = jnp.meshgrid(x, y)
    exp1, exp2 = jnp.meshgrid(x.value, y.value)

    assert isinstance(got1, u.Q)
    assert jnp.array_equal(got1.value, exp1)

    assert isinstance(got2, u.Q)
    assert jnp.array_equal(got2.value, exp2)


# -----------------------------------------------


@pytest.mark.xfail(reason="returns a jax.Array")
def test_ones():
    """Test `ones`."""
    assert isinstance(jnp.ones((2, 3)), u.Q)


# -----------------------------------------------


def test_ones_like():
    """Test `ones_like`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "m")
    got = jnp.ones_like(x)
    exp = u.Q(jnp.ones_like(x.value), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


# -----------------------------------------------


def test_tril():
    """Test `tril`."""
    x = u.Q([[1, 2, 3], [4, 5, 6], [7, 8, 9]], "m")
    got = jnp.tril(x)
    exp = u.Q(jnp.tril(x.value), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


# -----------------------------------------------


def test_triu():
    """Test `triu`."""
    x = u.Q([[1, 2, 3], [4, 5, 6], [7, 8, 9]], "m")
    got = jnp.triu(x)
    exp = u.Q(jnp.triu(x.value), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


# -----------------------------------------------


@pytest.mark.xfail(reason="returns a jax.Array")
def test_zeros():
    """Test `zeros`."""
    assert isinstance(jnp.zeros((2, 3)), u.Q)


# -----------------------------------------------


def test_zeros_like():
    """Test `zeros_like`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "m")
    got = jnp.zeros_like(x)
    exp = u.Q(jnp.zeros_like(x.value), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


# =============================================================================
# Data-type functions


def test_astype():
    """Test `astype`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "m")
    got = jnp.astype(x, jnp.float32)
    exp = u.Q(jnp.asarray(x.value, dtype=jnp.float32), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_can_cast():
    """Test `can_cast`."""
    # can_cast should work the same with or without quantities
    # since it only checks dtypes
    x = u.Q(jnp.asarray([1, 2, 3], dtype=jnp.int32), "m")

    # Test with quantity type
    got = jnp.can_cast(x, jnp.float32)
    exp = jnp.can_cast(x.value, jnp.float32)
    assert got == exp

    # Test with explicit dtypes
    got = jnp.can_cast(jnp.int32, jnp.float64)
    exp = jnp.can_cast(jnp.int32, jnp.float64)
    assert got == exp


def test_finfo():
    """Test `finfo`."""
    got = jnp.finfo(jnp.float32)
    exp = jnp.finfo(jnp.float32)

    assert isinstance(got, FInfo)
    for attr in ("bits", "eps", "max", "min", "smallest_normal", "dtype"):
        assert getattr(got, attr) == getattr(exp, attr)


def test_iinfo():
    """Test `iinfo`."""
    got = jnp.iinfo(jnp.int32)
    exp = jnp.iinfo(jnp.int32)

    assert isinstance(got, IInfo)
    for attr in ("kind", "bits", "min", "max", "dtype"):
        assert getattr(got, attr) == getattr(exp, attr)


def test_isdtype():
    """Test `isdtype`."""
    # True by definition


def test_result_type():
    """Test `result_type`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "m")
    y = u.Q(jnp.asarray([4, 5, 6], dtype=float), "m")
    got = jnp.result_type(x, y)
    exp = jnp.result_type(x.value, y.value)

    assert isinstance(got, jnp.dtype)
    assert got == exp


# =============================================================================
# Elementwise functions


def test_abs():
    """Test `abs`."""
    x = u.Q([-1, 2, -3], "m")
    got = jnp.abs(x)
    exp = u.Q(jnp.abs(x.value), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_acos():
    """Test `acos`."""
    x = u.Q(jnp.asarray([-1, 0, 1], dtype=float), "")
    got = jnp.acos(x)
    exp = u.Q(jnp.acos(x.value), "rad")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_acosh():
    """Test `acosh`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "")
    got = jnp.acosh(x)
    exp = u.Q(jnp.acosh(x.value), "rad")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_add():
    """Test `add`."""
    # Adding two quantities
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "m")
    y = u.Q(jnp.asarray([4, 5, 6], dtype=float), "m")
    got = jnp.add(x, y)
    exp = u.Q(jnp.add(x.value, y.value), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)

    # Adding a quantity and non-quantity
    match = re.escape("'m' (length) and '' (dimensionless) are not convertible")
    with pytest.raises(ValueError, match=match):
        jnp.add(x.value, y)

    with pytest.raises(ValueError, match=match):
        jnp.add(x, y.value)

    # Add a non-quantity and dimensionless quantity
    got = jnp.add(x.value, u.Q(1.0, ""))
    exp = u.Q(x.value + 1, "")
    assert jnp.array_equal(got.value, exp.value)

    got = jnp.add(u.Q(1.0, ""), y.value)
    exp = u.Q(1 + y.value, "")
    assert jnp.array_equal(got.value, exp.value)


def test_asin():
    """Test `asin`."""
    x = u.Q(jnp.asarray([-1, 0, 1], dtype=float), "")
    got = jnp.asin(x)
    exp = u.Q(jnp.asin(x.value), "rad")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_asinh():
    """Test `asinh`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "")
    got = jnp.asinh(x)
    exp = u.Q(jnp.asinh(x.value), "rad")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_atan():
    """Test `atan`."""
    x = u.Q(jnp.asarray([-1, 0, 1], dtype=float), "")
    got = jnp.atan(x)
    exp = u.Q(jnp.atan(x.value), "rad")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_atan2():
    """Test `atan2`."""
    x = u.Q(jnp.asarray([-1, 0, 1], dtype=float), "m")
    y = u.Q(jnp.asarray([4, 5, 6], dtype=float), "m")
    got = jnp.atan2(x, y)
    exp = u.Q(jnp.atan2(x.value, y.value), "rad")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_atanh():
    """Test `atanh`."""
    x = u.Q(jnp.asarray([-1, 0, 1], dtype=float), "")
    got = jnp.atanh(x)
    exp = u.Q(jnp.atanh(x.value), "rad")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_bitwise_and():
    """Test `bitwise_and`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=int), "")
    y = u.Q(jnp.asarray([4, 5, 6], dtype=int), "")
    got = jnp.bitwise_and(x, y)
    exp = jnp.bitwise_and(x.value, y.value)

    assert isinstance(got, Array)
    assert jnp.array_equal(got, exp)


def test_bitwise_left_shift():
    """Test `bitwise_left_shift`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=int), "")
    y = u.Q(jnp.asarray([4, 5, 6], dtype=int), "")
    got = jnp.bitwise_left_shift(x, y)
    exp = u.Q(jnp.bitwise_left_shift(x.value, y.value), "")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_bitwise_invert():
    """Test `bitwise_invert`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=int), "")
    got = jnp.bitwise_invert(x)
    exp = u.Q(jnp.bitwise_invert(x.value), "")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_bitwise_or():
    """Test `bitwise_or`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=int), "")
    y = u.Q(jnp.asarray([4, 5, 6], dtype=int), "")
    got = jnp.bitwise_or(x, y)
    exp = u.Q(jnp.bitwise_or(x.value, y.value), "")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_bitwise_right_shift():
    """Test `bitwise_right_shift`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=int), "")
    y = u.Q(jnp.asarray([4, 5, 6], dtype=int), "")
    got = jnp.bitwise_right_shift(x, y)
    exp = u.Q(jnp.bitwise_right_shift(x.value, y.value), "")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_bitwise_xor():
    """Test `bitwise_xor`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=int), "")
    y = u.Q(jnp.asarray([4, 5, 6], dtype=int), "")
    got = jnp.bitwise_xor(x, y)
    exp = u.Q(jnp.bitwise_xor(x.value, y.value), "")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_ceil():
    """Test `ceil`."""
    x = u.Q([1.1, 2.2, 3.3], "m")
    got = jnp.ceil(x)
    exp = u.Q(jnp.ceil(x.value), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_conj():
    """Test `conj`."""
    x = u.Q([1 + 2j, 3 + 4j], "m")
    got = jnp.conj(x)
    exp = u.Q(jnp.conj(x.value), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_cos():
    """Test `cos`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "rad")
    got = jnp.cos(x)
    exp = u.Q(jnp.cos(x.value), "")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_cosh():
    """Test `cosh`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "rad")
    got = jnp.cosh(x)
    exp = u.Q(jnp.cosh(x.value), "")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_divide():
    """Test `divide`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "m")
    y = u.Q(jnp.asarray([4, 5, 6], dtype=float), "km")
    got = jnp.divide(x, y)
    exp = u.Q(jnp.divide(x.value, y.value), "m / km")

    assert isinstance(got, u.Q)
    assert got.unit.is_equivalent(exp.unit)
    assert jnp.array_equal(got.value, exp.value)


def test_equal():
    """Test `equal`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "m")
    y = u.Q(jnp.asarray([4, 5, 6], dtype=float), "m")
    got = jnp.equal(x, y)
    exp = jnp.equal(x.value, y.value)

    assert isinstance(got, Array)
    assert jnp.array_equal(got, exp)


def test_exp():
    """Test `exp`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "")
    got = jnp.exp(x)
    exp = u.Q(jnp.exp(x.value), "")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_expm1():
    """Test `expm1`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "")
    got = jnp.expm1(x)
    exp = u.Q(jnp.expm1(x.value), "")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_floor():
    """Test `floor`."""
    x = u.Q([1.1, 2.2, 3.3], "m")
    got = jnp.floor(x)
    exp = u.Q(jnp.floor(x.value), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_floor_divide():
    """Test `floor_divide`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "m")
    y = u.Q(jnp.asarray([4, 5, 6], dtype=float), "m")
    got = jnp.floor_divide(x, y)
    exp = u.Q(jnp.floor_divide(x.value, y.value), "m / m")

    assert isinstance(got, u.Q)
    assert got.unit.is_equivalent(exp.unit)
    assert jnp.array_equal(got.value, exp.value)


def test_greater():
    """Test `greater`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "m")
    y = u.Q(jnp.asarray([4, 5, 6], dtype=float), "m")
    got = jnp.greater(x, y)
    exp = jnp.greater(x.value, y.value)

    assert isinstance(got, Array)
    assert jnp.array_equal(got, exp)


def test_greater_equal():
    """Test `greater_equal`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "m")
    y = u.Q(jnp.asarray([4, 5, 6], dtype=float), "m")
    got = jnp.greater_equal(x, y)
    exp = jnp.greater_equal(x.value, y.value)

    assert isinstance(got, Array)
    assert jnp.array_equal(got, exp)


def test_imag():
    """Test `imag`."""
    x = u.Q([1 + 2j, 3 + 4j], "m")
    got = jnp.imag(x)
    exp = u.Q(jnp.imag(x.value), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_isfinite():
    """Test `isfinite`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "m")
    got = jnp.isfinite(x)
    exp = jnp.isfinite(x.value)

    assert isinstance(got, Array)
    assert jnp.array_equal(got, exp)


def test_isinf():
    """Test `isinf`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "m")
    got = jnp.isinf(x)
    exp = jnp.isinf(x.value)

    assert isinstance(got, Array)
    assert jnp.array_equal(got, exp)


def test_isnan():
    """Test `isnan`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "m")
    got = jnp.isnan(x)
    exp = jnp.isnan(x.value)

    assert isinstance(got, Array)
    assert jnp.array_equal(got, exp)


def test_less():
    """Test `less`."""
    x = u.Q([1, 5, 3], "m")
    y = u.Q([4, 2, 6], "m")
    got = jnp.less(x, y)
    exp = jnp.less(x.value, y.value)

    assert isinstance(got, Array)
    assert jnp.array_equal(got, exp)


def test_less_equal():
    """Test `less_equal`."""
    x = u.Q([1, 5, 3], "m")
    y = u.Q([4, 2, 6], "m")
    got = jnp.less_equal(x, y)
    exp = jnp.less_equal(x.value, y.value)

    assert isinstance(got, Array)
    assert jnp.array_equal(got, exp)


def test_log():
    """Test `log`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "")
    got = jnp.log(x)
    exp = u.Q(jnp.log(x.value), "")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_log1p():
    """Test `log1p`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "")
    got = jnp.log1p(x)
    exp = u.Q(jnp.log1p(x.value), "")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_log2():
    """Test `log2`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "")
    got = jnp.log2(x)
    exp = u.Q(jnp.log2(x.value), "")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_log10():
    """Test `log10`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "")
    got = jnp.log10(x)
    exp = u.Q(jnp.log10(x.value), "")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.allclose(got, exp, atol=u.Q(1e-8, ""))


def test_logaddexp():
    """Test `logaddexp`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "")
    y = u.Q(jnp.asarray([4, 5, 6], dtype=float), "")
    got = jnp.logaddexp(x, y)
    exp = u.Q(jnp.logaddexp(x.value, y.value), "")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_logical_and():
    """Test `logical_and`."""
    x = u.Q([True, False, True], "")
    y = u.Q([False, True, False], "")
    got = jnp.logical_and(x, y)
    exp = jnp.logical_and(x.value, y.value)

    assert isinstance(got, Array)
    assert jnp.array_equal(got, exp)


def test_logical_not():
    """Test `logical_not`."""
    x = u.Q([True, False, True], "")
    got = jnp.logical_not(x)
    exp = u.Q(jnp.logical_not(x.value), "")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_logical_or():
    """Test `logical_or`."""
    x = u.Q([True, False, True], "")
    y = u.Q([False, True, False], "")
    got = jnp.logical_or(x, y)
    exp = u.Q(jnp.logical_or(x.value, y.value), "")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_logical_xor():
    """Test `logical_xor`."""
    x = u.Q([True, False, True], "")
    y = u.Q([False, True, False], "")
    got = jnp.logical_xor(x, y)
    exp = u.Q(jnp.logical_xor(x.value, y.value), "")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_multiply():
    """Test `multiply`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "m")
    y = u.Q(jnp.asarray([4, 5, 6], dtype=float), "m")
    got = jnp.multiply(x, y)
    exp = u.Q(jnp.multiply(x.value, y.value), "m2")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_negative():
    """Test `negative`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "m")
    got = jnp.negative(x)
    exp = u.Q(jnp.negative(x.value), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_not_equal():
    """Test `not_equal`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "m")
    y = u.Q(jnp.asarray([4, 2, 6], dtype=float), "m")
    got = jnp.not_equal(x, y)
    exp = jnp.not_equal(x.value, y.value)

    assert isinstance(got, Array)
    assert jnp.array_equal(got, exp)


def test_positive():
    """Test `positive`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "m")
    got = jnp.positive(x)
    exp = u.Q(jnp.positive(x.value), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_pow_quantity_power():
    """Test `pow`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "m")
    y = u.Q(jnp.asarray([4], dtype=float), "")
    got = jnp.pow(x, y)
    exp = u.Q(jnp.pow(x.value, y.value), "m4")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_pow():
    """Test `pow`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "m")
    y = jnp.asarray([4], dtype=float)
    got = jnp.pow(x, y)
    exp = u.Q(jnp.pow(x.value, y), "m4")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_real():
    """Test `real`."""
    x = u.Q([1 + 2j, 3 + 4j], "m")
    got = jnp.real(x)
    exp = u.Q(jnp.real(x.value), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_remainder():
    """Test `remainder`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "m")
    y = u.Q(jnp.asarray([4, 5, 6], dtype=float), "m")
    got = jnp.remainder(x, y)
    exp = u.Q(jnp.remainder(x.value, y.value), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_round():
    """Test `round`."""
    x = u.Q([1.1, 2.2, 3.3], "m")
    got = jnp.round(x)
    exp = u.Q(jnp.round(x.value), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_sign():
    """Test `sign`."""
    x = u.Q([-1, 2, -3], "m")
    got = jnp.sign(x)
    exp = jnp.sign(x.value)

    assert isinstance(got, Array)
    assert jnp.array_equal(got, exp)


def test_sin():
    """Test `sin`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "rad")
    got = jnp.sin(x)
    exp = u.Q(jnp.sin(x.value), "")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_sinh():
    """Test `sinh`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "rad")
    got = jnp.sinh(x)
    exp = u.Q(jnp.sinh(x.value), "")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_square():
    """Test `square`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "m")
    got = jnp.square(x)
    exp = u.Q(jnp.square(x.value), "m2")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_sqrt():
    """Test `sqrt`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "m")
    got = jnp.sqrt(x)
    exp = u.Q(jnp.sqrt(x.value), u.unit("m(1/2)"))

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_subtract():
    """Test `subtract`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "m")
    y = u.Q(jnp.asarray([4, 5, 6], dtype=float), "m")
    got = jnp.subtract(x, y)
    exp = u.Q(jnp.subtract(x.value, y.value), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_tan():
    """Test `tan`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "rad")
    got = jnp.tan(x)
    exp = u.Q(jnp.tan(x.value), "")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_tanh():
    """Test `tanh`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "rad")
    got = jnp.tanh(x)
    exp = u.Q(jnp.tanh(x.value), "")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_trunc():
    """Test `trunc`."""
    x = u.Q([1.1, 2.2, 3.3], "m")
    got = jnp.trunc(x)
    exp = u.Q(jnp.trunc(x.value), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


# =============================================================================
# Indexing functions


def test_take():
    """Test `take`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "m")
    indices = jnp.asarray([0, 1, 2], dtype=int)
    got = jnp.take(x, indices)
    exp = u.Q(jnp.take(x.value, indices, axis=None), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


# =============================================================================
# Linear algebra functions


def test_matmul():
    """Test `matmul`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "m")
    y = u.Q(jnp.asarray([4, 5, 6], dtype=float), "m")
    got = jnp.matmul(x, y)
    exp = u.Q(jnp.matmul(x.value, y.value), "m2")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_matrix_transpose():
    """Test `matrix_transpose`."""
    x = u.Q(jnp.asarray([[1, 2, 3], [4, 5, 6]], dtype=float), "m")
    got = jnp.matrix_transpose(x)
    exp = u.Q(jnp.matrix_transpose(x.value), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_tensordot():
    """Test `tensordot`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "m")
    y = u.Q(jnp.asarray([4, 5, 6], dtype=float), "m")
    axes = 1
    got = jnp.tensordot(x, y, axes=axes)
    exp = u.Q(jnp.tensordot(x.value, y.value, axes=axes), "m2")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_vecdot():
    """Test `vecdot`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "m")
    y = u.Q(jnp.asarray([4, 5, 6], dtype=float), "m")
    got = jnp.vecdot(x, y)
    exp = u.Q(jnp.vecdot(x.value, y.value), "m2")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


# =============================================================================
# Manipulation functions


def test_broadcast_arrays():
    """Test `broadcast_arrays`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "m")
    y = u.Q(jnp.asarray([4], dtype=float), "s")
    got = jnp.broadcast_arrays(x, y)
    exp = jnp.broadcast_arrays(x.value, y.value)

    assert isinstance(got, tuple | list)
    assert len(got) == len(exp)
    for got_, exp_ in zip(got, exp, strict=True):
        assert isinstance(got_, u.Q)
        assert jnp.array_equal(got_.value, exp_)


def test_broadcast_to():
    """Test `broadcast_to`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "m")
    shape = (2, 3)
    got = jnp.broadcast_to(x, shape)
    exp = u.Q(jnp.broadcast_to(x.value, shape), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_concat():
    """Test `concat`."""
    # TODO: test the axis argument
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "m")
    y = u.Q(jnp.asarray([4], dtype=float), "m")
    got = jnp.concat((x, y))
    exp = u.Q(jnp.concat((x.value, y.value)), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_expand_dims():
    """Test `expand_dims`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "m")
    got = jnp.expand_dims(x, axis=0)
    exp = u.Q(jnp.expand_dims(x.value, axis=0), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_flip():
    """Test `flip`."""
    x = u.Q(jnp.asarray([[1, 2, 3], [4, 5, 6]], dtype=float), "m")
    got = jnp.flip(x)
    exp = u.Q(jnp.flip(x.value), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_permute_dims():
    """Test `permute_dims`."""
    x = u.Q(jnp.asarray([[1, 2, 3], [4, 5, 6]], dtype=float), "m")
    got = jnp.permute_dims(x, (1, 0))
    exp = u.Q(jnp.permute_dims(x.value, (1, 0)), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_reshape():
    """Test `reshape`."""
    x = u.Q(jnp.asarray([[1, 2, 3], [4, 5, 6]], dtype=float), "m")
    got = jnp.reshape(x, (3, 2))
    exp = u.Q(jnp.reshape(x.value, (3, 2)), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_roll():
    """Test `roll`."""
    x = u.Q(jnp.asarray([[1, 2, 3], [4, 5, 6]], dtype=float), "m")
    got = jnp.roll(x, shift=1, axis=0)
    exp = u.Q(jnp.roll(x.value, shift=1, axis=0), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_squeeze():
    """Test `squeeze`."""
    x = u.Q(jnp.asarray([[[0], [1], [2]]], dtype=float), "m")
    got = jnp.squeeze(x, axis=(0, 2))
    exp = u.Q(jnp.squeeze(x.value, axis=(0, 2)), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_stack():
    """Test `stack`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "m")
    y = u.Q(jnp.asarray([4, 5, 6], dtype=float), "m")
    got = jnp.stack((x, y))
    exp = u.Q(jnp.stack((x.value, y.value)), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


# =============================================================================
# Searching functions


def test_argmax():
    """Test `argmax`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "m")
    got = jnp.argmax(x)
    exp = jnp.argmax(x.value)

    assert jnp.array_equal(got, exp)


def test_argmin():
    """Test `argmin`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "m")
    got = jnp.argmin(x)
    exp = jnp.argmin(x.value)
    assert jnp.array_equal(got, exp)


@pytest.mark.xfail(
    reason="nonzero requires static size argument in JAX transformations"
)
def test_nonzero_dynamic_size():
    """Test `nonzero`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "m")
    (got,) = jnp.nonzero(x)
    (exp,) = u.Q(jnp.nonzero(x.value), "")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp)


@pytest.mark.xfail(reason="TODO: returns array")
def test_nonzero_set_size():
    """Test `nonzero`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "m")
    (got,) = jnp.nonzero(x, size=len(x.value))
    (exp,) = u.Q(jnp.nonzero(x.value, size=len(x.value)), "")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp)


def test_where():
    """Test `where`."""
    condition = u.Q(jnp.asarray([True, False, True]), "")
    y = u.Q(jnp.asarray([1, 2, 3], dtype=float), "m")
    z = u.Q(jnp.asarray([4, 5, 6], dtype=float), "m")
    got = jnp.where(condition, y, z)
    exp = u.Q(jnp.where(condition.value, y.value, z.value), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


# =============================================================================
# Set functions


@pytest.mark.xfail(reason="value is not a Quantity")
def test_unique_all():
    """Test `unique_all`."""
    x = u.Q(jnp.asarray([1, 2, 2, 3, 3, 3], dtype=float))
    got = jnp.unique_all(x)
    exp = jnp.unique_all(x.value)

    assert isinstance(got, _UniqueAllResult)

    assert isinstance(got.values, u.Q)
    assert jnp.array_equal(got.values, exp.values)

    assert isinstance(got.inverse, u.Q)
    assert jnp.array_equal(got.inverse, exp.inverse)

    assert isinstance(got.inverse_indices, Array)
    assert jnp.array_equal(got.inverse_indices, exp.inverse_indices)

    assert isinstance(got.counts, Array)
    assert jnp.array_equal(got.counts, exp.counts)


@pytest.mark.xfail(reason="value is not a Quantity")
def test_unique_counts():
    """Test `unique_counts`."""
    x = u.Q(jnp.asarray([1, 2, 2, 3, 3, 3], dtype=float))
    got = jnp.unique_counts(x)
    exp = jnp.unique_counts(x.value)

    assert isinstance(got, _UniqueCountsResult)

    assert isinstance(got.values, u.Q)
    assert jnp.array_equal(got.values.value, exp.values)

    assert isinstance(got.counts, Array)
    assert jnp.array_equal(got.counts, exp.counts)


@pytest.mark.xfail(reason="value is not a Quantity")
def test_unique_inverse():
    """Test `unique_inverse`."""
    x = u.Q(jnp.asarray([1, 2, 2, 3, 3, 3], dtype=float))
    got = jnp.unique_inverse(x)
    exp = jnp.unique_inverse(x.value)

    assert isinstance(got, _UniqueInverseResult)

    assert isinstance(got.values, u.Q)
    assert jnp.array_equal(got.values.value, exp.values)

    assert isinstance(got.inverse, u.Q)
    assert jnp.array_equal(got.inverse.value, exp.inverse)


@pytest.mark.xfail(reason="value is not a Quantity")
def test_unique_values():
    """Test `unique_values`."""
    x = u.Q(jnp.asarray([1, 2, 2, 3, 3, 3], dtype=float))
    got = jnp.unique_values(x)
    exp = u.Q(jnp.unique_values(x.value), "")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


# =============================================================================
# Sorting functions


def test_argsort():
    """Test `argsort`."""
    q = u.Q(jnp.asarray([3, 2, 1], dtype=float), "m")
    got = jnp.argsort(q)

    assert isinstance(got, Array)
    assert jnp.array_equal(got, jnp.argsort(q.value))


def test_sort():
    """Test `sort`."""
    q = u.Q(jnp.asarray([3, 2, 1], dtype=float), "m")
    got = jnp.sort(q)

    assert isinstance(got, u.Q)
    assert got.unit == u.unit("m")
    assert jnp.array_equal(got.value, jnp.sort(q.value))


# =============================================================================
# Statistical functions


def test_max():
    """Test `max`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "m")
    got = jnp.max(x)
    exp = u.Q(jnp.max(x.value), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_mean():
    """Test `mean`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "m")
    got = jnp.mean(x)
    exp = u.Q(jnp.mean(x.value), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_min():
    """Test `min`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "m")
    got = jnp.min(x)
    exp = u.Q(jnp.min(x.value), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


@pytest.mark.filterwarnings("ignore:Explicitly requested dtype")  # TODO: Why?
def test_prod():
    """Test `prod`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "m")
    got = jnp.prod(x)
    exp = u.Q(jnp.prod(x.value), "m3")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_std():
    """Test `std`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "m")
    got = jnp.std(x)
    exp = u.Q(jnp.std(x.value), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


@pytest.mark.filterwarnings("ignore:Explicitly requested dtype")  # TODO: Why?
def test_sum():
    """Test `sum`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "m")
    got = jnp.sum(x)
    exp = u.Q(jnp.sum(x.value), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_var():
    """Test `var`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=float), "m")
    got = jnp.var(x)
    exp = u.Q(jnp.var(x.value), "m2")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


# =============================================================================
# Utility functions


@pytest.mark.xfail(reason="returns a jax.Array")
def test_all():
    """Test `all`."""
    x = u.Q(jnp.asarray([True, False, True], dtype=bool), "")
    got = jnp.all(x)
    exp = u.Q(jnp.all(x.value), "")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_any():
    """Test `any`."""
    x = u.Q(jnp.asarray([1, 2, 3], dtype=bool), "m")
    got = jnp.any(x)
    exp = u.Q(jnp.any(x.value), "")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


# =============================================================================
# FFT


def test_fft_fft():
    """Test `fft.fft`."""
    x = u.Q(jnp.asarray([1.0, 2.0, 3.0, 4.0]), "m")
    got = jnp.fft.fft(x)
    # FFT converts spatial domain to frequency domain, inverting units
    exp = u.Q(jnp.fft.fft(x.value), "1/m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.allclose(got.value, exp.value)


def test_fft_ifft():
    """Test `fft.ifft`."""
    # Start with frequency domain (1/m), inverse FFT returns spatial (m)
    x = u.Q(jnp.asarray([1.0, 2.0, 3.0, 4.0]), "1/m")
    got = jnp.fft.ifft(x)
    exp = u.Q(jnp.fft.ifft(x.value), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.allclose(got.value, exp.value)


def test_fft_fftn():
    """Test `fft.fftn`."""
    x = u.Q(jnp.asarray([[1.0, 2.0], [3.0, 4.0]]), "m")
    got = jnp.fft.fftn(x)
    # N-D FFT inverts units in all dimensions
    exp = u.Q(jnp.fft.fftn(x.value), "1/m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.allclose(got.value, exp.value)


def test_fft_ifftn():
    """Test `fft.ifftn`."""
    # Start in frequency domain
    x = u.Q(jnp.asarray([[1.0, 2.0], [3.0, 4.0]]), "1/m")
    got = jnp.fft.ifftn(x)
    exp = u.Q(jnp.fft.ifftn(x.value), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.allclose(got.value, exp.value)


def test_fft_rfft():
    """Test `fft.rfft`."""
    x = u.Q(jnp.asarray([1.0, 2.0, 3.0, 4.0]), "m")
    got = jnp.fft.rfft(x)
    exp = u.Q(jnp.fft.rfft(x.value), "1/m")

    assert got.unit == exp.unit
    assert jnp.allclose(got.value, exp.value)


def test_fft_irfft():
    """Test `fft.irfft`."""
    # Start in frequency domain
    x = u.Q(jnp.asarray([1.0 + 0j, 2.0 + 1j, 3.0 + 0j]), "1/m")
    got = jnp.fft.irfft(x)
    exp = u.Q(jnp.fft.irfft(x.value), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.allclose(got.value, exp.value)


def test_fft_rfftn():
    """Test `fft.rfftn`."""
    x = u.Q(jnp.asarray([[1.0, 2.0], [3.0, 4.0]]), "m")
    got = jnp.fft.rfftn(x)
    exp = u.Q(jnp.fft.rfftn(x.value), "1/m")

    assert got.unit == exp.unit
    assert jnp.allclose(got.value, exp.value)


def test_fft_irfftn():
    """Test `fft.irfftn`."""
    # Start in frequency domain
    x = u.Q(jnp.asarray([[1.0 + 0j, 2.0 + 1j], [3.0 + 0j, 4.0 + 1j]]), "1/m")
    got = jnp.fft.irfftn(x)
    exp = u.Q(jnp.fft.irfftn(x.value), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.allclose(got.value, exp.value)


def test_fft_hfft():
    """Test `fft.hfft`."""
    # hfft transforms Hermitian symmetric frequency domain to real spatial
    x = u.Q(jnp.asarray([1.0 + 0j, 2.0 - 1j, 0.0 + 0j, 2.0 + 1j]), "1/m")
    got = jnp.fft.hfft(x)
    exp = u.Q(jnp.fft.hfft(x.value), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.allclose(got.value, exp.value)


def test_fft_ihfft():
    """Test `fft.ihfft`."""
    # ihfft transforms real spatial to Hermitian frequency domain
    x = u.Q(jnp.asarray([1.0, 2.0, 3.0, 4.0]), "m")
    got = jnp.fft.ihfft(x)
    exp = u.Q(jnp.fft.ihfft(x.value), "1/m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.allclose(got.value, exp.value)


def test_fft_fftfreq():
    """Test `fft.fftfreq`."""
    # fftfreq returns frequencies, not quantities
    n = 10
    d = u.Q(0.1, "s")
    got = jnp.fft.fftfreq(n, d)
    exp = u.Q(jnp.fft.fftfreq(n, d.value), "1/s")

    assert isinstance(got, u.Q)
    assert got.unit.is_equivalent(exp.unit)
    assert jnp.allclose(got.value, exp.value)


def test_fft_rfftfreq():
    """Test `fft.rfftfreq`."""
    # rfftfreq returns frequencies, not quantities
    n = 10
    d = u.Q(0.1, "s")
    got = jnp.fft.rfftfreq(n, d)
    exp = u.Q(jnp.fft.rfftfreq(n, d.value), "1/s")

    assert isinstance(got, u.Q)
    assert got.unit.is_equivalent(exp.unit)
    assert jnp.allclose(got.value, exp.value)


def test_fft_fftshift():
    """Test `fft.fftshift`."""
    # fftshift just reorders, doesn't change units
    x = u.Q(jnp.asarray([0, 1, 2, 3, 4, -5, -4, -3, -2, -1]), "1/m")
    got = jnp.fft.fftshift(x)
    exp = u.Q(jnp.fft.fftshift(x.value), "1/m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_fft_ifftshift():
    """Test `fft.ifftshift`."""
    # ifftshift just reorders, doesn't change units
    x = u.Q(jnp.asarray([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]), "1/m")
    got = jnp.fft.ifftshift(x)
    exp = u.Q(jnp.fft.ifftshift(x.value), "1/m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


# =============================================================================
# Linalg


def test_linalg_cholesky():
    """Test `linalg.cholesky`."""
    # Cholesky decomposition of a positive definite matrix
    x = u.Q(jnp.asarray([[4.0, 2.0], [2.0, 3.0]]), "m2")
    got = jnp.linalg.cholesky(x)
    exp = u.Q(jnp.linalg.cholesky(x.value), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.allclose(got.value, exp.value)


def test_linalg_cross():
    """Test `linalg.cross`."""
    x = u.Q(jnp.asarray([1.0, 2.0, 3.0]), "m")
    y = u.Q(jnp.asarray([4.0, 5.0, 6.0]), "m")
    got = jnp.linalg.cross(x, y)
    exp = u.Q(jnp.linalg.cross(x.value, y.value), "m2")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.allclose(got.value, exp.value)


def test_linalg_det():
    """Test `linalg.det`."""
    # For a 2x2 matrix with unit m, determinant has unit m^2
    x = u.Q(jnp.asarray([[1.0, 2.0], [3.0, 4.0]]), "m")
    got = jnp.linalg.det(x)
    exp = u.Q(jnp.linalg.det(x.value), "m2")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.allclose(got.value, exp.value)


def test_linalg_diagonal():
    """Test `linalg.diagonal`."""
    x = u.Q(jnp.asarray([[1.0, 2.0], [3.0, 4.0]]), "m")
    got = jnp.linalg.diagonal(x)
    exp = u.Q(jnp.linalg.diagonal(x.value), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


@pytest.mark.xfail(reason="complex return types need special handling")
def test_linalg_eigh():
    """Test `linalg.eigh`."""
    x = u.Q(jnp.asarray([[1.0, 0.5], [0.5, 2.0]]), "")
    eigenvalues, eigenvectors = jnp.linalg.eigh(x)
    exp_vals, exp_vecs = jnp.linalg.eigh(x.value)

    assert isinstance(eigenvalues, u.Q)
    assert isinstance(eigenvectors, u.Q)
    assert jnp.allclose(eigenvalues.value, exp_vals)
    assert jnp.allclose(eigenvectors.value, exp_vecs)


def test_linalg_eigvalsh():
    """Test `linalg.eigvalsh`."""
    x = u.Q(jnp.asarray([[1.0, 0.5], [0.5, 2.0]]), "")
    got = jnp.linalg.eigvalsh(x)
    exp = u.Q(jnp.linalg.eigvalsh(x.value), "")

    assert isinstance(got, u.Q)
    assert jnp.allclose(got.value, exp.value)


@pytest.mark.xfail(reason="TODO: fix")
def test_linalg_inv():
    """Test `linalg.inv`."""
    # Inverse of matrix with unit m has unit 1/m
    x = u.Q(jnp.asarray([[1.0, 2.0], [3.0, 5.0]]), "m")
    got = jnp.linalg.inv(x)
    exp = u.Q(jnp.linalg.inv(x.value), "1/m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.allclose(got.value, exp.value)


def test_linalg_matmul():
    """Test `linalg.matmul`."""
    x = u.Q(jnp.asarray([[1.0, 2.0], [3.0, 4.0]]), "m")
    y = u.Q(jnp.asarray([[5.0, 6.0], [7.0, 8.0]]), "s")
    got = jnp.linalg.matmul(x, y)
    exp = u.Q(jnp.linalg.matmul(x.value, y.value), "m*s")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.allclose(got.value, exp.value)


def test_linalg_matrix_norm():
    """Test `linalg.matrix_norm`."""
    x = u.Q(jnp.asarray([[1.0, 2.0], [3.0, 4.0]]), "m")
    got = jnp.linalg.matrix_norm(x)
    exp = u.Q(jnp.linalg.matrix_norm(x.value), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.allclose(got.value, exp.value)


def test_linalg_matrix_power():
    """Test `linalg.matrix_power`."""
    x = u.Q(jnp.asarray([[1.0, 2.0], [3.0, 4.0]]), "m")
    n = 2
    got = jnp.linalg.matrix_power(x, n)
    exp = u.Q(jnp.linalg.matrix_power(x.value, n), "m2")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.allclose(got.value, exp.value)


def test_linalg_matrix_rank():
    """Test `linalg.matrix_rank`."""
    x = u.Q(jnp.asarray([[1.0, 2.0], [3.0, 4.0]]), "m")
    # Rank is dimensionless
    got = jnp.linalg.matrix_rank(x)
    exp = jnp.linalg.matrix_rank(x.value)

    # Rank returns a scalar integer
    assert got == exp


def test_linalg_matrix_transpose():
    """Test `linalg.matrix_transpose`."""
    x = u.Q(jnp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), "m")
    got = jnp.linalg.matrix_transpose(x)
    exp = u.Q(jnp.linalg.matrix_transpose(x.value), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_linalg_outer():
    """Test `linalg.outer`."""
    x = u.Q(jnp.asarray([1.0, 2.0, 3.0]), "m")
    y = u.Q(jnp.asarray([4.0, 5.0]), "s")
    got = jnp.linalg.outer(x, y)
    exp = u.Q(jnp.linalg.outer(x.value, y.value), "m*s")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.array_equal(got.value, exp.value)


def test_linalg_pinv():
    """Test `linalg.pinv`."""
    x = u.Q(jnp.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), "m")
    got = jnp.linalg.pinv(x)
    exp = u.Q(jnp.linalg.pinv(x.value), "1/m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.allclose(got.value, exp.value)


def test_linalg_qr():
    """Test `linalg.qr`."""
    x = u.Q(jnp.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), "m")
    q, r = jnp.linalg.qr(x)
    q_exp, r_exp = jnp.linalg.qr(x.value)

    # Q is orthonormal (dimensionless), R has same units as input
    assert isinstance(q, u.Q)
    assert isinstance(r, u.Q)
    assert q.unit == u.unit("")
    assert r.unit == u.unit("m")
    assert jnp.allclose(q.value, q_exp)
    assert jnp.allclose(r.value, r_exp)


@pytest.mark.skip(
    reason="slogdet requires log of dimensional quantities, which is non-physical"
)
def test_linalg_slogdet():
    """Test `linalg.slogdet`."""
    # x = u.Q(jnp.asarray([[1.0, 2.0], [3.0, 4.0]]), "m")
    # sign, logdet = jnp.linalg.slogdet(x)
    # sign_exp, logdet_exp = jnp.linalg.slogdet(x.value)

    # # Sign is dimensionless, logdet contains log of units
    # assert isinstance(sign, jax.Array)
    # assert isinstance(logdet, u.Q)
    # assert jnp.allclose(sign, sign_exp)
    # assert jnp.allclose(logdet.value, logdet_exp)


def test_linalg_solve():
    """Test `linalg.solve`."""
    # Solve Ax = b where A has unit m and b has unit m*s, x has unit s
    a = u.Q(jnp.asarray([[1.0, 2.0], [3.0, 5.0]]), "m")
    b = u.Q(jnp.asarray([1.0, 2.0]), "m*s")
    got = jnp.linalg.solve(a, b)
    exp = u.Q(jnp.linalg.solve(a.value, b.value), "s")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.allclose(got.value, exp.value)


def test_linalg_svd():
    """Test `linalg.svd`."""
    x = u.Q(jnp.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), "m")
    u_mat, s, vt = jnp.linalg.svd(x, full_matrices=False)
    u_exp, s_exp, vt_exp = jnp.linalg.svd(x.value, full_matrices=False)

    # U and V^T are orthonormal (dimensionless), S has same units as input
    assert isinstance(u_mat, jax.Array)
    assert isinstance(s, u.Q)
    assert isinstance(vt, jax.Array)
    assert jnp.allclose(u_mat, u_exp)
    assert jnp.allclose(s.value, s_exp)
    assert jnp.allclose(vt, vt_exp)


def test_linalg_svdvals():
    """Test `linalg.svdvals`."""
    x = u.Q(jnp.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), "m")
    got = jnp.linalg.svdvals(x)
    exp = u.Q(jnp.linalg.svdvals(x.value), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.allclose(got.value, exp.value)


def test_linalg_tensordot():
    """Test `linalg.tensordot`."""
    x = u.Q(jnp.asarray([[1.0, 2.0], [3.0, 4.0]]), "m")
    y = u.Q(jnp.asarray([[5.0, 6.0], [7.0, 8.0]]), "s")
    got = jnp.linalg.tensordot(x, y, axes=1)
    exp = u.Q(jnp.linalg.tensordot(x.value, y.value, axes=1), "m*s")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.allclose(got.value, exp.value)


def test_linalg_trace():
    """Test `linalg.trace`."""
    x = u.Q(jnp.asarray([[1.0, 2.0], [3.0, 4.0]]), "m")
    got = jnp.linalg.trace(x)
    exp = u.Q(jnp.linalg.trace(x.value), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.allclose(got.value, exp.value)


def test_linalg_vecdot():
    """Test `linalg.vecdot`."""
    x = u.Q(jnp.asarray([1.0, 2.0, 3.0]), "m")
    y = u.Q(jnp.asarray([4.0, 5.0, 6.0]), "s")
    got = jnp.linalg.vecdot(x, y)
    exp = u.Q(jnp.linalg.vecdot(x.value, y.value), "m*s")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.allclose(got.value, exp.value)


def test_linalg_vector_norm():
    """Test `linalg.vector_norm`."""
    x = u.Q(jnp.asarray([3.0, 4.0]), "m")
    got = jnp.linalg.vector_norm(x)
    exp = u.Q(jnp.linalg.vector_norm(x.value), "m")

    assert isinstance(got, u.Q)
    assert got.unit == exp.unit
    assert jnp.allclose(got.value, exp.value)


##############################################################################
# Functions


def test_allclose():
    """Test `e`."""
    q = u.Q(100.0, "m")

    with pytest.raises(Exception):  # noqa: B017, PT011
        assert jnp.allclose(q, u.Q(0.1, "km"))

    # Need the `atol` argument.
    assert jnp.allclose(q, u.Q(0.1, "km"), atol=u.Q(1e-6, "m"))
