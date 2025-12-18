"""Test the Array API."""
# pylint: disable=import-error, too-many-lines

import re

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
    assert not isinstance(jnp.e, u.Quantity)


def test_inf():
    """Test `inf`."""
    assert not isinstance(jnp.inf, u.Quantity)


def test_nan():
    """Test `nan`."""
    assert not isinstance(jnp.nan, u.Quantity)


def test_newaxis():
    """Test `newaxis`."""
    assert not isinstance(jnp.newaxis, u.Quantity)


def test_pi():
    """Test `pi`."""
    assert not isinstance(jnp.pi, u.Quantity)


# =============================================================================
# Creation functions

# -----------------------------------------------
# arange


def test_arange_start():
    """Test `arange`."""
    # -----------------------
    got = jnp.arange(u.Quantity(10, "m"))
    expected = u.Quantity(jnp.arange(10), "m")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)

    # -----------------------
    got = jnp.arange(start=u.Quantity(10, "m"))

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_arange_stop():
    """Test `arange`."""
    start = u.Quantity(2, "m")
    stop = u.Quantity(10, "km")
    got = jnp.arange(start, stop)
    expected = u.Quantity(jnp.arange(start.value, u.ustrip(start.unit, stop)), "m")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_arange_step():
    """Test `arange`."""
    start = u.Quantity(2, "m")
    stop = u.Quantity(10, "km")
    step = u.Quantity(2, "m")
    got = jnp.arange(start, stop, step)
    expected = u.Quantity(
        jnp.arange(start.value, u.ustrip(start.unit, stop), u.ustrip(start.unit, step)),
        "m",
    )

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


# -----------------------------------------------


def test_asarray():
    """Test `asarray`."""
    # TODO: test the dtype, device, copy arguments
    x = [1, 2, 3]
    got = jnp.asarray(u.Quantity(x, "m"))
    expected = u.Quantity(jnp.asarray(x), "m")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


# -----------------------------------------------


@pytest.mark.xfail(reason="returns a jax.Array")
def test_empty():
    """Test `empty`."""
    # TODO: test the dtype, device arguments
    got = jnp.empty((2, 3))
    assert isinstance(got, u.Quantity)


# -----------------------------------------------


def test_empty_like():
    """Test `empty_like`."""
    x = jnp.asarray([1, 2, 3], dtype=float)
    q = u.Quantity(x, "m")
    got = jnp.empty_like(q)
    expected = u.Quantity(jnp.empty_like(x), "m")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


# -----------------------------------------------


@pytest.mark.xfail(reason="returns a jax.Array")
def test_eye():
    """Test `eye`."""
    got = jnp.eye(3)

    assert isinstance(got, u.Quantity)


# -----------------------------------------------


@pytest.mark.skip("TODO")
def test_from_dlpack():
    """Test `from_dlpack`."""
    raise NotImplementedError


# -----------------------------------------------


@pytest.mark.xfail(reason="returns a jax.Array")
def test_full():
    """Test `full`."""
    got = jnp.full((2, 3), 1.0)

    assert isinstance(got, u.Quantity)


# -----------------------------------------------


def test_full_like_single_arg():
    """Test `full_like`."""
    x = jnp.asarray([1, 2, 3], dtype=float)
    q = u.Quantity(x, "m")
    got = jnp.full_like(q, fill_value=1.0)
    expected = u.Quantity(jnp.full_like(x, fill_value=1.0), "m")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_full_like_double_arg():
    """Test `full_like`."""
    x = jnp.asarray([1, 2, 3], dtype=float)
    q = u.Quantity(x, "m")
    got = jnp.full_like(q, 1.0)
    expected = u.Quantity(jnp.full_like(x, 1.0), "m")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_full_like_dtype():
    """Test `full_like`."""
    x = jnp.asarray([1, 2, 3], dtype=float)
    q = u.Quantity(x, "m")
    got = jnp.full_like(q, 1.0, dtype=int)
    expected = u.Quantity(jnp.full_like(q.value, 1.0, dtype=int), "m")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)
    assert got.value.dtype == expected.value.dtype


# -----------------------------------------------


def test_linspace():
    """Test `linspace`."""
    # TODO: test the dtype, device, endpoint arguments
    got = jnp.linspace(u.Quantity(0.0, "m"), u.Quantity(10.0, "m"), 11)
    expected = u.Quantity(jnp.linspace(0.0, 10.0, 11), "m")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


# -----------------------------------------------


def test_meshgrid():
    """Test `meshgrid`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "m")
    y = u.Quantity(jnp.asarray([4, 5, 6], dtype=float), "m")

    got1, got2 = jnp.meshgrid(x, y)
    exp1, exp2 = jnp.meshgrid(x.value, y.value)

    assert isinstance(got1, u.Quantity)
    assert jnp.array_equal(got1.value, exp1)

    assert isinstance(got2, u.Quantity)
    assert jnp.array_equal(got2.value, exp2)


# -----------------------------------------------


@pytest.mark.xfail(reason="returns a jax.Array")
def test_ones():
    """Test `ones`."""
    assert isinstance(jnp.ones((2, 3)), u.Quantity)


# -----------------------------------------------


def test_ones_like():
    """Test `ones_like`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "m")
    got = jnp.ones_like(x)
    expected = u.Quantity(jnp.ones_like(x.value), "m")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


# -----------------------------------------------


def test_tril():
    """Test `tril`."""
    x = u.Quantity([[1, 2, 3], [4, 5, 6], [7, 8, 9]], "m")
    got = jnp.tril(x)
    expected = u.Quantity(jnp.tril(x.value), "m")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


# -----------------------------------------------


def test_triu():
    """Test `triu`."""
    x = u.Quantity([[1, 2, 3], [4, 5, 6], [7, 8, 9]], "m")
    got = jnp.triu(x)
    expected = u.Quantity(jnp.triu(x.value), "m")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


# -----------------------------------------------


@pytest.mark.xfail(reason="returns a jax.Array")
def test_zeros():
    """Test `zeros`."""
    assert isinstance(jnp.zeros((2, 3)), u.Quantity)


# -----------------------------------------------


def test_zeros_like():
    """Test `zeros_like`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "m")
    got = jnp.zeros_like(x)
    expected = u.Quantity(jnp.zeros_like(x.value), "m")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


# =============================================================================
# Data-type functions


def test_astype():
    """Test `astype`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "m")
    got = jnp.astype(x, jnp.float32)
    expected = u.Quantity(jnp.asarray(x.value, dtype=jnp.float32), "m")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


@pytest.mark.skip("TODO")
def test_can_cast():
    """Test `can_cast`."""


def test_finfo():
    """Test `finfo`."""
    got = jnp.finfo(jnp.float32)
    expected = jnp.finfo(jnp.float32)

    assert isinstance(got, FInfo)
    for attr in ("bits", "eps", "max", "min", "smallest_normal", "dtype"):
        assert getattr(got, attr) == getattr(expected, attr)


def test_iinfo():
    """Test `iinfo`."""
    got = jnp.iinfo(jnp.int32)
    expected = jnp.iinfo(jnp.int32)

    assert isinstance(got, IInfo)
    for attr in ("kind", "bits", "min", "max", "dtype"):
        assert getattr(got, attr) == getattr(expected, attr)


def test_isdtype():
    """Test `isdtype`."""
    # True by definition


def test_result_type():
    """Test `result_type`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "m")
    y = u.Quantity(jnp.asarray([4, 5, 6], dtype=float), "m")
    got = jnp.result_type(x, y)
    expected = jnp.result_type(x.value, y.value)

    assert isinstance(got, jnp.dtype)
    assert got == expected


# =============================================================================
# Elementwise functions


def test_abs():
    """Test `abs`."""
    x = u.Quantity([-1, 2, -3], "m")
    got = jnp.abs(x)
    expected = u.Quantity(jnp.abs(x.value), "m")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_acos():
    """Test `acos`."""
    x = u.Quantity(jnp.asarray([-1, 0, 1], dtype=float), "")
    got = jnp.acos(x)
    expected = u.Quantity(jnp.acos(x.value), "rad")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_acosh():
    """Test `acosh`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "")
    got = jnp.acosh(x)
    expected = u.Quantity(jnp.acosh(x.value), "rad")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_add():
    """Test `add`."""
    # Adding two quantities
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "m")
    y = u.Quantity(jnp.asarray([4, 5, 6], dtype=float), "m")
    got = jnp.add(x, y)
    expected = u.Quantity(jnp.add(x.value, y.value), "m")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)

    # Adding a quantity and non-quantity
    match = re.escape("'m' (length) and '' (dimensionless) are not convertible")
    with pytest.raises(ValueError, match=match):
        jnp.add(x.value, y)

    with pytest.raises(ValueError, match=match):
        jnp.add(x, y.value)

    # Add a non-quantity and dimensionless quantity
    got = jnp.add(x.value, u.Quantity(1.0, ""))
    expected = u.Quantity(x.value + 1, "")
    assert jnp.array_equal(got.value, expected.value)

    got = jnp.add(u.Quantity(1.0, ""), y.value)
    expected = u.Quantity(1 + y.value, "")
    assert jnp.array_equal(got.value, expected.value)


def test_asin():
    """Test `asin`."""
    x = u.Quantity(jnp.asarray([-1, 0, 1], dtype=float), "")
    got = jnp.asin(x)
    expected = u.Quantity(jnp.asin(x.value), "rad")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_asinh():
    """Test `asinh`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "")
    got = jnp.asinh(x)
    expected = u.Quantity(jnp.asinh(x.value), "rad")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_atan():
    """Test `atan`."""
    x = u.Quantity(jnp.asarray([-1, 0, 1], dtype=float), "")
    got = jnp.atan(x)
    expected = u.Quantity(jnp.atan(x.value), "rad")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_atan2():
    """Test `atan2`."""
    x = u.Quantity(jnp.asarray([-1, 0, 1], dtype=float), "m")
    y = u.Quantity(jnp.asarray([4, 5, 6], dtype=float), "m")
    got = jnp.atan2(x, y)
    expected = u.Quantity(jnp.atan2(x.value, y.value), "rad")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_atanh():
    """Test `atanh`."""
    x = u.Quantity(jnp.asarray([-1, 0, 1], dtype=float), "")
    got = jnp.atanh(x)
    expected = u.Quantity(jnp.atanh(x.value), "rad")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_bitwise_and():
    """Test `bitwise_and`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=int), "")
    y = u.Quantity(jnp.asarray([4, 5, 6], dtype=int), "")
    got = jnp.bitwise_and(x, y)
    expected = jnp.bitwise_and(x.value, y.value)

    assert isinstance(got, Array)
    assert jnp.array_equal(got, expected)


def test_bitwise_left_shift():
    """Test `bitwise_left_shift`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=int), "")
    y = u.Quantity(jnp.asarray([4, 5, 6], dtype=int), "")
    got = jnp.bitwise_left_shift(x, y)
    expected = u.Quantity(jnp.bitwise_left_shift(x.value, y.value), "")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_bitwise_invert():
    """Test `bitwise_invert`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=int), "")
    got = jnp.bitwise_invert(x)
    expected = u.Quantity(jnp.bitwise_invert(x.value), "")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_bitwise_or():
    """Test `bitwise_or`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=int), "")
    y = u.Quantity(jnp.asarray([4, 5, 6], dtype=int), "")
    got = jnp.bitwise_or(x, y)
    expected = u.Quantity(jnp.bitwise_or(x.value, y.value), "")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_bitwise_right_shift():
    """Test `bitwise_right_shift`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=int), "")
    y = u.Quantity(jnp.asarray([4, 5, 6], dtype=int), "")
    got = jnp.bitwise_right_shift(x, y)
    expected = u.Quantity(jnp.bitwise_right_shift(x.value, y.value), "")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_bitwise_xor():
    """Test `bitwise_xor`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=int), "")
    y = u.Quantity(jnp.asarray([4, 5, 6], dtype=int), "")
    got = jnp.bitwise_xor(x, y)
    expected = u.Quantity(jnp.bitwise_xor(x.value, y.value), "")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_ceil():
    """Test `ceil`."""
    x = u.Quantity([1.1, 2.2, 3.3], "m")
    got = jnp.ceil(x)
    expected = u.Quantity(jnp.ceil(x.value), "m")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_conj():
    """Test `conj`."""
    x = u.Quantity([1 + 2j, 3 + 4j], "m")
    got = jnp.conj(x)
    expected = u.Quantity(jnp.conj(x.value), "m")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_cos():
    """Test `cos`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "rad")
    got = jnp.cos(x)
    expected = u.Quantity(jnp.cos(x.value), "")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_cosh():
    """Test `cosh`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "rad")
    got = jnp.cosh(x)
    expected = u.Quantity(jnp.cosh(x.value), "")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_divide():
    """Test `divide`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "m")
    y = u.Quantity(jnp.asarray([4, 5, 6], dtype=float), "km")
    got = jnp.divide(x, y)
    expected = u.Quantity(jnp.divide(x.value, y.value), "m / km")

    assert isinstance(got, u.Quantity)
    assert got.unit.is_equivalent(expected.unit)
    assert jnp.array_equal(got.value, expected.value)


def test_equal():
    """Test `equal`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "m")
    y = u.Quantity(jnp.asarray([4, 5, 6], dtype=float), "m")
    got = jnp.equal(x, y)
    expected = jnp.equal(x.value, y.value)

    assert isinstance(got, Array)
    assert jnp.array_equal(got, expected)


def test_exp():
    """Test `exp`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "")
    got = jnp.exp(x)
    expected = u.Quantity(jnp.exp(x.value), "")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_expm1():
    """Test `expm1`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "")
    got = jnp.expm1(x)
    expected = u.Quantity(jnp.expm1(x.value), "")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_floor():
    """Test `floor`."""
    x = u.Quantity([1.1, 2.2, 3.3], "m")
    got = jnp.floor(x)
    expected = u.Quantity(jnp.floor(x.value), "m")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_floor_divide():
    """Test `floor_divide`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "m")
    y = u.Quantity(jnp.asarray([4, 5, 6], dtype=float), "m")
    got = jnp.floor_divide(x, y)
    expected = u.Quantity(jnp.floor_divide(x.value, y.value), "m / m")

    assert isinstance(got, u.Quantity)
    assert got.unit.is_equivalent(expected.unit)
    assert jnp.array_equal(got.value, expected.value)


def test_greater():
    """Test `greater`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "m")
    y = u.Quantity(jnp.asarray([4, 5, 6], dtype=float), "m")
    got = jnp.greater(x, y)
    expected = jnp.greater(x.value, y.value)

    assert isinstance(got, Array)
    assert jnp.array_equal(got, expected)


def test_greater_equal():
    """Test `greater_equal`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "m")
    y = u.Quantity(jnp.asarray([4, 5, 6], dtype=float), "m")
    got = jnp.greater_equal(x, y)
    expected = jnp.greater_equal(x.value, y.value)

    assert isinstance(got, Array)
    assert jnp.array_equal(got, expected)


def test_imag():
    """Test `imag`."""
    x = u.Quantity([1 + 2j, 3 + 4j], "m")
    got = jnp.imag(x)
    expected = u.Quantity(jnp.imag(x.value), "m")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_isfinite():
    """Test `isfinite`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "m")
    got = jnp.isfinite(x)
    expected = jnp.isfinite(x.value)

    assert isinstance(got, Array)
    assert jnp.array_equal(got, expected)


def test_isinf():
    """Test `isinf`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "m")
    got = jnp.isinf(x)
    expected = jnp.isinf(x.value)

    assert isinstance(got, Array)
    assert jnp.array_equal(got, expected)


def test_isnan():
    """Test `isnan`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "m")
    got = jnp.isnan(x)
    expected = jnp.isnan(x.value)

    assert isinstance(got, Array)
    assert jnp.array_equal(got, expected)


def test_less():
    """Test `less`."""
    x = u.Quantity([1, 5, 3], "m")
    y = u.Quantity([4, 2, 6], "m")
    got = jnp.less(x, y)
    expected = jnp.less(x.value, y.value)

    assert isinstance(got, Array)
    assert jnp.array_equal(got, expected)


def test_less_equal():
    """Test `less_equal`."""
    x = u.Quantity([1, 5, 3], "m")
    y = u.Quantity([4, 2, 6], "m")
    got = jnp.less_equal(x, y)
    expected = jnp.less_equal(x.value, y.value)

    assert isinstance(got, Array)
    assert jnp.array_equal(got, expected)


def test_log():
    """Test `log`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "")
    got = jnp.log(x)
    expected = u.Quantity(jnp.log(x.value), "")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_log1p():
    """Test `log1p`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "")
    got = jnp.log1p(x)
    expected = u.Quantity(jnp.log1p(x.value), "")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_log2():
    """Test `log2`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "")
    got = jnp.log2(x)
    expected = u.Quantity(jnp.log2(x.value), "")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_log10():
    """Test `log10`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "")
    got = jnp.log10(x)
    expected = u.Quantity(jnp.log10(x.value), "")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.allclose(got, expected, atol=u.Quantity(1e-8, ""))


def test_logaddexp():
    """Test `logaddexp`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "")
    y = u.Quantity(jnp.asarray([4, 5, 6], dtype=float), "")
    got = jnp.logaddexp(x, y)
    expected = u.Quantity(jnp.logaddexp(x.value, y.value), "")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_logical_and():
    """Test `logical_and`."""
    x = u.Quantity([True, False, True], "")
    y = u.Quantity([False, True, False], "")
    got = jnp.logical_and(x, y)
    expected = jnp.logical_and(x.value, y.value)

    assert isinstance(got, Array)
    assert jnp.array_equal(got, expected)


def test_logical_not():
    """Test `logical_not`."""
    x = u.Quantity([True, False, True], "")
    got = jnp.logical_not(x)
    expected = u.Quantity(jnp.logical_not(x.value), "")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_logical_or():
    """Test `logical_or`."""
    x = u.Quantity([True, False, True], "")
    y = u.Quantity([False, True, False], "")
    got = jnp.logical_or(x, y)
    expected = u.Quantity(jnp.logical_or(x.value, y.value), "")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_logical_xor():
    """Test `logical_xor`."""
    x = u.Quantity([True, False, True], "")
    y = u.Quantity([False, True, False], "")
    got = jnp.logical_xor(x, y)
    expected = u.Quantity(jnp.logical_xor(x.value, y.value), "")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_multiply():
    """Test `multiply`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "m")
    y = u.Quantity(jnp.asarray([4, 5, 6], dtype=float), "m")
    got = jnp.multiply(x, y)
    expected = u.Quantity(jnp.multiply(x.value, y.value), "m2")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_negative():
    """Test `negative`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "m")
    got = jnp.negative(x)
    expected = u.Quantity(jnp.negative(x.value), "m")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_not_equal():
    """Test `not_equal`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "m")
    y = u.Quantity(jnp.asarray([4, 2, 6], dtype=float), "m")
    got = jnp.not_equal(x, y)
    expected = jnp.not_equal(x.value, y.value)

    assert isinstance(got, Array)
    assert jnp.array_equal(got, expected)


def test_positive():
    """Test `positive`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "m")
    got = jnp.positive(x)
    expected = u.Quantity(jnp.positive(x.value), "m")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_pow_quantity_power():
    """Test `pow`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "m")
    y = u.Quantity(jnp.asarray([4], dtype=float), "")
    got = jnp.pow(x, y)
    expected = u.Quantity(jnp.pow(x.value, y.value), "m4")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_pow():
    """Test `pow`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "m")
    y = jnp.asarray([4], dtype=float)
    got = jnp.pow(x, y)
    expected = u.Quantity(jnp.pow(x.value, y), "m4")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_real():
    """Test `real`."""
    x = u.Quantity([1 + 2j, 3 + 4j], "m")
    got = jnp.real(x)
    expected = u.Quantity(jnp.real(x.value), "m")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


@pytest.mark.skip(reason="TODO")
def test_remainder():
    """Test `remainder`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "m")
    y = u.Quantity(jnp.asarray([4, 5, 6], dtype=float), "m")
    got = jnp.remainder(x, y)
    expected = u.Quantity(jnp.remainder(x.value, y.value), "m")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_round():
    """Test `round`."""
    x = u.Quantity([1.1, 2.2, 3.3], "m")
    got = jnp.round(x)
    expected = u.Quantity(jnp.round(x.value), "m")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_sign():
    """Test `sign`."""
    x = u.Quantity([-1, 2, -3], "m")
    got = jnp.sign(x)
    expected = jnp.sign(x.value)

    assert isinstance(got, Array)
    assert jnp.array_equal(got, expected)


def test_sin():
    """Test `sin`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "rad")
    got = jnp.sin(x)
    expected = u.Quantity(jnp.sin(x.value), "")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_sinh():
    """Test `sinh`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "rad")
    got = jnp.sinh(x)
    expected = u.Quantity(jnp.sinh(x.value), "")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_square():
    """Test `square`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "m")
    got = jnp.square(x)
    expected = u.Quantity(jnp.square(x.value), "m2")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_sqrt():
    """Test `sqrt`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "m")
    got = jnp.sqrt(x)
    expected = u.Quantity(jnp.sqrt(x.value), u.unit("m(1/2)"))

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_subtract():
    """Test `subtract`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "m")
    y = u.Quantity(jnp.asarray([4, 5, 6], dtype=float), "m")
    got = jnp.subtract(x, y)
    expected = u.Quantity(jnp.subtract(x.value, y.value), "m")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_tan():
    """Test `tan`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "rad")
    got = jnp.tan(x)
    expected = u.Quantity(jnp.tan(x.value), "")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_tanh():
    """Test `tanh`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "rad")
    got = jnp.tanh(x)
    expected = u.Quantity(jnp.tanh(x.value), "")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


@pytest.mark.skip(reason="TODO")
def test_trunc():
    """Test `trunc`."""
    x = u.Quantity([1.1, 2.2, 3.3], "m")
    got = jnp.trunc(x)
    expected = u.Quantity(jnp.trunc(x.value), "m")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


# =============================================================================
# Indexing functions


@pytest.mark.skip(reason="TODO")
def test_take():
    """Test `take`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "m")
    indices = u.Quantity(jnp.asarray([0, 1, 2], dtype=int), "")
    got = jnp.take(x, indices)
    expected = u.Quantity(jnp.take(x.value, indices.value, axis=None), "m")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


# =============================================================================
# Linear algebra functions


@pytest.mark.skip(reason="TODO")
def test_matmul():
    """Test `matmul`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "m")
    y = u.Quantity(jnp.asarray([4, 5, 6], dtype=float), "m")
    got = jnp.matmul(x, y)
    expected = u.Quantity(jnp.matmul(x.value, y.value), "m2")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_matrix_transpose():
    """Test `matrix_transpose`."""
    x = u.Quantity(jnp.asarray([[1, 2, 3], [4, 5, 6]], dtype=float), "m")
    got = jnp.matrix_transpose(x)
    expected = u.Quantity(jnp.matrix_transpose(x.value), "m")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


@pytest.mark.skip(reason="TODO")
def test_tensordot():
    """Test `tensordot`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "m")
    y = u.Quantity(jnp.asarray([4, 5, 6], dtype=float), "m")
    axes = 1
    got = jnp.tensordot(x, y, axes=axes)
    expected = u.Quantity(jnp.tensordot(x.value, y.value, axes=axes), "m2")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


@pytest.mark.skip(reason="TODO")
def test_vecdot():
    """Test `vecdot`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "m")
    y = u.Quantity(jnp.asarray([4, 5, 6], dtype=float), "m")
    got = jnp.vecdot(x, y)
    expected = u.Quantity(jnp.vecdot(x.value, y.value), "m2")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


# =============================================================================
# Manipulation functions


def test_broadcast_arrays():
    """Test `broadcast_arrays`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "m")
    y = u.Quantity(jnp.asarray([4], dtype=float), "s")
    got = jnp.broadcast_arrays(x, y)
    expected = jnp.broadcast_arrays(x.value, y.value)

    assert isinstance(got, tuple | list)
    assert len(got) == len(expected)
    for got_, expected_ in zip(got, expected, strict=True):
        assert isinstance(got_, u.Quantity)
        assert jnp.array_equal(got_.value, expected_)


def test_broadcast_to():
    """Test `broadcast_to`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "m")
    shape = (2, 3)
    got = jnp.broadcast_to(x, shape)
    expected = u.Quantity(jnp.broadcast_to(x.value, shape), "m")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_concat():
    """Test `concat`."""
    # TODO: test the axis argument
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "m")
    y = u.Quantity(jnp.asarray([4], dtype=float), "m")
    got = jnp.concat((x, y))
    expected = u.Quantity(jnp.concat((x.value, y.value)), "m")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_expand_dims():
    """Test `expand_dims`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "m")
    got = jnp.expand_dims(x, axis=0)
    expected = u.Quantity(jnp.expand_dims(x.value, axis=0), "m")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_flip():
    """Test `flip`."""
    x = u.Quantity(jnp.asarray([[1, 2, 3], [4, 5, 6]], dtype=float), "m")
    got = jnp.flip(x)
    expected = u.Quantity(jnp.flip(x.value), "m")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_permute_dims():
    """Test `permute_dims`."""
    x = u.Quantity(jnp.asarray([[1, 2, 3], [4, 5, 6]], dtype=float), "m")
    got = jnp.permute_dims(x, (1, 0))
    expected = u.Quantity(jnp.permute_dims(x.value, (1, 0)), "m")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_reshape():
    """Test `reshape`."""
    x = u.Quantity(jnp.asarray([[1, 2, 3], [4, 5, 6]], dtype=float), "m")
    got = jnp.reshape(x, (3, 2))
    expected = u.Quantity(jnp.reshape(x.value, (3, 2)), "m")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_roll():
    """Test `roll`."""
    x = u.Quantity(jnp.asarray([[1, 2, 3], [4, 5, 6]], dtype=float), "m")
    got = jnp.roll(x, shift=1, axis=0)
    expected = u.Quantity(jnp.roll(x.value, shift=1, axis=0), "m")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_squeeze():
    """Test `squeeze`."""
    x = u.Quantity(jnp.asarray([[[0], [1], [2]]], dtype=float), "m")
    got = jnp.squeeze(x, axis=(0, 2))
    expected = u.Quantity(jnp.squeeze(x.value, axis=(0, 2)), "m")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_stack():
    """Test `stack`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "m")
    y = u.Quantity(jnp.asarray([4, 5, 6], dtype=float), "m")
    got = jnp.stack((x, y))
    expected = u.Quantity(jnp.stack((x.value, y.value)), "m")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


# =============================================================================
# Searching functions


def test_argmax():
    """Test `argmax`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "m")
    got = jnp.argmax(x)
    exp = jnp.argmax(x.value)

    assert jnp.array_equal(got, exp)


def test_argmin():
    """Test `argmin`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "m")
    got = jnp.argmin(x)
    exp = jnp.argmin(x.value)
    assert jnp.array_equal(got, exp)


@pytest.mark.xfail(reason="TODO")
def test_nonzero():
    """Test `nonzero`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "m")
    (got,) = jnp.nonzero(x)
    (expected,) = u.Quantity(jnp.nonzero(x.value), "")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected)


def test_where():
    """Test `where`."""
    condition = u.Quantity(jnp.asarray([True, False, True]), "")
    y = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "m")
    z = u.Quantity(jnp.asarray([4, 5, 6], dtype=float), "m")
    got = jnp.where(condition, y, z)
    expected = u.Quantity(jnp.where(condition.value, y.value, z.value), "m")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


# =============================================================================
# Set functions


@pytest.mark.xfail(reason="value is not a Quantity")
def test_unique_all():
    """Test `unique_all`."""
    x = u.Quantity(jnp.asarray([1, 2, 2, 3, 3, 3], dtype=float))
    got = jnp.unique_all(x)
    expected = jnp.unique_all(x.value)

    assert isinstance(got, _UniqueAllResult)

    assert isinstance(got.values, u.Quantity)
    assert jnp.array_equal(got.values, expected.values)

    assert isinstance(got.inverse, u.Quantity)
    assert jnp.array_equal(got.inverse, expected.inverse)

    assert isinstance(got.inverse_indices, Array)
    assert jnp.array_equal(got.inverse_indices, expected.inverse_indices)

    assert isinstance(got.counts, Array)
    assert jnp.array_equal(got.counts, expected.counts)


@pytest.mark.xfail(reason="value is not a Quantity")
def test_unique_counts():
    """Test `unique_counts`."""
    x = u.Quantity(jnp.asarray([1, 2, 2, 3, 3, 3], dtype=float))
    got = jnp.unique_counts(x)
    expected = jnp.unique_counts(x.value)

    assert isinstance(got, _UniqueCountsResult)

    assert isinstance(got.values, u.Quantity)
    assert jnp.array_equal(got.values.value, expected.values)

    assert isinstance(got.counts, Array)
    assert jnp.array_equal(got.counts, expected.counts)


@pytest.mark.xfail(reason="value is not a Quantity")
def test_unique_inverse():
    """Test `unique_inverse`."""
    x = u.Quantity(jnp.asarray([1, 2, 2, 3, 3, 3], dtype=float))
    got = jnp.unique_inverse(x)
    expected = jnp.unique_inverse(x.value)

    assert isinstance(got, _UniqueInverseResult)

    assert isinstance(got.values, u.Quantity)
    assert jnp.array_equal(got.values.value, expected.values)

    assert isinstance(got.inverse, u.Quantity)
    assert jnp.array_equal(got.inverse.value, expected.inverse)


@pytest.mark.xfail(reason="value is not a Quantity")
def test_unique_values():
    """Test `unique_values`."""
    x = u.Quantity(jnp.asarray([1, 2, 2, 3, 3, 3], dtype=float))
    got = jnp.unique_values(x)
    expected = u.Quantity(jnp.unique_values(x.value), "")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


# =============================================================================
# Sorting functions


def test_argsort():
    """Test `argsort`."""
    q = u.Quantity(jnp.asarray([3, 2, 1], dtype=float), "m")
    got = jnp.argsort(q)

    assert isinstance(got, Array)
    assert jnp.array_equal(got, jnp.argsort(q.value))


def test_sort():
    """Test `sort`."""
    q = u.Quantity(jnp.asarray([3, 2, 1], dtype=float), "m")
    got = jnp.sort(q)

    assert isinstance(got, u.Quantity)
    assert got.unit == u.unit("m")
    assert jnp.array_equal(got.value, jnp.sort(q.value))


# =============================================================================
# Statistical functions


def test_max():
    """Test `max`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "m")
    got = jnp.max(x)
    expected = u.Quantity(jnp.max(x.value), "m")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


@pytest.mark.skip("TODO")
def test_mean():
    """Test `mean`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "m")
    got = jnp.mean(x)
    expected = u.Quantity(jnp.mean(x.value), "m")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


@pytest.mark.skip("TODO")
def test_min():
    """Test `min`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "m")
    got = jnp.min(x)
    expected = u.Quantity(jnp.min(x.value), "m")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


@pytest.mark.filterwarnings("ignore:Explicitly requested dtype")  # TODO: Why?
def test_prod():
    """Test `prod`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "m")
    got = jnp.prod(x)
    expected = u.Quantity(jnp.prod(x.value), "m3")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_std():
    """Test `std`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "m")
    got = jnp.std(x)
    expected = u.Quantity(jnp.std(x.value), "m")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


@pytest.mark.filterwarnings("ignore:Explicitly requested dtype")  # TODO: Why?
def test_sum():
    """Test `sum`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "m")
    got = jnp.sum(x)
    expected = u.Quantity(jnp.sum(x.value), "m")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_var():
    """Test `var`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=float), "m")
    got = jnp.var(x)
    expected = u.Quantity(jnp.var(x.value), "m2")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


# =============================================================================
# Utility functions


@pytest.mark.xfail(reason="returns a jax.Array")
def test_all():
    """Test `all`."""
    x = u.Quantity(jnp.asarray([True, False, True], dtype=bool), "")
    got = jnp.all(x)
    expected = u.Quantity(jnp.all(x.value), "")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_any():
    """Test `any`."""
    x = u.Quantity(jnp.asarray([1, 2, 3], dtype=bool), "m")
    got = jnp.any(x)
    expected = u.Quantity(jnp.any(x.value), "")

    assert isinstance(got, u.Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


# =============================================================================
# FFT


@pytest.mark.skip("TODO")
def test_fft_fft():
    """Test `fft.fft`."""


@pytest.mark.skip("TODO")
def test_fft_ifft():
    """Test `fft.ifft`."""


@pytest.mark.skip("TODO")
def test_fft_fftn():
    """Test `fft.fftn`."""


@pytest.mark.skip("TODO")
def test_fft_ifftn():
    """Test `fft.ifftn`."""


@pytest.mark.skip("TODO")
def test_fft_rfft():
    """Test `fft.rfft`."""


@pytest.mark.skip("TODO")
def test_fft_irfft():
    """Test `fft.irfft`."""


@pytest.mark.skip("TODO")
def test_fft_rfftn():
    """Test `fft.rfftn`."""


@pytest.mark.skip("TODO")
def test_fft_irfftn():
    """Test `fft.irfftn`."""


@pytest.mark.skip("TODO")
def test_fft_hfft():
    """Test `fft.hfft`."""


@pytest.mark.skip("TODO")
def test_fft_ihfft():
    """Test `fft.ihfft`."""


@pytest.mark.skip("TODO")
def test_fft_fftfreq():
    """Test `fft.fftfreq`."""


@pytest.mark.skip("TODO")
def test_fft_rfftfreq():
    """Test `fft.rfftfreq`."""


@pytest.mark.skip("TODO")
def test_fft_fftshift():
    """Test `fft.fftshift`."""


@pytest.mark.skip("TODO")
def test_fft_ifftshift():
    """Test `fft.ifftshift`."""


# =============================================================================
# Linalg


@pytest.mark.skip("TODO")
def test_linalg_cholesky():
    """Test `linalg.cholesky`."""


@pytest.mark.skip("TODO")
def test_linalg_cross():
    """Test `linalg.cross`."""


@pytest.mark.skip("TODO")
def test_linalg_det():
    """Test `linalg.det`."""


@pytest.mark.skip("TODO")
def test_linalg_diagonal():
    """Test `linalg.diagonal`."""


@pytest.mark.skip("TODO")
def test_linalg_eigh():
    """Test `linalg.eigh`."""


@pytest.mark.skip("TODO")
def test_linalg_eigvalsh():
    """Test `linalg.eigvalsh`."""


@pytest.mark.skip("TODO")
def test_linalg_inv():
    """Test `linalg.inv`."""


@pytest.mark.skip("TODO")
def test_linalg_matmul():
    """Test `linalg.matmul`."""


@pytest.mark.skip("TODO")
def test_linalg_matrix_norm():
    """Test `linalg.matrix_norm`."""


@pytest.mark.skip("TODO")
def test_linalg_matrix_power():
    """Test `linalg.matrix_power`."""


@pytest.mark.skip("TODO")
def test_linalg_matrix_rank():
    """Test `linalg.matrix_rank`."""


@pytest.mark.skip("TODO")
def test_linalg_matrix_transpose():
    """Test `linalg.matrix_transpose`."""


@pytest.mark.skip("TODO")
def test_linalg_outer():
    """Test `linalg.outer`."""


@pytest.mark.skip("TODO")
def test_linalg_pinv():
    """Test `linalg.pinv`."""


@pytest.mark.skip("TODO")
def test_linalg_qr():
    """Test `linalg.qr`."""


@pytest.mark.skip("TODO")
def test_linalg_slogdet():
    """Test `linalg.slogdet`."""


@pytest.mark.skip("TODO")
def test_linalg_solve():
    """Test `linalg.solve`."""


@pytest.mark.skip("TODO")
def test_linalg_svd():
    """Test `linalg.svd`."""


@pytest.mark.skip("TODO")
def test_linalg_svdvals():
    """Test `linalg.svdvals`."""


@pytest.mark.skip("TODO")
def test_linalg_tensordot():
    """Test `linalg.tensordot`."""


@pytest.mark.skip("TODO")
def test_linalg_trace():
    """Test `linalg.trace`."""


@pytest.mark.skip("TODO")
def test_linalg_vecdot():
    """Test `linalg.vecdot`."""


@pytest.mark.skip("TODO")
def test_linalg_vector_norm():
    """Test `linalg.vector_norm`."""


##############################################################################
# Functions


def test_allclose():
    """Test `e`."""
    q = u.Quantity(100.0, "m")

    with pytest.raises(Exception):  # noqa: B017, PT011
        assert jnp.allclose(q, u.Quantity(0.1, "km"))

    # Need the `atol` argument.
    assert jnp.allclose(q, u.Quantity(0.1, "km"), atol=u.Quantity(1e-6, "m"))
