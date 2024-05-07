# pylint: disable=import-error, too-many-lines
# ruff:noqa: E402

"""Test the Array API."""

import re

import astropy.units as u
import jax.numpy as jnp
import pytest
from jax import Array
from jax._src.numpy.setops import (
    _UniqueAllResult,
    _UniqueCountsResult,
    _UniqueInverseResult,
)

import quaxed.array_api as xp
import quaxed.numpy as qnp
from quaxed.array_api._data_type_functions import FInfo, IInfo

from unxt import Quantity

# =============================================================================
# Constants


def test_e():
    """Test `e`."""
    assert not isinstance(xp.e, Quantity)


def test_inf():
    """Test `inf`."""
    assert not isinstance(xp.inf, Quantity)


def test_nan():
    """Test `nan`."""
    assert not isinstance(xp.nan, Quantity)


def test_newaxis():
    """Test `newaxis`."""
    assert not isinstance(xp.newaxis, Quantity)


def test_pi():
    """Test `pi`."""
    assert not isinstance(xp.pi, Quantity)


# =============================================================================
# Creation functions

# -----------------------------------------------
# arange


def test_arange_start():
    """Test `arange`."""
    # -----------------------
    got = xp.arange(Quantity(10, u.m))
    expected = Quantity(xp.arange(10), u.m)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)

    # -----------------------
    got = xp.arange(start=Quantity(10, u.m))

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_arange_stop():
    """Test `arange`."""
    start = Quantity(2, u.m)
    stop = Quantity(10, u.km)
    got = xp.arange(start, stop)
    expected = Quantity(xp.arange(start.value, stop.to_units_value(start.unit)), u.m)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_arange_step():
    """Test `arange`."""
    start = Quantity(2, u.m)
    stop = Quantity(10, u.km)
    step = Quantity(2, u.m)
    got = xp.arange(start, stop, step)
    expected = Quantity(
        xp.arange(
            start.value,
            stop.to_units_value(start.unit),
            step.to_units_value(start.unit),
        ),
        u.m,
    )

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


# -----------------------------------------------


def test_asarray():
    """Test `asarray`."""
    # TODO: test the dtype, device, copy arguments
    x = [1, 2, 3]
    got = xp.asarray(Quantity(x, u.m))
    expected = Quantity(xp.asarray(x), u.m)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


# -----------------------------------------------


@pytest.mark.xfail(reason="returns a jax.Array")
def test_empty():
    """Test `empty`."""
    # TODO: test the dtype, device arguments
    got = xp.empty((2, 3))
    assert isinstance(got, Quantity)


# -----------------------------------------------


def test_empty_like():
    """Test `empty_like`."""
    x = xp.asarray([1, 2, 3], dtype=float)
    q = Quantity(x, u.m)
    got = xp.empty_like(q)
    expected = Quantity(xp.empty_like(x), u.m)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


# -----------------------------------------------


@pytest.mark.xfail(reason="returns a jax.Array")
def test_eye():
    """Test `eye`."""
    got = xp.eye(3)

    assert isinstance(got, Quantity)


# -----------------------------------------------


@pytest.mark.skip("TODO")
def test_from_dlpack():
    """Test `from_dlpack`."""
    raise NotImplementedError


# -----------------------------------------------


@pytest.mark.xfail(reason="returns a jax.Array")
def test_full():
    """Test `full`."""
    got = xp.full((2, 3), 1.0)

    assert isinstance(got, Quantity)


# -----------------------------------------------


def test_full_like_single_arg():
    """Test `full_like`."""
    x = xp.asarray([1, 2, 3], dtype=float)
    q = Quantity(x, u.m)
    got = xp.full_like(q, fill_value=1.0)
    expected = Quantity(xp.full_like(x, fill_value=1.0), u.m)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_full_like_double_arg():
    """Test `full_like`."""
    x = xp.asarray([1, 2, 3], dtype=float)
    q = Quantity(x, u.m)
    got = xp.full_like(q, 1.0)
    expected = Quantity(xp.full_like(x, 1.0), u.m)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_full_like_dtype():
    """Test `full_like`."""
    x = xp.asarray([1, 2, 3], dtype=float)
    q = Quantity(x, u.m)
    got = xp.full_like(q, 1.0, dtype=int)
    expected = Quantity(xp.full_like(q.value, 1.0, dtype=int), u.m)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)
    assert got.value.dtype == expected.value.dtype


# -----------------------------------------------


def test_linspace():
    """Test `linspace`."""
    # TODO: test the dtype, device, endpoint arguments
    got = xp.linspace(Quantity(0.0, u.m), Quantity(10.0, u.m), 11)
    expected = Quantity(xp.linspace(0.0, 10.0, 11), u.m)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


# -----------------------------------------------


def test_meshgrid():
    """Test `meshgrid`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.m)
    y = Quantity(xp.asarray([4, 5, 6], dtype=float), u.m)

    got1, got2 = xp.meshgrid(x, y)
    exp1, exp2 = xp.meshgrid(x.value, y.value)

    assert isinstance(got1, Quantity)
    assert jnp.array_equal(got1.value, exp1)

    assert isinstance(got2, Quantity)
    assert jnp.array_equal(got2.value, exp2)


# -----------------------------------------------


@pytest.mark.xfail(reason="returns a jax.Array")
def test_ones():
    """Test `ones`."""
    assert isinstance(xp.ones((2, 3)), Quantity)


# -----------------------------------------------


def test_ones_like():
    """Test `ones_like`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.m)
    got = xp.ones_like(x)
    expected = Quantity(xp.ones_like(x.value), u.m)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


# -----------------------------------------------


def test_tril():
    """Test `tril`."""
    x = Quantity([[1, 2, 3], [4, 5, 6], [7, 8, 9]], u.m)
    got = xp.tril(x)
    expected = Quantity(xp.tril(x.value), u.m)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


# -----------------------------------------------


def test_triu():
    """Test `triu`."""
    x = Quantity([[1, 2, 3], [4, 5, 6], [7, 8, 9]], u.m)
    got = xp.triu(x)
    expected = Quantity(xp.triu(x.value), u.m)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


# -----------------------------------------------


@pytest.mark.xfail(reason="returns a jax.Array")
def test_zeros():
    """Test `zeros`."""
    assert isinstance(xp.zeros((2, 3)), Quantity)


# -----------------------------------------------


def test_zeros_like():
    """Test `zeros_like`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.m)
    got = xp.zeros_like(x)
    expected = Quantity(xp.zeros_like(x.value), u.m)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


# =============================================================================
# Data-type functions


def test_astype():
    """Test `astype`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.m)
    got = xp.astype(x, jnp.float32)
    expected = Quantity(xp.asarray(x.value, dtype=jnp.float32), u.m)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


@pytest.mark.skip("TODO")
def test_can_cast():
    """Test `can_cast`."""


def test_finfo():
    """Test `finfo`."""
    got = xp.finfo(jnp.float32)
    expected = xp.finfo(jnp.float32)

    assert isinstance(got, FInfo)
    for attr in FInfo.__slots__:
        assert getattr(got, attr) == getattr(expected, attr)


def test_iinfo():
    """Test `iinfo`."""
    got = xp.iinfo(jnp.int32)
    expected = xp.iinfo(jnp.int32)

    assert isinstance(got, IInfo)
    for attr in ("kind", "bits", "min", "max", "dtype"):
        assert getattr(got, attr) == getattr(expected, attr)


def test_isdtype():
    """Test `isdtype`."""
    # True by definition


def test_result_type():
    """Test `result_type`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.m)
    y = Quantity(xp.asarray([4, 5, 6], dtype=float), u.m)
    got = xp.result_type(x, y)
    expected = xp.result_type(x.value, y.value)

    assert isinstance(got, jnp.dtype)
    assert got == expected


# =============================================================================
# Elementwise functions


def test_abs():
    """Test `abs`."""
    x = Quantity([-1, 2, -3], u.m)
    got = xp.abs(x)
    expected = Quantity(xp.abs(x.value), u.m)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_acos():
    """Test `acos`."""
    x = Quantity(xp.asarray([-1, 0, 1], dtype=float), u.one)
    got = xp.acos(x)
    expected = Quantity(xp.acos(x.value), u.rad)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_acosh():
    """Test `acosh`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.one)
    got = xp.acosh(x)
    expected = Quantity(xp.acosh(x.value), u.rad)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_add():
    """Test `add`."""
    # Adding two quantities
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.m)
    y = Quantity(xp.asarray([4, 5, 6], dtype=float), u.m)
    got = xp.add(x, y)
    expected = Quantity(xp.add(x.value, y.value), u.m)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)

    # Adding a quantity and non-quantity
    with pytest.raises(
        Exception, match=re.escape("Cannot add a non-quantity and quantity.")
    ):
        xp.add(x.value, y)

    with pytest.raises(
        Exception, match=re.escape("Cannot add a quantity and a non-quantity.")
    ):
        xp.add(x, y.value)

    # Add a non-quantity and dimensionless quantity
    got = xp.add(x.value, Quantity(1.0, u.one))
    expected = Quantity(x.value + 1, u.one)
    assert jnp.array_equal(got.value, expected.value)

    got = xp.add(Quantity(1.0, u.one), y.value)
    expected = Quantity(1 + y.value, u.one)
    assert jnp.array_equal(got.value, expected.value)


def test_asin():
    """Test `asin`."""
    x = Quantity(xp.asarray([-1, 0, 1], dtype=float), u.one)
    got = xp.asin(x)
    expected = Quantity(xp.asin(x.value), u.rad)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_asinh():
    """Test `asinh`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.one)
    got = xp.asinh(x)
    expected = Quantity(xp.asinh(x.value), u.rad)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_atan():
    """Test `atan`."""
    x = Quantity(xp.asarray([-1, 0, 1], dtype=float), u.one)
    got = xp.atan(x)
    expected = Quantity(xp.atan(x.value), u.rad)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_atan2():
    """Test `atan2`."""
    x = Quantity(xp.asarray([-1, 0, 1], dtype=float), u.m)
    y = Quantity(xp.asarray([4, 5, 6], dtype=float), u.m)
    got = xp.atan2(x, y)
    expected = Quantity(xp.atan2(x.value, y.value), u.rad)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_atanh():
    """Test `atanh`."""
    x = Quantity(xp.asarray([-1, 0, 1], dtype=float), u.one)
    got = xp.atanh(x)
    expected = Quantity(xp.atanh(x.value), u.rad)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_bitwise_and():
    """Test `bitwise_and`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=int), u.one)
    y = Quantity(xp.asarray([4, 5, 6], dtype=int), u.one)
    got = xp.bitwise_and(x, y)
    expected = xp.bitwise_and(x.value, y.value)

    assert isinstance(got, Array)
    assert jnp.array_equal(got, expected)


@pytest.mark.xfail(reason="TODO")
def test_bitwise_left_shift():
    """Test `bitwise_left_shift`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=int), u.one)
    y = Quantity(xp.asarray([4, 5, 6], dtype=int), u.one)
    got = xp.bitwise_left_shift(x, y)
    expected = Quantity(xp.bitwise_left_shift(x.value, y.value), u.one)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


@pytest.mark.xfail(reason="TODO")
def test_bitwise_invert():
    """Test `bitwise_invert`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=int), u.one)
    got = xp.bitwise_invert(x)
    expected = Quantity(xp.bitwise_invert(x.value), u.one)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


@pytest.mark.xfail(reason="TODO")
def test_bitwise_or():
    """Test `bitwise_or`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=int), u.one)
    y = Quantity(xp.asarray([4, 5, 6], dtype=int), u.one)
    got = xp.bitwise_or(x, y)
    expected = Quantity(xp.bitwise_or(x.value, y.value), u.one)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


@pytest.mark.xfail(reason="TODO")
def test_bitwise_right_shift():
    """Test `bitwise_right_shift`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=int), u.one)
    y = Quantity(xp.asarray([4, 5, 6], dtype=int), u.one)
    got = xp.bitwise_right_shift(x, y)
    expected = Quantity(xp.bitwise_right_shift(x.value, y.value), u.one)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


@pytest.mark.xfail(reason="TODO")
def test_bitwise_xor():
    """Test `bitwise_xor`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=int), u.one)
    y = Quantity(xp.asarray([4, 5, 6], dtype=int), u.one)
    got = xp.bitwise_xor(x, y)
    expected = Quantity(xp.bitwise_xor(x.value, y.value), u.one)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_ceil():
    """Test `ceil`."""
    x = Quantity([1.1, 2.2, 3.3], u.m)
    got = xp.ceil(x)
    expected = Quantity(xp.ceil(x.value), u.m)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_conj():
    """Test `conj`."""
    x = Quantity([1 + 2j, 3 + 4j], u.m)
    got = xp.conj(x)
    expected = Quantity(xp.conj(x.value), u.m)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_cos():
    """Test `cos`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.rad)
    got = xp.cos(x)
    expected = Quantity(xp.cos(x.value), u.one)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_cosh():
    """Test `cosh`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.rad)
    got = xp.cosh(x)
    expected = Quantity(xp.cosh(x.value), u.one)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_divide():
    """Test `divide`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.m)
    y = Quantity(xp.asarray([4, 5, 6], dtype=float), u.km)
    got = xp.divide(x, y)
    expected = Quantity(xp.divide(x.value, y.value), u.m / u.km)

    assert isinstance(got, Quantity)
    assert got.unit.is_equivalent(expected.unit)
    assert jnp.array_equal(got.value, expected.value)


def test_equal():
    """Test `equal`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.m)
    y = Quantity(xp.asarray([4, 5, 6], dtype=float), u.m)
    got = xp.equal(x, y)
    expected = xp.equal(x.value, y.value)

    assert isinstance(got, Array)
    assert jnp.array_equal(got, expected)


def test_exp():
    """Test `exp`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.one)
    got = xp.exp(x)
    expected = Quantity(xp.exp(x.value), u.one)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_expm1():
    """Test `expm1`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.one)
    got = xp.expm1(x)
    expected = Quantity(xp.expm1(x.value), u.one)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_floor():
    """Test `floor`."""
    x = Quantity([1.1, 2.2, 3.3], u.m)
    got = xp.floor(x)
    expected = Quantity(xp.floor(x.value), u.m)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_floor_divide():
    """Test `floor_divide`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.m)
    y = Quantity(xp.asarray([4, 5, 6], dtype=float), u.m)
    got = xp.floor_divide(x, y)
    expected = Quantity(xp.floor_divide(x.value, y.value), u.m / u.m)

    assert isinstance(got, Quantity)
    assert got.unit.is_equivalent(expected.unit)
    assert jnp.array_equal(got.value, expected.value)


def test_greater():
    """Test `greater`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.m)
    y = Quantity(xp.asarray([4, 5, 6], dtype=float), u.m)
    got = xp.greater(x, y)
    expected = xp.greater(x.value, y.value)

    assert isinstance(got, Array)
    assert jnp.array_equal(got, expected)


def test_greater_equal():
    """Test `greater_equal`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.m)
    y = Quantity(xp.asarray([4, 5, 6], dtype=float), u.m)
    got = xp.greater_equal(x, y)
    expected = xp.greater_equal(x.value, y.value)

    assert isinstance(got, Array)
    assert jnp.array_equal(got, expected)


def test_imag():
    """Test `imag`."""
    x = Quantity([1 + 2j, 3 + 4j], u.m)
    got = xp.imag(x)
    expected = Quantity(xp.imag(x.value), u.m)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_isfinite():
    """Test `isfinite`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.m)
    got = xp.isfinite(x)
    expected = xp.isfinite(x.value)

    assert isinstance(got, Array)
    assert jnp.array_equal(got, expected)


def test_isinf():
    """Test `isinf`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.m)
    got = xp.isinf(x)
    expected = xp.isinf(x.value)

    assert isinstance(got, Array)
    assert jnp.array_equal(got, expected)


def test_isnan():
    """Test `isnan`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.m)
    got = xp.isnan(x)
    expected = xp.isnan(x.value)

    assert isinstance(got, Array)
    assert jnp.array_equal(got, expected)


def test_less():
    """Test `less`."""
    x = Quantity([1, 5, 3], u.m)
    y = Quantity([4, 2, 6], u.m)
    got = xp.less(x, y)
    expected = xp.less(x.value, y.value)

    assert isinstance(got, Array)
    assert jnp.array_equal(got, expected)


def test_less_equal():
    """Test `less_equal`."""
    x = Quantity([1, 5, 3], u.m)
    y = Quantity([4, 2, 6], u.m)
    got = xp.less_equal(x, y)
    expected = xp.less_equal(x.value, y.value)

    assert isinstance(got, Array)
    assert jnp.array_equal(got, expected)


def test_log():
    """Test `log`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.one)
    got = xp.log(x)
    expected = Quantity(xp.log(x.value), u.one)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_log1p():
    """Test `log1p`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.one)
    got = xp.log1p(x)
    expected = Quantity(xp.log1p(x.value), u.one)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_log2():
    """Test `log2`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.one)
    got = xp.log2(x)
    expected = Quantity(xp.log2(x.value), u.one)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_log10():
    """Test `log10`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.one)
    got = xp.log10(x)
    expected = Quantity(xp.log10(x.value), u.one)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert qnp.allclose(got, expected, atol=Quantity(1e-8, ""))


def test_logaddexp():
    """Test `logaddexp`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.one)
    y = Quantity(xp.asarray([4, 5, 6], dtype=float), u.one)
    got = xp.logaddexp(x, y)
    expected = Quantity(xp.logaddexp(x.value, y.value), u.one)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_logical_and():
    """Test `logical_and`."""
    x = Quantity([True, False, True], u.one)
    y = Quantity([False, True, False], u.one)
    got = xp.logical_and(x, y)
    expected = xp.logical_and(x.value, y.value)

    assert isinstance(got, Array)
    assert jnp.array_equal(got, expected)


@pytest.mark.xfail(reason="TODO")
def test_logical_not():
    """Test `logical_not`."""
    x = Quantity([True, False, True], u.one)
    got = xp.logical_not(x)
    expected = Quantity(xp.logical_not(x.value), u.one)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


@pytest.mark.xfail(reason="TODO")
def test_logical_or():
    """Test `logical_or`."""
    x = Quantity([True, False, True], u.one)
    y = Quantity([False, True, False], u.one)
    got = xp.logical_or(x, y)
    expected = Quantity(xp.logical_or(x.value, y.value), u.one)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


@pytest.mark.xfail(reason="TODO")
def test_logical_xor():
    """Test `logical_xor`."""
    x = Quantity([True, False, True], u.one)
    y = Quantity([False, True, False], u.one)
    got = xp.logical_xor(x, y)
    expected = Quantity(xp.logical_xor(x.value, y.value), u.one)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_multiply():
    """Test `multiply`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.m)
    y = Quantity(xp.asarray([4, 5, 6], dtype=float), u.m)
    got = xp.multiply(x, y)
    expected = Quantity(xp.multiply(x.value, y.value), u.m**2)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_negative():
    """Test `negative`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.m)
    got = xp.negative(x)
    expected = Quantity(xp.negative(x.value), u.m)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_not_equal():
    """Test `not_equal`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.m)
    y = Quantity(xp.asarray([4, 2, 6], dtype=float), u.m)
    got = xp.not_equal(x, y)
    expected = xp.not_equal(x.value, y.value)

    assert isinstance(got, Array)
    assert jnp.array_equal(got, expected)


def test_positive():
    """Test `positive`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.m)
    got = xp.positive(x)
    expected = Quantity(xp.positive(x.value), u.m)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_pow_quantity_power():
    """Test `pow`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.m)
    y = Quantity(xp.asarray([4], dtype=float), u.one)
    got = xp.pow(x, y)
    expected = Quantity(xp.pow(x.value, y.value), u.m**4)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_pow():
    """Test `pow`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.m)
    y = xp.asarray([4], dtype=float)
    got = xp.pow(x, y)
    expected = Quantity(xp.pow(x.value, y), u.m**4)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_real():
    """Test `real`."""
    x = Quantity([1 + 2j, 3 + 4j], u.m)
    got = xp.real(x)
    expected = Quantity(xp.real(x.value), u.m)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


@pytest.mark.skip(reason="TODO")
def test_remainder():
    """Test `remainder`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.m)
    y = Quantity(xp.asarray([4, 5, 6], dtype=float), u.m)
    got = xp.remainder(x, y)
    expected = Quantity(xp.remainder(x.value, y.value), u.m)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_round():
    """Test `round`."""
    x = Quantity([1.1, 2.2, 3.3], u.m)
    got = xp.round(x)
    expected = Quantity(xp.round(x.value), u.m)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_sign():
    """Test `sign`."""
    x = Quantity([-1, 2, -3], u.m)
    got = xp.sign(x)
    expected = xp.sign(x.value)

    assert isinstance(got, Array)
    assert jnp.array_equal(got, expected)


def test_sin():
    """Test `sin`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.rad)
    got = xp.sin(x)
    expected = Quantity(xp.sin(x.value), u.one)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_sinh():
    """Test `sinh`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.rad)
    got = xp.sinh(x)
    expected = Quantity(xp.sinh(x.value), u.one)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_square():
    """Test `square`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.m)
    got = xp.square(x)
    expected = Quantity(xp.square(x.value), u.m**2)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_sqrt():
    """Test `sqrt`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.m)
    got = xp.sqrt(x)
    expected = Quantity(xp.sqrt(x.value), u.m**0.5)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_subtract():
    """Test `subtract`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.m)
    y = Quantity(xp.asarray([4, 5, 6], dtype=float), u.m)
    got = xp.subtract(x, y)
    expected = Quantity(xp.subtract(x.value, y.value), u.m)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_tan():
    """Test `tan`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.rad)
    got = xp.tan(x)
    expected = Quantity(xp.tan(x.value), u.one)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_tanh():
    """Test `tanh`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.rad)
    got = xp.tanh(x)
    expected = Quantity(xp.tanh(x.value), u.one)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


@pytest.mark.skip(reason="TODO")
def test_trunc():
    """Test `trunc`."""
    x = Quantity([1.1, 2.2, 3.3], u.m)
    got = xp.trunc(x)
    expected = Quantity(xp.trunc(x.value), u.m)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


# =============================================================================
# Indexing functions


@pytest.mark.skip(reason="TODO")
def test_take():
    """Test `take`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.m)
    indices = Quantity(xp.asarray([0, 1, 2], dtype=int), u.one)
    got = xp.take(x, indices)
    expected = Quantity(xp.take(x.value, indices.value, axis=None), u.m)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


# =============================================================================
# Linear algebra functions


@pytest.mark.skip(reason="TODO")
def test_matmul():
    """Test `matmul`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.m)
    y = Quantity(xp.asarray([4, 5, 6], dtype=float), u.m)
    got = xp.matmul(x, y)
    expected = Quantity(xp.matmul(x.value, y.value), u.m**2)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_matrix_transpose():
    """Test `matrix_transpose`."""
    x = Quantity(xp.asarray([[1, 2, 3], [4, 5, 6]], dtype=float), u.m)
    got = xp.matrix_transpose(x)
    expected = Quantity(xp.matrix_transpose(x.value), u.m)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


@pytest.mark.skip(reason="TODO")
def test_tensordot():
    """Test `tensordot`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.m)
    y = Quantity(xp.asarray([4, 5, 6], dtype=float), u.m)
    axes = 1
    got = xp.tensordot(x, y, axes=axes)
    expected = Quantity(xp.tensordot(x.value, y.value, axes=axes), u.m**2)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


@pytest.mark.skip(reason="TODO")
def test_vecdot():
    """Test `vecdot`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.m)
    y = Quantity(xp.asarray([4, 5, 6], dtype=float), u.m)
    got = xp.vecdot(x, y)
    expected = Quantity(xp.vecdot(x.value, y.value), u.m**2)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


# =============================================================================
# Manipulation functions


def test_broadcast_arrays():
    """Test `broadcast_arrays`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.m)
    y = Quantity(xp.asarray([4], dtype=float), u.s)
    got = xp.broadcast_arrays(x, y)
    expected = xp.broadcast_arrays(x.value, y.value)

    assert isinstance(got, tuple | list)
    assert len(got) == len(expected)
    for got_, expected_ in zip(got, expected, strict=True):
        assert isinstance(got_, Quantity)
        assert jnp.array_equal(got_.value, expected_)


def test_broadcast_to():
    """Test `broadcast_to`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.m)
    shape = (2, 3)
    got = xp.broadcast_to(x, shape)
    expected = Quantity(xp.broadcast_to(x.value, shape), u.m)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_concat():
    """Test `concat`."""
    # TODO: test the axis argument
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.m)
    y = Quantity(xp.asarray([4], dtype=float), u.m)
    got = xp.concat((x, y))
    expected = Quantity(xp.concat((x.value, y.value)), u.m)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_expand_dims():
    """Test `expand_dims`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.m)
    got = xp.expand_dims(x, axis=0)
    expected = Quantity(xp.expand_dims(x.value, axis=0), u.m)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_flip():
    """Test `flip`."""
    x = Quantity(xp.asarray([[1, 2, 3], [4, 5, 6]], dtype=float), u.m)
    got = xp.flip(x)
    expected = Quantity(xp.flip(x.value), u.m)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_permute_dims():
    """Test `permute_dims`."""
    x = Quantity(xp.asarray([[1, 2, 3], [4, 5, 6]], dtype=float), u.m)
    got = xp.permute_dims(x, (1, 0))
    expected = Quantity(xp.permute_dims(x.value, (1, 0)), u.m)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_reshape():
    """Test `reshape`."""
    x = Quantity(xp.asarray([[1, 2, 3], [4, 5, 6]], dtype=float), u.m)
    got = xp.reshape(x, (3, 2))
    expected = Quantity(xp.reshape(x.value, (3, 2)), u.m)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_roll():
    """Test `roll`."""
    x = Quantity(xp.asarray([[1, 2, 3], [4, 5, 6]], dtype=float), u.m)
    got = xp.roll(x, shift=1, axis=0)
    expected = Quantity(xp.roll(x.value, shift=1, axis=0), u.m)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_squeeze():
    """Test `squeeze`."""
    x = Quantity(xp.asarray([[[0], [1], [2]]], dtype=float), u.m)
    got = xp.squeeze(x, axis=(0, 2))
    expected = Quantity(xp.squeeze(x.value, axis=(0, 2)), u.m)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_stack():
    """Test `stack`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.m)
    y = Quantity(xp.asarray([4, 5, 6], dtype=float), u.m)
    got = xp.stack((x, y))
    expected = Quantity(xp.stack((x.value, y.value)), u.m)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


# =============================================================================
# Searching functions


def test_argmax():
    """Test `argmax`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.m)
    got = xp.argmax(x)
    expected = Quantity(xp.argmax(x.value), u.m)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_argmin():
    """Test `argmin`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.m)
    got = xp.argmin(x)
    expected = Quantity(xp.argmin(x.value), u.m)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


@pytest.mark.xfail(reason="TODO")
def test_nonzero():
    """Test `nonzero`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.m)
    (got,) = xp.nonzero(x)
    (expected,) = Quantity(xp.nonzero(x.value), u.one)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected)


def test_where():
    """Test `where`."""
    condition = Quantity(xp.asarray([True, False, True]), u.one)
    y = Quantity(xp.asarray([1, 2, 3], dtype=float), u.m)
    z = Quantity(xp.asarray([4, 5, 6], dtype=float), u.m)
    got = xp.where(condition, y, z)
    expected = Quantity(xp.where(condition.value, y.value, z.value), u.m)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


# =============================================================================
# Set functions


@pytest.mark.xfail(reason="value is not a Quantity")
def test_unique_all():
    """Test `unique_all`."""
    x = Quantity(xp.asarray([1, 2, 2, 3, 3, 3], dtype=float))
    got = xp.unique_all(x)
    expected = xp.unique_all(x.value)

    assert isinstance(got, _UniqueAllResult)

    assert isinstance(got.values, Quantity)
    assert jnp.array_equal(got.values, expected.values)

    assert isinstance(got.inverse, Quantity)
    assert jnp.array_equal(got.inverse, expected.inverse)

    assert isinstance(got.inverse_indices, Array)
    assert jnp.array_equal(got.inverse_indices, expected.inverse_indices)

    assert isinstance(got.counts, Array)
    assert jnp.array_equal(got.counts, expected.counts)


@pytest.mark.xfail(reason="value is not a Quantity")
def test_unique_counts():
    """Test `unique_counts`."""
    x = Quantity(xp.asarray([1, 2, 2, 3, 3, 3], dtype=float))
    got = xp.unique_counts(x)
    expected = xp.unique_counts(x.value)

    assert isinstance(got, _UniqueCountsResult)

    assert isinstance(got.values, Quantity)
    assert jnp.array_equal(got.values.value, expected.values)

    assert isinstance(got.counts, Array)
    assert jnp.array_equal(got.counts, expected.counts)


@pytest.mark.xfail(reason="value is not a Quantity")
def test_unique_inverse():
    """Test `unique_inverse`."""
    x = Quantity(xp.asarray([1, 2, 2, 3, 3, 3], dtype=float))
    got = xp.unique_inverse(x)
    expected = xp.unique_inverse(x.value)

    assert isinstance(got, _UniqueInverseResult)

    assert isinstance(got.values, Quantity)
    assert jnp.array_equal(got.values.value, expected.values)

    assert isinstance(got.inverse, Quantity)
    assert jnp.array_equal(got.inverse.value, expected.inverse)


@pytest.mark.xfail(reason="value is not a Quantity")
def test_unique_values():
    """Test `unique_values`."""
    x = Quantity(xp.asarray([1, 2, 2, 3, 3, 3], dtype=float))
    got = xp.unique_values(x)
    expected = Quantity(xp.unique_values(x.value), u.one)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


# =============================================================================
# Sorting functions


def test_argsort():
    """Test `argsort`."""
    q = Quantity(xp.asarray([3, 2, 1], dtype=float), u.m)
    got = xp.argsort(q)

    assert isinstance(got, Array)
    assert jnp.array_equal(got, xp.argsort(q.value))


def test_sort():
    """Test `sort`."""
    q = Quantity(xp.asarray([3, 2, 1], dtype=float), u.m)
    got = xp.sort(q)

    assert isinstance(got, Quantity)
    assert got.unit == u.m
    assert jnp.array_equal(got.value, xp.sort(q.value))


# =============================================================================
# Statistical functions


def test_max():
    """Test `max`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.m)
    got = xp.max(x)
    expected = Quantity(xp.max(x.value), u.m)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


@pytest.mark.skip("TODO")
def test_mean():
    """Test `mean`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.m)
    got = xp.mean(x)
    expected = Quantity(xp.mean(x.value), u.m)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


@pytest.mark.skip("TODO")
def test_min():
    """Test `min`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.m)
    got = xp.min(x)
    expected = Quantity(xp.min(x.value), u.m)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


@pytest.mark.filterwarnings("ignore:Explicitly requested dtype")  # TODO: Why?
def test_prod():
    """Test `prod`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.m)
    got = xp.prod(x)
    expected = Quantity(xp.prod(x.value), u.m**3)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_std():
    """Test `std`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.m)
    got = xp.std(x)
    expected = Quantity(xp.std(x.value), u.m)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


@pytest.mark.filterwarnings("ignore:Explicitly requested dtype")  # TODO: Why?
def test_sum():
    """Test `sum`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.m)
    got = xp.sum(x)
    expected = Quantity(xp.sum(x.value), u.m)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_var():
    """Test `var`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=float), u.m)
    got = xp.var(x)
    expected = Quantity(xp.var(x.value), u.m**2)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


# =============================================================================
# Utility functions


@pytest.mark.xfail(reason="returns a jax.Array")
def test_all():
    """Test `all`."""
    x = Quantity(xp.asarray([True, False, True], dtype=bool), u.one)
    got = xp.all(x)
    expected = Quantity(xp.all(x.value), u.one)

    assert isinstance(got, Quantity)
    assert got.unit == expected.unit
    assert jnp.array_equal(got.value, expected.value)


def test_any():
    """Test `any`."""
    x = Quantity(xp.asarray([1, 2, 3], dtype=bool), u.m)
    got = xp.any(x)
    expected = Quantity(xp.any(x.value), u.one)

    assert isinstance(got, Quantity)
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
