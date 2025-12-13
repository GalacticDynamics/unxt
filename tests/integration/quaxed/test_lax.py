"""Test `quaxed.lax` functions."""
# ruff:noqa: N816

import operator as ops

import jax.numpy as jnp
import jax.tree as jtu
import pytest
from astropy.units import UnitBase
from jax import lax

import quaxed.lax as qlax

import unxt as u
from unxt.quantity import is_any_quantity

# ------------
# Pytest marks

mark_todo = pytest.mark.skip(reason="TODO: Implement this function.")

# ------------

x_val = jnp.array([[1, 2], [3, 4]], dtype=float)
x_L = u.Quantity(x_val, unit="m")
x_ND = u.Quantity(x_val, unit="")

y_val = jnp.array([[5, 6], [7, 8]], dtype=float)
y_L = u.Quantity(y_val, unit="m")
y_ND = u.Quantity(y_val, unit="")

xtrig_val = x_val / 10
xtrig = u.Quantity(xtrig_val, unit="")

xbit_val = jnp.array([[1, 0], [1, 0]], dtype=int)
xbit = u.Quantity(xbit_val, unit="")

xcomplex_val = jnp.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype=complex)
xcomplex = u.Quantity(xcomplex_val, unit="m")

conv_kernel = u.Quantity(
    jnp.array([[[[1.0, 0.0], [0.0, -1.0]]]], dtype=float), unit="m"
)

xround_val = jnp.array([[1.1, 2.2], [3.3, 4.4]], dtype=float)
xround = u.Quantity(xround_val, unit="m")


# ------------


@pytest.mark.parametrize(
    ("func_name", "args", "kw", "expected"),
    [
        ("abs", (x_L,), {}, u.Quantity(lax.abs(x_val), unit="m")),
        ("acos", (xtrig,), {}, u.Quantity(lax.acos(xtrig_val), unit="rad")),
        ("acosh", (x_ND,), {}, u.Quantity(lax.acosh(x_val), unit="rad")),
        ("add", (x_L, y_L), {}, u.Quantity(lax.add(x_val, y_val), unit="m")),
        pytest.param("after_all", (), {}, True, marks=mark_todo),
        (
            "approx_max_k",
            (x_L,),
            {"k": 2},
            [
                u.Quantity([[2.0, 1], [4, 3]], unit="m"),
                u.Quantity([[1.0, 0], [1, 0]], unit="m"),
            ],
        ),
        (
            "approx_min_k",
            (x_L,),
            {"k": 2},
            [
                u.Quantity([[1.0, 2.0], [3.0, 4.0]], unit="m"),
                u.Quantity([[0.0, 1.0], [0.0, 1.0]], unit="m"),
            ],
        ),
        ("argmax", (x_L,), {"axis": 0, "index_dtype": int}, jnp.array([1, 1])),
        ("argmin", (x_L,), {"axis": 0, "index_dtype": int}, jnp.array([0, 0])),
        ("asin", (xtrig,), {}, u.Quantity(lax.asin(xtrig_val), unit="rad")),
        ("asinh", (xtrig,), {}, u.Quantity(lax.asinh(xtrig_val), unit="rad")),
        ("atan", (xtrig,), {}, u.Quantity(lax.atan(xtrig_val), unit="rad")),
        ("atan2", (x_L, y_L), {}, u.Quantity(lax.atan2(x_val, y_val), unit="rad")),
        ("atanh", (xtrig,), {}, u.Quantity(lax.atanh(xtrig_val), unit="rad")),
        (
            "batch_matmul",
            (
                u.Quantity(
                    jnp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=float),
                    unit="m",
                ),
                u.Quantity(
                    jnp.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=float),
                    unit="m",
                ),
            ),
            {},
            u.Quantity(
                [[[31.0, 34.0], [71.0, 78.0]], [[155.0, 166.0], [211.0, 226.0]]],
                unit="m2",
            ),
        ),
        ("bessel_i0e", (x_ND,), {}, u.Quantity(lax.bessel_i0e(x_val), unit="")),
        ("bessel_i1e", (x_ND,), {}, u.Quantity(lax.bessel_i1e(x_val), unit="")),
        (
            "betainc",
            (1.0, xtrig, xtrig),
            {},
            u.Quantity(lax.betainc(1.0, xtrig_val, xtrig_val), unit=""),
        ),
        (
            "bitcast_convert_type",
            (x_L, jnp.float16),
            {},
            u.Quantity(lax.bitcast_convert_type(x_val, jnp.float16), unit="m"),
        ),
        ("bitwise_and", (xbit, xbit), {}, lax.bitwise_and(xbit_val, xbit_val)),
        ("bitwise_not", (xbit,), {}, u.Quantity(lax.bitwise_not(xbit_val), unit="")),
        (
            "bitwise_or",
            (xbit, xbit),
            {},
            u.Quantity(lax.bitwise_or(xbit_val, xbit_val), unit=""),
        ),
        (
            "bitwise_xor",
            (xbit, xbit),
            {},
            u.Quantity(lax.bitwise_xor(xbit_val, xbit_val), unit=""),
        ),
        (
            "broadcast",
            (x_L,),
            {"sizes": (2, 2)},
            u.Quantity(lax.broadcast(x_val, (2, 2)), unit="m"),
        ),
        (
            "broadcast_in_dim",
            (x_L, (1, 1, 2, 2), (2, 3)),
            {},
            u.Quantity(lax.broadcast_in_dim(x_val, (1, 1, 2, 2), (2, 3)), unit="m"),
        ),
        (
            "broadcast_to_rank",
            (x_L,),
            {"rank": 3},
            u.Quantity(lax.broadcast_to_rank(x_val, rank=3), unit="m"),
        ),
        pytest.param("broadcasted_iota", (), {}, True, marks=mark_todo),
        ("cbrt", (x_L,), {}, u.Quantity(lax.cbrt(x_val), unit=u.unit("m**(1/3)"))),
        ("ceil", (x_L,), {}, u.Quantity(lax.ceil(x_val), unit="m")),
        (
            "clamp",
            (u.Quantity(2.0, "m"), x_L, u.Quantity(3.0, "m")),
            {},
            u.Quantity(lax.clamp(2.0, x_val, 3.0), unit="m"),
        ),
        (
            "clz",
            (x_L.astype(int),),
            {},
            u.Quantity(lax.clz(x_val.astype(int)), unit="m"),
        ),
        ("collapse", (x_L, 1), {}, u.Quantity(lax.collapse(x_val, 1), unit="m")),
        (
            "concatenate",
            ((x_L, y_L), 0),
            {},
            u.Quantity(jnp.concatenate((x_val, y_val), 0), unit="m"),
        ),
        ("conj", (xcomplex,), {}, u.Quantity(lax.conj(xcomplex_val), unit="m")),
        pytest.param(
            "conv",
            (
                u.Quantity(
                    jnp.arange(1, 17, dtype=float).reshape((1, 1, 4, 4)), unit="m"
                ),
                conv_kernel,
            ),
            {"window_strides": (1, 1), "padding": "SAME"},
            None,
            marks=mark_todo,
        ),
        (
            "convert_element_type",
            (x_L, jnp.int16),
            {},
            u.Quantity(lax.convert_element_type(x_val, jnp.int16), unit="m"),
        ),
        pytest.param(
            "conv_general_dilated",
            (
                u.Quantity(
                    jnp.arange(1, 17, dtype=float).reshape((1, 1, 4, 4)), unit="m"
                ),
                conv_kernel,
            ),
            {"window_strides": (1, 1), "padding": "SAME"},
            None,
            marks=mark_todo,
        ),
        pytest.param("conv_general_dilated_local", (), {}, True, marks=mark_todo),
        pytest.param(
            "conv_general_dilated_patches",
            (
                u.Quantity(
                    jnp.arange(1, 17, dtype=float).reshape((1, 1, 4, 4)), unit="m"
                ),
            ),
            {"filter_shape": (2, 2), "window_strides": (1, 1), "padding": "VALID"},
            True,
            marks=mark_todo,
        ),
        pytest.param(
            "conv_transpose",
            (
                u.Quantity(
                    jnp.arange(1, 17, dtype=float).reshape((1, 1, 4, 4)), unit="m"
                ),
                conv_kernel,
            ),
            {
                "strides": (2, 2),
                "padding": "SAME",
                "dimension_numbers": ("NCHW", "OIHW", "NCHW"),
            },
            True,
            marks=mark_todo,
        ),
        pytest.param("conv_with_general_padding", (), {}, True, marks=mark_todo),
        ("cos", (x_ND,), {}, u.Quantity(lax.cos(x_val), unit="")),
        ("cosh", (x_ND,), {}, u.Quantity(lax.cosh(x_val), unit="")),
        (
            "cumlogsumexp",
            (x_L,),
            {"axis": 0},
            u.Quantity(lax.cumlogsumexp(x_val), unit="m"),
        ),
        ("cummax", (x_L,), {"axis": 0}, u.Quantity(lax.cummax(x_val), unit="m")),
        ("cummin", (x_L,), {"axis": 0}, u.Quantity(lax.cummin(x_val), unit="m")),
        ("cumprod", (x_ND,), {"axis": 0}, u.Quantity(lax.cumprod(x_val), unit="")),
        ("cumsum", (x_L,), {"axis": 0}, u.Quantity(lax.cumsum(x_val), unit="m")),
        ("digamma", (x_ND,), {}, u.Quantity(lax.digamma(x_val), unit="")),
        ("div", (x_L, y_L), {}, u.Quantity(lax.div(x_val, y_val), unit="")),
        ("dot", (x_L, y_L), {}, u.Quantity(lax.dot(x_val, y_val), unit="m2")),
        pytest.param("dot_general", (), {}, True, marks=mark_todo),
        pytest.param("dynamic_index_in_dim", (), {}, True, marks=mark_todo),
        (
            "dynamic_slice",
            (x_L, (0, 0), (2, 2)),
            {},
            u.Quantity(lax.dynamic_slice(x_val, (0, 0), (2, 2)), unit="m"),
        ),
        pytest.param("dynamic_slice_in_dim", (), {}, True, marks=mark_todo),
        pytest.param("dynamic_update_index_in_dim", (), {}, True, marks=mark_todo),
        (
            "dynamic_update_slice",
            (x_L, y_L, (0, 0)),
            {},
            u.Quantity(lax.dynamic_update_slice(x_val, y_val, (0, 0)), unit="m"),
        ),
        (
            "dynamic_update_slice_in_dim",
            (x_L, y_L, 0, 0),
            {},
            u.Quantity(lax.dynamic_update_slice_in_dim(x_val, y_val, 0, 0), unit="m"),
        ),
        ("eq", (x_L, x_L), {}, lax.eq(x_val, x_val)),
        ("erf", (xtrig,), {}, u.Quantity(lax.erf(xtrig_val), unit="")),
        ("erfc", (xtrig,), {}, u.Quantity(lax.erfc(xtrig_val), unit="")),
        ("erf_inv", (xtrig,), {}, u.Quantity(lax.erf_inv(xtrig_val), unit="")),
        ("exp", (x_ND,), {}, u.Quantity(lax.exp(x_val), unit="")),
        ("exp2", (x_ND,), {}, u.Quantity(lax.exp2(x_val), unit="")),
        (
            "expand_dims",
            (x_L, (0,)),
            {},
            u.Quantity(lax.expand_dims(x_val, (0,)), unit="m"),
        ),
        ("expm1", (x_ND,), {}, u.Quantity(lax.expm1(x_val), unit="")),
        (
            "fft",
            (x_L,),
            {"fft_type": "fft", "fft_lengths": (2, 2)},
            u.Quantity(lax.fft(x_val, fft_type="fft", fft_lengths=(2, 2)), unit="m-1"),
        ),
        ("floor", (xround,), {}, u.Quantity(lax.floor(xround_val), unit="m")),
        (
            "full_like",
            (x_L, u.Quantity(1.0, "m")),
            {},
            u.Quantity(lax.full_like(x_val, 1.0), unit="m"),
        ),
        pytest.param("gather", (), {}, True, marks=mark_todo),
        ("ge", (x_L, y_L), {}, lax.ge(x_val, y_val)),
        ("gt", (x_L, y_L), {}, lax.gt(x_val, y_val)),
        ("igamma", (1.0, xtrig), {}, u.Quantity(lax.igamma(1.0, xtrig_val), unit="")),
        ("igammac", (1.0, xtrig), {}, u.Quantity(lax.igammac(1.0, xtrig_val), unit="")),
        ("imag", (xcomplex,), {}, u.Quantity(lax.imag(xcomplex_val), unit="m")),
        (
            "index_in_dim",
            (x_L, 0, 0),
            {},
            u.Quantity(lax.index_in_dim(x_val, 0, 0), unit="m"),
        ),
        pytest.param("index_take", (), {}, True, marks=mark_todo),
        ("integer_pow", (x_L, 2), {}, u.Quantity(lax.integer_pow(x_val, 2), unit="m2")),
        pytest.param("iota", (), {}, True, marks=mark_todo),
        ("is_finite", (x_L,), {}, lax.is_finite(x_val)),
        ("le", (x_L, y_L), {}, lax.le(x_val, y_val)),
        ("lgamma", (x_ND,), {}, u.Quantity(lax.lgamma(x_val), unit="")),
        ("log", (x_ND,), {}, u.Quantity(lax.log(x_val), unit="")),
        ("log1p", (x_ND,), {}, u.Quantity(lax.log1p(x_val), unit="")),
        ("logistic", (x_ND,), {}, u.Quantity(lax.logistic(x_val), unit="")),
        ("lt", (x_L, y_L), {}, lax.lt(x_val, y_val)),
        ("max", (x_L, y_L), {}, u.Quantity(lax.max(x_val, y_val), unit="m")),
        ("min", (x_L, y_L), {}, u.Quantity(lax.min(x_val, y_val), unit="m")),
        ("mul", (x_L, y_L), {}, u.Quantity(lax.mul(x_val, y_val), unit="m2")),
        ("ne", (x_L, y_L), {}, lax.ne(x_val, y_val)),
        ("neg", (x_L,), {}, u.Quantity(lax.neg(x_val), unit="m")),
        (
            "nextafter",
            (x_L, y_L),
            {},
            u.Quantity(lax.nextafter(x_val, y_val), unit="m"),
        ),
        pytest.param("pad", (), {}, True, marks=mark_todo),
        (
            "polygamma",
            (1.0, xtrig),
            {},
            u.Quantity(lax.polygamma(1.0, xtrig_val), unit=""),
        ),
        (
            "population_count",
            (xbit,),
            {},
            u.Quantity(lax.population_count(xbit_val), unit=""),
        ),
        ("pow", (x_L, 2), {}, u.Quantity(lax.pow(x_val, 2), unit="m2")),
        pytest.param("random_gamma_grad", (1.0, x_ND), {}, True, marks=mark_todo),
        ("real", (xcomplex,), {}, u.Quantity(lax.real(xcomplex_val), unit="m")),
        ("reciprocal", (x_L,), {}, u.Quantity(lax.reciprocal(x_val), unit="m-1")),
        pytest.param("reduce", (), {}, True, marks=mark_todo),
        pytest.param("reduce_precision", (), {}, True, marks=mark_todo),
        pytest.param("reduce_window", (), {}, True, marks=mark_todo),
        ("rem", (x_L, y_L), {}, u.Quantity(lax.rem(x_val, y_val), unit="m")),
        (
            "reshape",
            (x_L, (1, 4)),
            {},
            u.Quantity(lax.reshape(x_val, (1, 4)), unit="m"),
        ),
        (
            "rev",
            (x_L,),
            {"dimensions": (0,)},
            u.Quantity(lax.rev(x_val, (0,)), unit="m"),
        ),
        pytest.param("rng_bit_generator", (), {}, True, marks=mark_todo),
        ("round", (xround,), {}, u.Quantity(lax.round(xround_val), unit="m")),
        ("rsqrt", (x_L**2,), {}, u.Quantity(lax.rsqrt(x_val**2), unit="m-1")),
        pytest.param("scatter", (), {}, True, marks=mark_todo),
        pytest.param("scatter_apply", (), {}, True, marks=mark_todo),
        pytest.param("scatter_max", (), {}, True, marks=mark_todo),
        pytest.param("scatter_min", (), {}, True, marks=mark_todo),
        pytest.param("scatter_mul", (), {}, True, marks=mark_todo),
        (
            "shift_left",
            (xbit, xbit),
            {},
            u.Quantity(lax.shift_left(xbit_val, xbit_val), unit=""),
        ),
        pytest.param("shift_right_arithmetic", (xbit, 1), {}, True, marks=mark_todo),
        pytest.param("shift_right_logical", (xbit, 1), {}, True, marks=mark_todo),
        ("sign", (x_L,), {}, lax.sign(x_val)),
        ("sin", (x_ND,), {}, u.Quantity(lax.sin(x_val), unit="")),
        ("sinh", (x_ND,), {}, u.Quantity(lax.sinh(x_val), unit="")),
        (
            "slice",
            (x_L, (0, 0), (2, 2)),
            {},
            u.Quantity(lax.slice(x_val, (0, 0), (2, 2)), unit="m"),
        ),
        (
            "slice_in_dim",
            (x_L, 0, 1, 2),
            {},
            u.Quantity(lax.slice_in_dim(x_val, 0, 1, 2), unit="m"),
        ),
        ("sort", (x_L,), {}, u.Quantity(lax.sort(x_val), unit="m")),
        pytest.param("sort_key_val", (), {}, True, marks=mark_todo),
        ("sqrt", (x_L**2,), {}, u.Quantity(lax.sqrt(x_val**2), unit="m")),
        ("square", (x_L,), {}, u.Quantity(lax.square(x_val), unit="m2")),
        ("sub", (x_L, y_L), {}, u.Quantity(lax.sub(x_val, y_val), unit="m")),
        ("tan", (x_ND,), {}, u.Quantity(lax.tan(x_val), unit="")),
        ("tanh", (x_ND,), {}, u.Quantity(lax.tanh(x_val), unit="")),
        (
            "top_k",
            (x_L, 1),
            {},
            [
                u.Quantity([[2.0], [4.0]], unit="m"),
                u.Quantity([[1.0], [1.0]], unit="m"),
            ],
        ),
        (
            "transpose",
            (x_L, (1, 0)),
            {},
            u.Quantity(lax.transpose(x_val, (1, 0)), unit="m"),
        ),
        ("zeta", (x_L, 2.0), {}, u.Quantity(lax.zeta(x_val, 2.0), unit="m")),
        pytest.param("associative_scan", (), {}, True, marks=mark_todo),
        pytest.param("fori_loop", (), {}, True, marks=mark_todo),
        pytest.param("scan", (), {}, True, marks=mark_todo),
        (
            "select",
            (jnp.array([[True, False], [True, False]], dtype=bool), x_L, y_L),
            {},
            u.Quantity(
                lax.select(
                    jnp.array([[True, False], [True, False]], dtype=bool), x_val, y_val
                ),
                unit="m",
            ),
        ),
        pytest.param("select_n", (), {}, True, marks=mark_todo),
        pytest.param("switch", (), {}, True, marks=mark_todo),
        (
            "while_loop",
            (lambda x: jnp.all(x < 10), lambda x: x + 1, x_ND),
            {},
            u.Quantity(
                lax.while_loop(lambda x: jnp.all(x < 10), lambda x: x + 1, x_val),
                unit="",
            ),
        ),
        ("stop_gradient", (x_L,), {}, u.Quantity(lax.stop_gradient(x_val), unit="m")),
        pytest.param("custom_linear_solve", (), {}, True, marks=mark_todo),
        pytest.param("custom_root", (), {}, True, marks=mark_todo),
        pytest.param("all_gather", (), {}, True, marks=mark_todo),
        pytest.param("all_to_all", (), {}, True, marks=mark_todo),
        pytest.param("psum", (), {}, True, marks=mark_todo),
        pytest.param("psum_scatter", (), {}, True, marks=mark_todo),
        pytest.param("pmax", (), {}, True, marks=mark_todo),
        pytest.param("pmin", (), {}, True, marks=mark_todo),
        pytest.param("pmean", (), {}, True, marks=mark_todo),
        pytest.param("ppermute", (), {}, True, marks=mark_todo),
        pytest.param("pshuffle", (), {}, True, marks=mark_todo),
        pytest.param("pswapaxes", (), {}, True, marks=mark_todo),
        pytest.param("axis_index", (), {}, True, marks=mark_todo),
        # # --- Sharding-related operators ---
        pytest.param("with_sharding_constraint", (), {}, True, marks=mark_todo),
    ],
)
def test_lax_functions(func_name, args, kw, expected):
    """Test lax vs qlax functions."""
    got = getattr(qlax, func_name)(*args, **kw)

    got_vals = jtu.map(
        lambda x: u.ustrip(x) if is_any_quantity(x) else x, got, is_leaf=is_any_quantity
    )
    got_units = jtu.map(u.unit_of, got, is_leaf=is_any_quantity)

    exp_vals = jtu.map(
        lambda x: u.ustrip(x) if is_any_quantity(x) else x,
        expected,
        is_leaf=is_any_quantity,
    )
    exp_units = jtu.map(u.unit_of, expected, is_leaf=is_any_quantity)

    assert jtu.all(jtu.map(jnp.allclose, got_vals, exp_vals))
    assert jtu.all(
        jtu.map(ops.eq, got_units, exp_units, is_leaf=lambda x: isinstance(x, UnitBase))
    )
