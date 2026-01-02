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
x_L = u.Q(x_val, unit="m")
x_ND = u.Q(x_val, unit="")

y_val = jnp.array([[5, 6], [7, 8]], dtype=float)
y_L = u.Q(y_val, unit="m")
y_ND = u.Q(y_val, unit="")

xtrig_val = x_val / 10
xtrig = u.Q(xtrig_val, unit="")

xbit_val = jnp.array([[1, 0], [1, 0]], dtype=int)
xbit = u.Q(xbit_val, unit="")

xcomplex_val = jnp.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype=complex)
xcomplex = u.Q(xcomplex_val, unit="m")

conv_kernel = u.Q(jnp.array([[[[1.0, 0.0], [0.0, -1.0]]]], dtype=float), unit="m")

xround_val = jnp.array([[1.1, 2.2], [3.3, 4.4]], dtype=float)
xround = u.Q(xround_val, unit="m")


# ------------


@pytest.mark.parametrize(
    ("func_name", "args", "kw", "expected"),
    [
        ("abs", (x_L,), {}, u.Q(lax.abs(x_val), unit="m")),
        ("acos", (xtrig,), {}, u.Q(lax.acos(xtrig_val), unit="rad")),
        ("acosh", (x_ND,), {}, u.Q(lax.acosh(x_val), unit="rad")),
        ("add", (x_L, y_L), {}, u.Q(lax.add(x_val, y_val), unit="m")),
        pytest.param("after_all", (), {}, True, marks=mark_todo),
        (
            "approx_max_k",
            (x_L,),
            {"k": 2},
            [u.Q([[2.0, 1], [4, 3]], unit="m"), u.Q([[1.0, 0], [1, 0]], unit="m")],
        ),
        (
            "approx_min_k",
            (x_L,),
            {"k": 2},
            [
                u.Q([[1.0, 2.0], [3.0, 4.0]], unit="m"),
                u.Q([[0.0, 1.0], [0.0, 1.0]], unit="m"),
            ],
        ),
        ("argmax", (x_L,), {"axis": 0, "index_dtype": int}, jnp.array([1, 1])),
        ("argmin", (x_L,), {"axis": 0, "index_dtype": int}, jnp.array([0, 0])),
        ("asin", (xtrig,), {}, u.Q(lax.asin(xtrig_val), unit="rad")),
        ("asinh", (xtrig,), {}, u.Q(lax.asinh(xtrig_val), unit="rad")),
        ("atan", (xtrig,), {}, u.Q(lax.atan(xtrig_val), unit="rad")),
        ("atan2", (x_L, y_L), {}, u.Q(lax.atan2(x_val, y_val), unit="rad")),
        ("atanh", (xtrig,), {}, u.Q(lax.atanh(xtrig_val), unit="rad")),
        (
            "batch_matmul",
            (
                u.Q(
                    jnp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=float),
                    unit="m",
                ),
                u.Q(
                    jnp.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=float),
                    unit="m",
                ),
            ),
            {},
            u.Q(
                [[[31.0, 34.0], [71.0, 78.0]], [[155.0, 166.0], [211.0, 226.0]]],
                unit="m2",
            ),
        ),
        ("bessel_i0e", (x_ND,), {}, u.Q(lax.bessel_i0e(x_val), unit="")),
        ("bessel_i1e", (x_ND,), {}, u.Q(lax.bessel_i1e(x_val), unit="")),
        (
            "betainc",
            (1.0, xtrig, xtrig),
            {},
            u.Q(lax.betainc(1.0, xtrig_val, xtrig_val), ""),
        ),
        (
            "bitcast_convert_type",
            (x_L, jnp.float16),
            {},
            u.Q(lax.bitcast_convert_type(x_val, jnp.float16), unit="m"),
        ),
        ("bitwise_and", (xbit, xbit), {}, lax.bitwise_and(xbit_val, xbit_val)),
        ("bitwise_not", (xbit,), {}, u.Q(lax.bitwise_not(xbit_val), unit="")),
        ("bitwise_or", (xbit, xbit), {}, u.Q(lax.bitwise_or(xbit_val, xbit_val), "")),
        ("bitwise_xor", (xbit, xbit), {}, u.Q(lax.bitwise_xor(xbit_val, xbit_val), "")),
        (
            "broadcast",
            (x_L,),
            {"sizes": (2, 2)},
            u.Q(lax.broadcast(x_val, (2, 2)), unit="m"),
        ),
        (
            "broadcast_in_dim",
            (x_L, (1, 1, 2, 2), (2, 3)),
            {},
            u.Q(lax.broadcast_in_dim(x_val, (1, 1, 2, 2), (2, 3)), unit="m"),
        ),
        (
            "broadcast_to_rank",
            (x_L,),
            {"rank": 3},
            u.Q(lax.broadcast_to_rank(x_val, rank=3), unit="m"),
        ),
        pytest.param("broadcasted_iota", (), {}, True, marks=mark_todo),
        ("cbrt", (x_L,), {}, u.Q(lax.cbrt(x_val), unit=u.unit("m**(1/3)"))),
        ("ceil", (x_L,), {}, u.Q(lax.ceil(x_val), unit="m")),
        (
            "clamp",
            (u.Q(2.0, "m"), x_L, u.Q(3.0, "m")),
            {},
            u.Q(lax.clamp(2.0, x_val, 3.0), unit="m"),
        ),
        (
            "clz",
            (x_L.astype(int),),
            {},
            u.Q(lax.clz(x_val.astype(int)), unit="m"),
        ),
        ("collapse", (x_L, 1), {}, u.Q(lax.collapse(x_val, 1), unit="m")),
        (
            "concatenate",
            ((x_L, y_L), 0),
            {},
            u.Q(jnp.concatenate((x_val, y_val), 0), unit="m"),
        ),
        ("conj", (xcomplex,), {}, u.Q(lax.conj(xcomplex_val), unit="m")),
        pytest.param(
            "conv",
            (
                u.Q(jnp.arange(1, 17, dtype=float).reshape((1, 1, 4, 4)), unit="m"),
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
            u.Q(lax.convert_element_type(x_val, jnp.int16), unit="m"),
        ),
        pytest.param(
            "conv_general_dilated",
            (
                u.Q(jnp.arange(1, 17, dtype=float).reshape((1, 1, 4, 4)), unit="m"),
                conv_kernel,
            ),
            {"window_strides": (1, 1), "padding": "SAME"},
            None,
            marks=mark_todo,
        ),
        pytest.param("conv_general_dilated_local", (), {}, True, marks=mark_todo),
        pytest.param(
            "conv_general_dilated_patches",
            (u.Q(jnp.arange(1, 17, dtype=float).reshape((1, 1, 4, 4)), unit="m"),),
            {"filter_shape": (2, 2), "window_strides": (1, 1), "padding": "VALID"},
            True,
            marks=mark_todo,
        ),
        pytest.param(
            "conv_transpose",
            (
                u.Q(jnp.arange(1, 17, dtype=float).reshape((1, 1, 4, 4)), unit="m"),
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
        ("cos", (x_ND,), {}, u.Q(lax.cos(x_val), unit="")),
        ("cosh", (x_ND,), {}, u.Q(lax.cosh(x_val), unit="")),
        ("cumlogsumexp", (x_L,), {"axis": 0}, u.Q(lax.cumlogsumexp(x_val), unit="m")),
        ("cummax", (x_L,), {"axis": 0}, u.Q(lax.cummax(x_val), unit="m")),
        ("cummin", (x_L,), {"axis": 0}, u.Q(lax.cummin(x_val), unit="m")),
        ("cumprod", (x_ND,), {"axis": 0}, u.Q(lax.cumprod(x_val), unit="")),
        ("cumsum", (x_L,), {"axis": 0}, u.Q(lax.cumsum(x_val), unit="m")),
        ("digamma", (x_ND,), {}, u.Q(lax.digamma(x_val), unit="")),
        ("div", (x_L, y_L), {}, u.Q(lax.div(x_val, y_val), unit="")),
        ("dot", (x_L, y_L), {}, u.Q(lax.dot(x_val, y_val), unit="m2")),
        pytest.param("dot_general", (), {}, True, marks=mark_todo),
        pytest.param("dynamic_index_in_dim", (), {}, True, marks=mark_todo),
        (
            "dynamic_slice",
            (x_L, (0, 0), (2, 2)),
            {},
            u.Q(lax.dynamic_slice(x_val, (0, 0), (2, 2)), unit="m"),
        ),
        pytest.param("dynamic_slice_in_dim", (), {}, True, marks=mark_todo),
        pytest.param("dynamic_update_index_in_dim", (), {}, True, marks=mark_todo),
        (
            "dynamic_update_slice",
            (x_L, y_L, (0, 0)),
            {},
            u.Q(lax.dynamic_update_slice(x_val, y_val, (0, 0)), unit="m"),
        ),
        (
            "dynamic_update_slice_in_dim",
            (x_L, y_L, 0, 0),
            {},
            u.Q(lax.dynamic_update_slice_in_dim(x_val, y_val, 0, 0), unit="m"),
        ),
        ("eq", (x_L, x_L), {}, lax.eq(x_val, x_val)),
        ("erf", (xtrig,), {}, u.Q(lax.erf(xtrig_val), unit="")),
        ("erfc", (xtrig,), {}, u.Q(lax.erfc(xtrig_val), unit="")),
        ("erf_inv", (xtrig,), {}, u.Q(lax.erf_inv(xtrig_val), unit="")),
        ("exp", (x_ND,), {}, u.Q(lax.exp(x_val), unit="")),
        ("exp2", (x_ND,), {}, u.Q(lax.exp2(x_val), unit="")),
        (
            "expand_dims",
            (x_L, (0,)),
            {},
            u.Q(lax.expand_dims(x_val, (0,)), unit="m"),
        ),
        ("expm1", (x_ND,), {}, u.Q(lax.expm1(x_val), unit="")),
        (
            "fft",
            (x_L,),
            {"fft_type": "fft", "fft_lengths": (2, 2)},
            u.Q(lax.fft(x_val, fft_type="fft", fft_lengths=(2, 2)), unit="m-1"),
        ),
        ("floor", (xround,), {}, u.Q(lax.floor(xround_val), "m")),
        ("full_like", (x_L, u.Q(1.0, "m")), {}, u.Q(lax.full_like(x_val, 1.0), "m")),
        pytest.param("gather", (), {}, True, marks=mark_todo),
        ("ge", (x_L, y_L), {}, lax.ge(x_val, y_val)),
        ("gt", (x_L, y_L), {}, lax.gt(x_val, y_val)),
        ("igamma", (1.0, xtrig), {}, u.Q(lax.igamma(1.0, xtrig_val), unit="")),
        ("igammac", (1.0, xtrig), {}, u.Q(lax.igammac(1.0, xtrig_val), unit="")),
        ("imag", (xcomplex,), {}, u.Q(lax.imag(xcomplex_val), unit="m")),
        ("index_in_dim", (x_L, 0, 0), {}, u.Q(lax.index_in_dim(x_val, 0, 0), unit="m")),
        pytest.param("index_take", (), {}, True, marks=mark_todo),
        ("integer_pow", (x_L, 2), {}, u.Q(lax.integer_pow(x_val, 2), unit="m2")),
        pytest.param("iota", (), {}, True, marks=mark_todo),
        ("is_finite", (x_L,), {}, lax.is_finite(x_val)),
        ("le", (x_L, y_L), {}, lax.le(x_val, y_val)),
        ("lgamma", (x_ND,), {}, u.Q(lax.lgamma(x_val), unit="")),
        ("log", (x_ND,), {}, u.Q(lax.log(x_val), unit="")),
        ("log1p", (x_ND,), {}, u.Q(lax.log1p(x_val), unit="")),
        ("logistic", (x_ND,), {}, u.Q(lax.logistic(x_val), unit="")),
        ("lt", (x_L, y_L), {}, lax.lt(x_val, y_val)),
        ("max", (x_L, y_L), {}, u.Q(lax.max(x_val, y_val), unit="m")),
        ("min", (x_L, y_L), {}, u.Q(lax.min(x_val, y_val), unit="m")),
        ("mul", (x_L, y_L), {}, u.Q(lax.mul(x_val, y_val), unit="m2")),
        ("ne", (x_L, y_L), {}, lax.ne(x_val, y_val)),
        ("neg", (x_L,), {}, u.Q(lax.neg(x_val), unit="m")),
        ("nextafter", (x_L, y_L), {}, u.Q(lax.nextafter(x_val, y_val), unit="m")),
        pytest.param("pad", (), {}, True, marks=mark_todo),
        ("polygamma", (1.0, xtrig), {}, u.Q(lax.polygamma(1.0, xtrig_val), unit="")),
        ("population_count", (xbit,), {}, u.Q(lax.population_count(xbit_val), unit="")),
        ("pow", (x_L, 2), {}, u.Q(lax.pow(x_val, 2), unit="m2")),
        pytest.param("random_gamma_grad", (1.0, x_ND), {}, True, marks=mark_todo),
        ("real", (xcomplex,), {}, u.Q(lax.real(xcomplex_val), unit="m")),
        ("reciprocal", (x_L,), {}, u.Q(lax.reciprocal(x_val), unit="m-1")),
        pytest.param("reduce", (), {}, True, marks=mark_todo),
        pytest.param("reduce_precision", (), {}, True, marks=mark_todo),
        pytest.param("reduce_window", (), {}, True, marks=mark_todo),
        ("rem", (x_L, y_L), {}, u.Q(lax.rem(x_val, y_val), unit="m")),
        (
            "reshape",
            (x_L, (1, 4)),
            {},
            u.Q(lax.reshape(x_val, (1, 4)), unit="m"),
        ),
        (
            "rev",
            (x_L,),
            {"dimensions": (0,)},
            u.Q(lax.rev(x_val, (0,)), unit="m"),
        ),
        pytest.param("rng_bit_generator", (), {}, True, marks=mark_todo),
        ("round", (xround,), {}, u.Q(lax.round(xround_val), unit="m")),
        ("rsqrt", (x_L**2,), {}, u.Q(lax.rsqrt(x_val**2), unit="m-1")),
        pytest.param("scatter", (), {}, True, marks=mark_todo),
        pytest.param("scatter_apply", (), {}, True, marks=mark_todo),
        pytest.param("scatter_max", (), {}, True, marks=mark_todo),
        pytest.param("scatter_min", (), {}, True, marks=mark_todo),
        pytest.param("scatter_mul", (), {}, True, marks=mark_todo),
        (
            "shift_left",
            (xbit, xbit),
            {},
            u.Q(lax.shift_left(xbit_val, xbit_val), unit=""),
        ),
        pytest.param("shift_right_arithmetic", (xbit, 1), {}, True, marks=mark_todo),
        pytest.param("shift_right_logical", (xbit, 1), {}, True, marks=mark_todo),
        ("sign", (x_L,), {}, lax.sign(x_val)),
        ("sin", (x_ND,), {}, u.Q(lax.sin(x_val), unit="")),
        ("sinh", (x_ND,), {}, u.Q(lax.sinh(x_val), unit="")),
        (
            "slice",
            (x_L, (0, 0), (2, 2)),
            {},
            u.Q(lax.slice(x_val, (0, 0), (2, 2)), unit="m"),
        ),
        (
            "slice_in_dim",
            (x_L, 0, 1, 2),
            {},
            u.Q(lax.slice_in_dim(x_val, 0, 1, 2), unit="m"),
        ),
        ("sort", (x_L,), {}, u.Q(lax.sort(x_val), "m")),
        pytest.param("sort_key_val", (), {}, True, marks=mark_todo),
        ("sqrt", (x_L**2,), {}, u.Q(lax.sqrt(x_val**2), "m")),
        ("square", (x_L,), {}, u.Q(lax.square(x_val), "m2")),
        ("sub", (x_L, y_L), {}, u.Q(lax.sub(x_val, y_val), "m")),
        ("tan", (x_ND,), {}, u.Q(lax.tan(x_val), "")),
        ("tanh", (x_ND,), {}, u.Q(lax.tanh(x_val), "")),
        ("top_k", (x_L, 1), {}, [u.Q([[2.0], [4.0]], "m"), u.Q([[1.0], [1.0]], "m")]),
        (
            "transpose",
            (x_L, (1, 0)),
            {},
            u.Q(lax.transpose(x_val, (1, 0)), unit="m"),
        ),
        ("zeta", (x_L, 2.0), {}, u.Q(lax.zeta(x_val, 2.0), unit="m")),
        pytest.param("associative_scan", (), {}, True, marks=mark_todo),
        pytest.param("fori_loop", (), {}, True, marks=mark_todo),
        pytest.param("scan", (), {}, True, marks=mark_todo),
        (
            "select",
            (jnp.array([[True, False], [True, False]], dtype=bool), x_L, y_L),
            {},
            u.Q(
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
            u.Q(
                lax.while_loop(lambda x: jnp.all(x < 10), lambda x: x + 1, x_val),
                unit="",
            ),
        ),
        ("stop_gradient", (x_L,), {}, u.Q(lax.stop_gradient(x_val), unit="m")),
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
