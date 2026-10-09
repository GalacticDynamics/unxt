"""Tests for `quax` registrations that the array-API suites do not reach."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import quax
from astropy.units import UnitConversionError
from jax import lax

import quaxed.numpy as qnp

import unxt as u
from unxt._src.quantity.register_primitives import cond_p_q
from unxt.quantity import AllowValue


def test_cond_on_a_quantity_operand():
    """``lax.cond`` with an array predicate strips the quantity operand."""
    f = quax.quaxify(lambda p, x: lax.cond(p, lambda v: v, lambda v: 2 * v, x))
    assert np.isclose(np.asarray(f(jnp.asarray(1, dtype=bool), u.Q(1.0, "m"))), 1.0)


def test_cond_on_a_quantity_predicate_is_unsupported():
    """A quantity *predicate* has no meaning and is refused."""
    q = u.Q(1.0, "m")
    with pytest.raises(NotImplementedError):
        cond_p_q(q, q)


def test_angle_divided_by_angle_is_dimensionless():
    """``Angle / Angle`` degrades to a plain, dimensionless `Quantity`."""
    a = u.Angle(1.0, "deg")
    got = qnp.divide(a, a)
    assert isinstance(got, u.quantity.Quantity)
    assert got.unit == u.unit("")
    assert np.isclose(np.asarray(got.value), 1.0)


_ANGLE = u.Angle([0.0, 1.0], "rad")


@pytest.mark.parametrize(
    "op",
    [
        qnp.isfinite,
        qnp.any,
        qnp.cosh,
        qnp.sinh,
        qnp.tanh,
        lambda a: qnp.linalg.qr(qnp.stack([a, a[::-1]]))[0],
    ],
    ids=["isfinite", "any", "cosh", "sinh", "tanh", "qr-Q"],
)
def test_angle_dimensionless_result_degrades_to_quantity(op):
    """A dimensionless result from an `Angle` degrades to a plain `Quantity`."""
    got = op(_ANGLE)
    assert type(got) is u.quantity.Quantity
    assert got.unit == u.unit("")


def test_angle_allclose():
    """``allclose`` on `Angle` works (GalacticDynamics/unxt#945)."""
    assert qnp.allclose(_ANGLE, _ANGLE, atol=u.Q(1e-8, "rad"))


def test_isfinite_staticquantity_under_jit():
    """Under ``jit`` a `StaticQuantity` result degrades to a plain `Quantity`."""
    got = jax.jit(qnp.isfinite)(u.StaticQuantity([1.0], "m"))
    assert type(got) is u.quantity.Quantity
    assert bool(got.value.all())


def test_scatter_add_quantity_operand_and_updates():
    """``scatter_add`` with quantity operand *and* updates keeps the unit."""
    dnums = lax.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )
    got = quax.quaxify(lax.scatter_add)(
        u.Q(jnp.ones(4), "m"), jnp.asarray([[1]]), u.Q(jnp.asarray([9.0]), "m"), dnums
    )
    assert got.unit == u.unit("m")
    assert np.allclose(np.asarray(got.value), [1.0, 10.0, 1.0, 1.0])


@pytest.mark.parametrize(
    "shift", [lax.shift_left, lax.shift_right_arithmetic, lax.shift_right_logical]
)
@pytest.mark.parametrize(
    "amount",
    [jnp.array([1, 2], jnp.int32), u.Q(jnp.array([1, 2], jnp.int32), ""), 1],
    ids=["array", "quantity", "int"],
)
def test_shift_dimensionless_quantity(shift, amount):
    """Shifting a dimensionless int quantity by any amount works (#952)."""
    x = jnp.array([4, 8], jnp.int32)

    got = quax.quaxify(shift)(u.Q(x, ""), amount)

    assert got.unit == u.unit("")
    assert np.array_equal(
        np.asarray(got.value), shift(x, u.ustrip(AllowValue, "", amount))
    )


@pytest.mark.parametrize(
    ("select", "want"),
    [
        (
            lambda q, *_: lax.select_n(
                jnp.array([0, 2]), q, jnp.ones(2), jnp.full(2, 3.0)
            ),
            [0.5, 3.0],
        ),
        (
            lambda q, *_: lax.select_n(
                jnp.array([1, 3]), jnp.ones(2), q, q, jnp.full(2, 3.0)
            ),
            [0.5, 3.0],
        ),
        (
            lambda q, lo, hi: jnp.select([q < lo, q < hi], [q, jnp.ones(2)]),
            [0.5, 1.0],
        ),
    ],
    ids=["select_n-3", "select_n-4", "jnp.select"],
)
def test_select_n_mixed_quantity_and_arrays(select, want):
    """>2 cases mixing quantities and raw arrays select like the 2-case rules (#954).

    As there, a raw case is taken to be in the quantity's unit.
    """
    q = u.Q(jnp.array([0.5, 2.0]), "km")

    got = quax.quaxify(select)(q, u.Q(1.0, "km"), u.Q(3.0, "km"))

    assert got.unit == u.unit("km")
    assert np.array_equal(np.asarray(got.value), want)


_TABLE = jnp.array([10.0, 20.0, 30.0])


def test_gather_array_with_quantity_indices():
    """A plain array indexed by a dimensionless quantity gives a plain array (#953)."""
    idx = u.Q(jnp.array([1, 2], jnp.int32), "")

    got = quax.quaxify(lambda a, i: a[i])(_TABLE, idx)

    assert not isinstance(got, u.quantity.AbstractQuantity)
    assert np.array_equal(np.asarray(got), [20.0, 30.0])


def test_gather_quantity_with_quantity_indices():
    """A quantity indexed by a dimensionless quantity keeps its unit."""
    idx = u.Q(jnp.array([1, 2], jnp.int32), "")

    got = quax.quaxify(lambda a, i: a[i])(u.Q(_TABLE, "m"), idx)

    assert got.unit == u.unit("m")
    assert np.array_equal(np.asarray(got.value), [20.0, 30.0])


@pytest.mark.parametrize(
    "operand", [_TABLE, u.Q(_TABLE, "m")], ids=["array", "quantity"]
)
def test_gather_with_scaled_dimensionless_indices_raises(operand):
    """A ``%`` int index converts to a float in true units, so it is rejected.

    It must never be read in its own unit, which would silently select element
    100 for ``Q(100, "%")``.
    """
    idx = u.Q(jnp.array([100], jnp.int32), "%")
    with pytest.raises(TypeError):
        quax.quaxify(lambda a, i: a[i])(operand, idx)


def test_gather_with_dimensionful_indices_raises():
    """An index with a dimension has no meaning."""
    idx = u.Q(jnp.array([1, 2], jnp.int32), "m")
    with pytest.raises(UnitConversionError):
        quax.quaxify(lambda a, i: a[i])(_TABLE, idx)


def test_bitcast_dimensionless_uses_true_value():
    """Equal dimensionless values bitcast to equal bits, with unit '' (#958)."""
    bitcast = quax.quaxify(lambda q: lax.bitcast_convert_type(q, jnp.int32))

    got = bitcast(u.Q(jnp.array([50.0]), "%"))

    assert got.unit == u.unit("")
    assert np.array_equal(
        np.asarray(got.value), lax.bitcast_convert_type(jnp.array([0.5]), jnp.int32)
    )


_NP_AXES = (np.int64(1),)


@pytest.mark.parametrize(
    ("bind", "want"),
    [
        (
            lambda q: lax.argmax_p.bind(q, axes=_NP_AXES, index_dtype=jnp.int32),
            [1, 0],
        ),
        (
            lambda q: lax.argmin_p.bind(q, axes=_NP_AXES, index_dtype=jnp.int32),
            [0, 1],
        ),
        (lambda q: lax.reduce_prod_p.bind(q, axes=_NP_AXES), [1.0, 3.0]),
    ],
    ids=["argmax", "argmin", "reduce_prod"],
)
def test_numpy_integer_axes(bind, want):
    """JAX can bind axis params as NumPy integers; the rules accept them (#956)."""
    q = u.Q(jnp.array([[0.5, 2.0], [3.0, 1.0]]), "")

    got = quax.quaxify(bind)(q)

    assert np.allclose(np.asarray(u.ustrip(AllowValue, "", got)), want)


@pytest.mark.skipif(not hasattr(lax, "stack_p"), reason="`stack_p` is JAX >= 0.10.1")
@pytest.mark.parametrize(
    ("quantity_first", "unit"),
    # A raw array may only be stacked with a dimensionless quantity.
    [(True, "m"), (False, "")],
    ids=["qq", "vq"],
)
def test_stack_numpy_integer_axis(quantity_first, unit):
    """``stack_p`` accepts a NumPy-integer ``axis`` in both overloads (#956)."""
    a, b = jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])
    first = u.Q(a, unit) if quantity_first else a
    stack = quax.quaxify(lambda x, y: lax.stack_p.bind(x, y, axis=np.int64(0)))

    got = stack(first, u.Q(b, unit))

    assert got.unit == u.unit(unit)
    assert np.array_equal(np.asarray(got.value), [[1.0, 2.0], [3.0, 4.0]])


@pytest.mark.parametrize(
    "op",
    [
        jnp.less,
        jnp.less_equal,
        jnp.greater,
        jnp.greater_equal,
        jnp.equal,
        jnp.not_equal,
    ],
)
@pytest.mark.parametrize("quantity_first", [True, False], ids=["qv", "vq"])
def test_compare_scaled_dimensionless_with_raw(op, quantity_first):
    """A raw operand compares to the quantity's true dimensionless value (#965)."""
    q = u.Q(jnp.array([50.0, 50.0, 50.0]), "%")  # == 0.5
    raw = jnp.array([0.25, 0.5, 4.65])
    f = (lambda q: op(q, raw)) if quantity_first else (lambda q: op(raw, q))
    want = op(jnp.full(3, 0.5), raw) if quantity_first else op(raw, jnp.full(3, 0.5))

    got = quax.quaxify(f)(q)

    assert got.unit == u.unit("")
    assert np.array_equal(np.asarray(got.value), want)


def test_signbit_copysign_on_dimensionful_quantity():
    """Sign bit tricks ignore the unit; copysign keeps ``x``'s unit (#973)."""
    q = u.Q(jnp.array([1.5, -2.0]), "m")

    sb = quax.quaxify(jnp.signbit)(q)
    assert sb.unit == u.unit("")
    assert np.array_equal(np.asarray(sb.value), [False, True])

    cs = quax.quaxify(jnp.copysign)(q, u.Q(jnp.array([-1.0, 1.0]), "s"))
    assert cs.unit == u.unit("m")
    assert np.array_equal(np.asarray(cs.value), [-1.5, 2.0])


def test_dimensionful_integer_bits_to_bool_and_logical_shift():
    """Bool cast of dimensionful int bits is a plain array; also for static (#973)."""
    bits = u.Q(jnp.array([1, 0]), "m")
    got = quax.quaxify(lambda q: lax.convert_element_type(q, jnp.bool_))(bits)
    assert got.unit == u.unit("")
    assert np.array_equal(np.asarray(got.value), [True, False])

    static = u.StaticQuantity(np.array([1, 0]), "m")
    got = quax.quaxify(lambda q: lax.convert_element_type(q, jnp.bool_))(static)
    assert got.unit == u.unit("")
    assert np.array_equal(np.asarray(got.value), [True, False])

    shifted = quax.quaxify(lambda q: lax.shift_right_logical(q, 1))(
        u.Q(jnp.array([4, 8]), "m")
    )
    assert shifted.unit == u.unit("m")
    assert np.array_equal(np.asarray(shifted.value), [2, 4])


_RAW_CMP = {
    "Q == x": qnp.equal,
    "x == Q": lambda q, x: qnp.equal(x, q),
    "Q != x": qnp.not_equal,
    "x != Q": lambda q, x: qnp.not_equal(x, q),
    "Q < x": qnp.less,
    "x < Q": lambda q, x: qnp.less(x, q),
    "Q <= x": qnp.less_equal,
    "x <= Q": lambda q, x: qnp.less_equal(x, q),
    "Q > x": qnp.greater,
    "x > Q": lambda q, x: qnp.greater(x, q),
    "Q >= x": qnp.greater_equal,
    "x >= Q": lambda q, x: qnp.greater_equal(x, q),
}


@pytest.mark.parametrize("form", list(_RAW_CMP))
@pytest.mark.parametrize("raw", [jnp.inf, -jnp.inf, 0.0], ids=["inf", "-inf", "0"])
def test_compare_dimensionful_with_unit_independent_raw(form, raw):
    """Every comparison accepts 0 and ±inf against a dimensionful quantity.

    Both are unit-independent, so all forms must agree with plain arrays
    (#978 for ``==``/``!=``; the ordering comparisons likewise).
    """
    v = jnp.array([1.0, raw])
    raw = jnp.asarray(raw)
    want = _RAW_CMP[form](v, raw)

    got = _RAW_CMP[form](u.Q(v, "m"), raw)

    assert np.array_equal(np.asarray(got.value), np.asarray(want))


@pytest.mark.parametrize("form", list(_RAW_CMP))
def test_compare_dimensionful_with_mixed_zero_and_inf_raw(form):
    """A raw array mixing 0 and ±inf is accepted: each element is unit-independent."""
    v = jnp.array([0.0, jnp.inf, -jnp.inf, 1.0])
    raw = jnp.array([0.0, jnp.inf, -jnp.inf, jnp.inf])

    got = _RAW_CMP[form](u.Q(v, "m"), raw)

    assert np.array_equal(np.asarray(got.value), np.asarray(_RAW_CMP[form](v, raw)))


@pytest.mark.parametrize("form", list(_RAW_CMP))
def test_compare_dimensionful_with_finite_raw_raises(form):
    """A nonzero finite raw number depends on the unit, so it is still refused."""
    with pytest.raises(eqx.EquinoxRuntimeError, match="Cannot compare"):
        _RAW_CMP[form](u.Q(jnp.array([1.0, 2.0]), "m"), jnp.asarray(2.0))


_FLOATS = jnp.array([1.5, -2.0, 1000.0])


@pytest.mark.parametrize("unit", ["m", ""])
def test_spacing_keeps_unit(unit):
    """``spacing`` (``nextafter`` toward inf) works and keeps the unit (#973)."""
    got = quax.quaxify(jnp.spacing)(u.Q(_FLOATS, unit))

    assert got.unit == u.unit(unit)
    assert np.array_equal(np.asarray(got.value), np.asarray(jnp.spacing(_FLOATS)))


@pytest.mark.parametrize("target", [jnp.inf, -jnp.inf], ids=["inf", "-inf"])
def test_nextafter_dimensionful_toward_raw_inf(target):
    """Stepping toward ±inf is unit-independent, so it is allowed (#973)."""
    got = quax.quaxify(lambda q: lax.nextafter(q, jnp.full(3, target)))(
        u.Q(_FLOATS, "m")
    )

    assert got.unit == u.unit("m")
    assert np.array_equal(
        np.asarray(got.value), np.asarray(lax.nextafter(_FLOATS, jnp.full(3, target)))
    )


def test_nextafter_scaled_dimensionless_toward_raw():
    """A raw target is dimensionless: 1.0 is 100% in the quantity's own unit."""
    x = jnp.array([50.0])

    got = quax.quaxify(lambda q: lax.nextafter(q, jnp.array([1.0])))(u.Q(x, "%"))

    assert got.unit == u.unit("%")
    assert np.array_equal(
        np.asarray(got.value), np.asarray(lax.nextafter(x, jnp.array([100.0])))
    )


def test_nextafter_dimensionful_toward_finite_raw_raises():
    """A finite raw target's direction depends on the unit, so it is refused."""
    with pytest.raises(eqx.EquinoxRuntimeError, match="nextafter"):
        quax.quaxify(lambda q: lax.nextafter(q, jnp.full(3, 2.0)))(u.Q(_FLOATS, "m"))
