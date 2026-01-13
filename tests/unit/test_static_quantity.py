"""Tests for StaticQuantity."""

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import wadler_lindig as wl
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays as np_arrays
from plum import promote

import unxt as u


def test_static_quantity_accepts_numpy() -> None:
    """StaticQuantity accepts Python scalars and NumPy arrays."""
    scalar = u.StaticQuantity(1, "m")
    assert np.array_equal(np.asarray(scalar.value), np.asarray(1))

    arr = np.array([1.0, 2.0])
    vec = u.StaticQuantity(arr, "m")
    assert np.array_equal(np.asarray(vec.value), arr)


def test_static_quantity_rejects_jax_array() -> None:
    """StaticQuantity rejects JAX arrays."""
    with pytest.raises(TypeError, match="StaticQuantity does not accept JAX arrays"):
        u.StaticQuantity(jnp.array([1.0, 2.0]), "m")


def test_static_quantity_accepts_array_like() -> None:
    """StaticQuantity accepts array-like inputs."""
    q = u.StaticQuantity([1.0, 2.0], "m")
    assert np.array_equal(np.asarray(q.value), np.array([1.0, 2.0]))


def test_static_quantity_hashable_python() -> None:
    """StaticQuantity can be hashed in Python."""
    q1 = u.StaticQuantity(1, "m")
    q2 = u.StaticQuantity(1, "m")
    assert hash(q1) == hash(q2)

    q_arr1 = u.StaticQuantity(np.array([1.0, 2.0]), "m")
    q_arr2 = u.StaticQuantity(np.array([1.0, 2.0]), "m")
    assert hash(q_arr1) == hash(q_arr2)


def test_static_quantity_hashable_jax() -> None:
    """StaticQuantity can be used as a static arg in jitted code."""
    q = u.StaticQuantity(np.array([1.0, 2.0]), "m")

    @partial(jax.jit, static_argnames=("q",))
    def add(x, q):
        return x + jnp.asarray(q.value)

    out = add(1.0, q)
    assert np.allclose(out, np.array([2.0, 3.0]))

    out_again = add(1.0, u.StaticQuantity(np.array([1.0, 2.0]), "m"))
    assert np.allclose(out_again, np.array([2.0, 3.0]))


def test_static_quantity_vmap_static_arg() -> None:
    """StaticQuantity can be used as a static arg in vmapped code."""
    q = u.StaticQuantity(np.array([1.0, 2.0]), "m")

    def add(x, q):
        return x + jnp.asarray(q.value)

    out = jax.vmap(add, in_axes=(0, None))(jnp.array([1.0, 2.0]), q)
    assert np.allclose(out, np.array([[2.0, 3.0], [3.0, 4.0]]))


def test_static_quantity_promotes_to_quantity() -> None:
    """StaticQuantity promotes to Quantity."""
    q = u.StaticQuantity(np.array([1.0, 2.0]), "m")
    r = u.Q(jnp.array([3.0, 4.0]), "m")

    q_promoted, r_promoted = promote(q, r)
    assert isinstance(q_promoted, u.Q)
    assert isinstance(r_promoted, u.Q)


def test_static_quantity_subtraction_with_quantity() -> None:
    """StaticQuantity subtraction with Quantity promotes to Quantity."""
    sq = u.StaticQuantity(np.array(1.0), "s")
    q = u.Quantity(0.5, "s")

    # Quantity - StaticQuantity -> Quantity
    result1 = q - sq
    assert isinstance(result1, u.Q)
    assert np.allclose(np.asarray(result1.value), -0.5)

    # StaticQuantity - Quantity -> Quantity
    result2 = sq - q
    assert isinstance(result2, u.Q)
    assert np.allclose(np.asarray(result2.value), 0.5)


def test_static_quantity_subtraction_preserves_static() -> None:
    """StaticQuantity subtraction with StaticQuantity stays StaticQuantity."""
    sq1 = u.StaticQuantity(np.array(1.0), "km")
    sq2 = u.StaticQuantity(np.array(500.0), "m")

    result = sq1 - sq2
    assert isinstance(result, u.StaticQuantity)
    assert np.allclose(np.asarray(result.value), 0.5)
    assert result.unit == sq1.unit


def test_static_quantity_addition_with_quantity() -> None:
    """StaticQuantity addition with Quantity promotes to Quantity."""
    sq = u.StaticQuantity(np.array(1.0), "s")
    q = u.Quantity(0.5, "s")

    # Quantity + StaticQuantity -> Quantity
    result1 = q + sq
    assert isinstance(result1, u.Q)
    assert np.allclose(np.asarray(result1.value), 1.5)

    # StaticQuantity + Quantity -> Quantity
    result2 = sq + q
    assert isinstance(result2, u.Q)
    assert np.allclose(np.asarray(result2.value), 1.5)


def test_static_quantity_addition_preserves_static() -> None:
    """StaticQuantity addition with StaticQuantity stays StaticQuantity."""
    sq1 = u.StaticQuantity(np.array(1.0), "km")
    sq2 = u.StaticQuantity(np.array(500.0), "m")

    result = sq1 + sq2
    assert isinstance(result, u.StaticQuantity)
    assert np.allclose(np.asarray(result.value), 1.5)
    assert result.unit == sq1.unit


def test_static_quantity_multiplication_with_quantity() -> None:
    """StaticQuantity multiplication with Quantity promotes to Quantity."""
    sq = u.StaticQuantity(2.0, "m")
    q = u.Quantity(3.0, "s")

    # Quantity * StaticQuantity -> Quantity
    result1 = q * sq
    assert isinstance(result1, u.Q)
    assert np.allclose(result1.value, 6.0)
    assert result1.unit == u.unit("m * s")

    # StaticQuantity * Quantity -> Quantity
    result2 = sq * q
    assert isinstance(result2, u.Q)
    assert np.allclose(result2.value, 6.0)
    assert result2.unit == u.unit("m * s")


def test_static_quantity_multiplication_preserves_static() -> None:
    """StaticQuantity multiplication with StaticQuantity stays StaticQuantity."""
    sq1 = u.StaticQuantity(2.0, "m")
    sq2 = u.StaticQuantity(3.0, "s")

    result = sq1 * sq2
    assert isinstance(result, u.StaticQuantity)
    assert np.allclose(result.value, 6.0)
    assert result.unit == u.unit("m * s")


def test_static_quantity_division_with_quantity() -> None:
    """StaticQuantity division with Quantity promotes to Quantity."""
    sq = u.StaticQuantity(6.0, "m")
    q = u.Quantity(2.0, "s")

    # Quantity / StaticQuantity -> Quantity
    result1 = q / sq
    assert isinstance(result1, u.Q)
    assert np.allclose(result1.value, 1.0 / 3.0)
    assert result1.unit == u.unit("s / m")

    # StaticQuantity / Quantity -> Quantity
    result2 = sq / q
    assert isinstance(result2, u.Q)
    assert np.allclose(result2.value, 3.0)
    assert result2.unit == u.unit("m / s")


def test_static_quantity_division_preserves_static() -> None:
    """StaticQuantity division with StaticQuantity stays StaticQuantity."""
    sq1 = u.StaticQuantity(6.0, "m")
    sq2 = u.StaticQuantity(2.0, "s")

    result = sq1 / sq2
    assert isinstance(result, u.StaticQuantity)
    assert np.allclose(result.value, 3.0)
    assert result.unit == u.unit("m / s")


def test_static_quantity_division_integer_inputs() -> None:
    """StaticQuantity division with integer inputs promotes to float."""
    sq1 = u.StaticQuantity(6, "m")
    sq2 = u.StaticQuantity(2, "s")

    result = sq1 / sq2
    assert isinstance(result, u.StaticQuantity)
    # Integer division should promote to float (true division semantics)
    assert result.value.dtype == np.float32
    assert np.allclose(result.value, 3.0)
    assert result.unit == u.unit("m / s")


def test_static_quantity_modulo_with_quantity() -> None:
    """StaticQuantity modulo with Quantity promotes to Quantity."""
    sq = u.StaticQuantity(7.0, "m")
    q = u.Quantity(3.0, "m")

    # Quantity % StaticQuantity -> Quantity (via promotion)
    result1 = q % sq
    assert isinstance(result1, u.Q)
    assert np.allclose(result1.value, 3.0 % 7.0)
    assert result1.unit == u.unit("m")

    # StaticQuantity % Quantity -> Quantity (via promotion)
    result2 = sq % q
    assert isinstance(result2, u.Q)
    assert np.allclose(result2.value, 7.0 % 3.0)
    assert result2.unit == u.unit("m")


def test_static_quantity_modulo_with_static_quantity_promotes() -> None:
    """StaticQuantity modulo StaticQuantity promotes to Quantity."""
    sq1 = u.StaticQuantity(7.0, "m")
    sq2 = u.StaticQuantity(3.0, "m")

    result = sq1 % sq2
    # Since there's no modulo_p primitive dispatch, plum promotion rules apply
    # StaticQuantity % StaticQuantity -> Quantity (via default promotion)
    assert isinstance(result, u.Q)
    assert np.allclose(result.value, 7.0 % 3.0)
    assert result.unit == u.unit("m")


def test_static_quantity_modulo_preserves_numpy_input_values() -> None:
    """StaticQuantity stores numpy arrays that persist through operations."""
    # Verify that StaticQuantity with numpy arrays stores them correctly
    sq = u.StaticQuantity(np.array([7.0, 8.0, 9.0]), "m")

    # The underlying value should be a StaticValue wrapping a numpy array
    assert isinstance(sq.value, u.quantity.StaticValue)
    # Verify the array inside is numpy, not JAX
    underlying_array = sq.value.array
    assert isinstance(underlying_array, np.ndarray)
    assert not isinstance(underlying_array, jnp.ndarray)
    assert np.array_equal(underlying_array, np.array([7.0, 8.0, 9.0]))


def test_static_value_modulo_operations() -> None:
    """StaticValue modulo operations work correctly with forward and reverse."""
    sv = u.quantity.StaticValue(np.array([7.0, 8.0]))

    # Forward modulo
    result_forward = sv % 3.0
    assert np.allclose(result_forward, np.array([1.0, 2.0]))

    # Reverse modulo
    result_reverse = 10.0 % sv
    assert np.allclose(result_reverse, np.array([3.0, 2.0]))
    # With another StaticValue
    sv2 = u.quantity.StaticValue(np.array([3.0, 3.0]))
    result_sv = sv % sv2
    assert isinstance(result_sv, u.quantity.StaticValue)
    assert np.allclose(result_sv.array, np.array([1.0, 2.0]))


def test_static_value_pdoc() -> None:
    """StaticValue uses the contained value for formatting."""
    value = u.quantity.StaticValue(np.array([1.0, 2.0]))
    assert wl.pformat(value, show_wrapper=False) == wl.pformat(np.array([1.0, 2.0]))


def test_static_quantity_pdoc_hides_static_value() -> None:
    """StaticQuantity formatting should not expose StaticValue."""
    q = u.StaticQuantity(np.array([1.0, 2.0]), "m")
    formatted = wl.pformat(q, short_arrays=False)
    assert "StaticValue" not in formatted
    assert "unit='m'" in formatted


def test_quantity_static_value_jit_grad() -> None:
    """StaticValue should behave as static in JAX transforms."""
    sv = u.quantity.StaticValue(np.array([1.0, 2.0]))
    q = u.Q(sv, "m")

    def f(x, q):
        return jnp.sum(jnp.asarray(q.value) * x)

    jit_f = eqx.filter_jit(f)
    out = jit_f(jnp.array([1.0, 1.0]), q)
    assert np.allclose(out, np.array(3.0))

    grad_f = eqx.filter_grad(lambda x, q: f(x, q))
    grad = grad_f(jnp.array([1.0, 1.0]), q)
    assert np.allclose(grad, np.array([1.0, 2.0]))


def test_static_value_binary_ops_degrade() -> None:
    """StaticValue should behave like its array in arithmetic."""
    sv = u.quantity.StaticValue(np.array([1.0, 2.0]))
    sv2 = u.quantity.StaticValue(np.array([3.0, 4.0]))
    assert np.allclose(np.asarray(sv + 1.0), np.array([2.0, 3.0]))
    assert np.allclose(np.asarray(1.0 + sv), np.array([2.0, 3.0]))
    assert np.allclose(np.asarray(sv - 1.0), np.array([0.0, 1.0]))
    assert np.allclose(np.asarray(3.0 - sv), np.array([2.0, 1.0]))
    assert np.allclose(np.asarray(sv * 2.0), np.array([2.0, 4.0]))
    assert np.allclose(np.asarray(2.0 * sv), np.array([2.0, 4.0]))
    assert np.allclose(np.asarray(sv / 2.0), np.array([0.5, 1.0]))
    assert np.allclose(np.asarray(4.0 / sv), np.array([4.0, 2.0]))
    assert np.allclose(np.asarray(sv % 2.0), np.array([1.0, 0.0]))

    sum_sv = sv + sv2
    assert isinstance(sum_sv, u.quantity.StaticValue)
    assert np.allclose(np.asarray(sum_sv.array), np.array([4.0, 6.0]))


def test_static_value_protocols() -> None:
    """StaticValue exposes NumPy-style protocols."""
    arr = np.array([1, 2], dtype=np.int32)
    sv = u.quantity.StaticValue(arr)

    assert np.array_equal(np.asarray(sv), arr)
    assert np.asarray(sv, dtype=np.float64).dtype == np.float64
    assert len(sv) == 2
    assert list(iter(sv)) == [1, 2]
    assert sv[0] == 1
    assert sv.shape == (2,)
    assert sv.dtype == jnp.asarray(arr).dtype
    assert repr(sv) == f"StaticValue({repr(arr)!s})"
    assert np.allclose(np.asarray(sv.sum()), np.sum(arr))


def test_static_value_eq_hash() -> None:
    """StaticValue equality and hashing follow the wrapped array."""
    sv1 = u.quantity.StaticValue(np.array([1.0, 2.0]))
    sv2 = u.quantity.StaticValue(np.array([1.0, 2.0]))
    sv3 = u.quantity.StaticValue(np.array([1.0, 3.0]))

    assert sv1 == sv2
    assert sv1 != sv3
    assert sv1 != 1
    assert sv1.__eq__(1) is NotImplemented
    assert hash(sv1) == hash(sv2)


def test_static_value_rich_comparisons() -> None:
    """StaticValue supports all rich comparison operators."""
    sv1 = u.quantity.StaticValue(np.array([1.0, 2.0, 3.0]))
    sv2 = u.quantity.StaticValue(np.array([2.0, 2.0, 2.0]))
    sv3 = u.quantity.StaticValue(np.array([1.0, 2.0, 3.0]))

    # Test with another StaticValue - eq/ne return bool
    assert (sv1 == sv3) is True
    assert (sv1 != sv2) is True
    assert (sv1 == sv2) is False

    # Ordering comparisons return arrays
    assert np.array_equal(sv1 < sv2, np.array([True, False, False]))
    assert np.array_equal(sv1 <= sv2, np.array([True, True, False]))
    assert np.array_equal(sv1 > sv2, np.array([False, False, True]))
    assert np.array_equal(sv1 >= sv2, np.array([False, True, True]))

    # Test with regular arrays
    arr = np.array([2.0, 2.0, 2.0])
    assert np.array_equal(sv1 == arr, np.array([False, True, False]))
    assert np.array_equal(sv1 != arr, np.array([True, False, True]))
    assert np.array_equal(sv1 < arr, np.array([True, False, False]))
    assert np.array_equal(sv1 <= arr, np.array([True, True, False]))
    assert np.array_equal(sv1 > arr, np.array([False, False, True]))
    assert np.array_equal(sv1 >= arr, np.array([False, True, True]))

    # Test with scalars
    assert np.array_equal(sv1 < 2.5, np.array([True, True, False]))
    assert np.array_equal(sv1 > 1.5, np.array([False, True, True]))


def test_static_value_comparison_incompatible_types() -> None:
    """StaticValue handles incompatible types correctly in comparisons."""
    sv = u.quantity.StaticValue(np.array([1.0, 2.0]))

    # __eq__ and __ne__ with incompatible types return NotImplemented
    assert sv.__eq__(object()) is NotImplemented
    assert sv.__eq__(1) is NotImplemented
    assert sv.__eq__("string") is NotImplemented
    assert sv.__eq__({"key": "value"}) is NotImplemented

    assert sv.__ne__(object()) is NotImplemented
    assert sv.__ne__(1) is NotImplemented
    assert sv.__ne__("string") is NotImplemented

    # Python's comparison fallback handles NotImplemented correctly
    # When comparing sv == 1, Python tries sv.__eq__(1), gets NotImplemented,
    # then tries (1).__eq__(sv), which returns False
    assert (sv == 1) is False
    assert (sv != 1) is True
    assert (sv == object()) is False
    assert (sv != object()) is True

    # Ordering comparisons with scalars that work with NumPy arrays
    # These delegate to NumPy's broadcasting rules
    assert np.array_equal(sv < 1.5, np.array([True, False]))
    assert np.array_equal(sv <= 2.0, np.array([True, True]))
    assert np.array_equal(sv > 1.5, np.array([False, True]))
    assert np.array_equal(sv >= 1.0, np.array([True, True]))

    # Ordering comparisons with incompatible types may raise errors
    # (this is expected NumPy behavior)
    with pytest.raises((TypeError, np.exceptions.DTypePromotionError)):
        sv < "string"  # noqa: B015

    with pytest.raises((TypeError, np.exceptions.DTypePromotionError)):
        sv <= object()  # noqa: B015


def test_static_value_from_dispatch() -> None:
    """StaticValue.from_ dispatches to expected constructors."""
    sv = u.quantity.StaticValue.from_([1.0, 2.0])
    assert isinstance(sv, u.quantity.StaticValue)
    assert np.array_equal(np.asarray(sv.array), np.array([1.0, 2.0]))

    sv2 = u.quantity.StaticValue.from_(sv)
    assert sv2 is sv

    with pytest.raises(TypeError, match="StaticQuantity does not accept JAX arrays"):
        u.quantity.StaticValue.from_(jnp.array([1.0, 2.0]))


def test_static_value_unary_ops() -> None:
    """StaticValue unary ops degrade to array values."""
    sv = u.quantity.StaticValue(np.array([-1.0, 2.0]))
    assert np.allclose(np.asarray(-sv), np.array([1.0, -2.0]))
    assert np.allclose(np.asarray(+sv), np.array([-1.0, 2.0]))
    assert np.allclose(np.asarray(abs(sv)), np.array([1.0, 2.0]))


def test_static_value_more_binary_ops() -> None:
    """StaticValue supports additional binary operators."""
    sv = u.quantity.StaticValue(np.array([3.0, 4.0]))
    assert np.allclose(np.asarray(sv // 2), np.array([1.0, 2.0]))
    assert np.allclose(np.asarray(5.0 // sv), np.array([1.0, 1.0]))
    assert np.allclose(np.asarray(sv**2), np.array([9.0, 16.0]))
    assert np.allclose(np.asarray(2.0**sv), np.array([8.0, 16.0]))
    assert np.allclose(np.asarray(5.0 % sv), np.array([2.0, 1.0]))

    sv_mat = u.quantity.StaticValue(np.array([[1.0, 2.0], [3.0, 4.0]]))
    sv_vec = u.quantity.StaticValue(np.array([1.0, 1.0]))
    out = sv_mat @ sv_vec
    assert isinstance(out, u.quantity.StaticValue)
    assert np.allclose(np.asarray(out.array), np.array([3.0, 7.0]))

    out_r = jnp.array([1.0, 1.0]) @ sv_mat
    assert np.allclose(np.asarray(out_r), np.array([4.0, 6.0]))

    rsum = sv.__radd__(u.quantity.StaticValue(np.array([1.0, 1.0])))
    assert isinstance(rsum, u.quantity.StaticValue)
    assert np.allclose(np.asarray(rsum.array), np.array([4.0, 5.0]))


def test_static_value_dunder_ops_explicit() -> None:
    """Exercise StaticValue dunder operators directly."""
    sv = u.quantity.StaticValue(np.array([3.0, 4.0]))
    assert np.allclose(np.asarray(sv.__floordiv__(2)), np.array([1.0, 2.0]))
    assert np.allclose(np.asarray(sv.__rfloordiv__(5.0)), np.array([1.0, 1.0]))
    assert np.allclose(np.asarray(sv.__pow__(2)), np.array([9.0, 16.0]))
    assert np.allclose(np.asarray(sv.__rpow__(2.0)), np.array([8.0, 16.0]))
    assert np.allclose(np.asarray(sv.__mod__(2.0)), np.array([1.0, 0.0]))
    assert np.allclose(np.asarray(sv.__rmod__(5.0)), np.array([2.0, 1.0]))

    sv_mat = u.quantity.StaticValue(np.array([[1.0, 2.0], [3.0, 4.0]]))
    sv_vec = u.quantity.StaticValue(np.array([1.0, 1.0]))
    out = sv_mat.__matmul__(sv_vec)
    assert isinstance(out, u.quantity.StaticValue)
    assert np.allclose(np.asarray(out.array), np.array([3.0, 7.0]))

    out_r = sv_mat.__rmatmul__(np.array([1.0, 1.0]))
    assert np.allclose(np.asarray(out_r), np.array([4.0, 6.0]))

    assert np.allclose(np.asarray(sv.__neg__()), np.array([-3.0, -4.0]))
    assert np.allclose(np.asarray(sv.__pos__()), np.array([3.0, 4.0]))
    assert np.allclose(np.asarray(sv.__abs__()), np.array([3.0, 4.0]))


@given(
    arr=np_arrays(
        dtype=np.float32,
        shape=st.integers(min_value=1, max_value=5),
        elements=st.floats(
            min_value=-10.0,
            max_value=10.0,
            allow_nan=False,
            allow_infinity=False,
            allow_subnormal=False,
            width=32,
        ),
    )
)
def test_static_value_ops_property(arr: np.ndarray) -> None:
    """StaticValue ops match array results for sampled inputs."""
    sv = u.quantity.StaticValue(arr)
    sv2 = u.quantity.StaticValue(arr)

    out = sv + sv2
    assert isinstance(out, u.quantity.StaticValue)
    assert np.allclose(np.asarray(out.array), arr + arr)
    assert np.allclose(np.asarray(sv + 1.0), arr + 1.0)
    assert np.allclose(np.asarray(1.0 + sv), arr + 1.0)


def test_quantity_ops_with_static_value() -> None:
    """Quantity should operate correctly with StaticValue-backed values."""
    sv = u.quantity.StaticValue(np.array([1.0, 2.0]))
    q_static = u.Q(sv, "m")
    q = u.Q(jnp.array([3.0, 4.0]), "m")

    out_add = q_static + q
    assert np.allclose(np.asarray(out_add.value), np.array([4.0, 6.0]))

    out_mul = q_static * q
    assert np.allclose(np.asarray(out_mul.value), np.array([3.0, 8.0]))
    assert out_mul.unit == u.unit("m2")


def test_quantity_static_value_as_jit_static_arg() -> None:
    """Quantity(StaticValue) can be used as a static argument to jitted functions."""
    # Create a Quantity with StaticValue
    sv = u.quantity.StaticValue(np.array([2.0, 3.0]))
    q_static = u.Q(sv, "m")

    # Define a function that uses the static quantity
    @partial(jax.jit, static_argnames=("q_static",))
    def compute(x, q_static):
        # Access the value from the static quantity
        return x * jnp.asarray(q_static.value)

    # First call - should compile
    x = jnp.array([1.0, 1.0])
    result = compute(x, q_static)
    assert np.allclose(result, np.array([2.0, 3.0]))

    # Second call with same static value - should use cached compilation
    result2 = compute(jnp.array([2.0, 2.0]), q_static)
    assert np.allclose(result2, np.array([4.0, 6.0]))

    # Third call with different static value - should trigger recompilation
    sv_new = u.quantity.StaticValue(np.array([5.0, 7.0]))
    q_static_new = u.Q(sv_new, "m")
    result3 = compute(jnp.array([1.0, 1.0]), q_static_new)
    assert np.allclose(result3, np.array([5.0, 7.0]))

    # Verify the quantity is actually being used as static (must be hashable)
    assert isinstance(hash(q_static), int)
    assert isinstance(hash(q_static_new), int)


def test_quantity_static_value_jit_with_operations() -> None:
    """Quantity(StaticValue) works correctly in jitted operations."""
    sv = u.quantity.StaticValue(np.array([2.0, 3.0]))
    q_static = u.Q(sv, "m")

    @partial(jax.jit, static_argnames=("scale",))
    def scale_and_add(x, scale):
        # Convert scale to array for computation
        scale_array = jnp.asarray(scale.value)
        return x + scale_array

    x = jnp.array([1.0, 2.0])
    result = scale_and_add(x, q_static)
    assert np.allclose(result, np.array([3.0, 5.0]))

    # Test with different units to ensure unit is also part of static args
    sv2 = u.quantity.StaticValue(np.array([2.0, 3.0]))
    q_static2 = u.Q(sv2, "km")

    result2 = scale_and_add(x, q_static2)
    assert np.allclose(result2, np.array([3.0, 5.0]))

    # Verify they have different hashes (different units)
    assert hash(q_static) != hash(q_static2)
