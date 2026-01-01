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


def test_static_value_pdoc() -> None:
    """StaticValue uses the contained value for formatting."""
    value = u.quantity.StaticValue(np.array([1.0, 2.0]))
    assert wl.pformat(value) == wl.pformat(np.array([1.0, 2.0]))


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
    assert repr(sv) == repr(arr)
    assert np.allclose(np.asarray(sv.sum()), np.sum(arr))


def test_static_value_dunder_protocols_explicit() -> None:
    """Exercise StaticValue protocol dunders directly."""
    arr = np.array([1, 2], dtype=np.int32)
    sv = u.quantity.StaticValue(arr)

    assert len(sv) == 2
    assert list(iter(sv)) == [1, 2]
    assert sv[1] == 2
    assert sv.shape == (2,)
    assert sv.__repr__() == repr(arr)
    assert sv.__eq__(u.quantity.StaticValue(arr)) is True
    assert sv.__eq__(object()) is NotImplemented


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
