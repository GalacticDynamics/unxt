"""Tests for ParametricQuantity printing with wadler-lindig."""

import equinox as eqx
import jax
import jax.numpy as jnp
import wadler_lindig as wl

import unxt as u
from unxt.units import unit as parse_unit


class FlaggedQuantity(u.AbstractQuantity):
    """Test helper quantity with a default-valued extra field."""

    value: jax.Array = eqx.field(converter=u.quantity.convert_to_quantity_value)
    unit: object = eqx.field(static=True, converter=parse_unit)
    flag: bool = eqx.field(static=True, kw_only=True, default=False)


class QuantityWithNonSingletonDefaults(u.AbstractQuantity):
    """Test helper quantity with non-singleton default values."""

    value: jax.Array = eqx.field(converter=u.quantity.convert_to_quantity_value)
    unit: object = eqx.field(static=True, converter=parse_unit)
    scale: float = eqx.field(static=True, kw_only=True, default=1.0)
    label: str = eqx.field(static=True, kw_only=True, default="default")


def test_repr_hides_default_extra_field() -> None:
    """Default-valued extra fields should be omitted from pretty reprs."""
    q_default = FlaggedQuantity([1, 2, 3], "m")
    q_nondefault = FlaggedQuantity([1, 2, 3], "m", flag=True)

    default_repr = wl.pformat(q_default)
    nondefault_repr = wl.pformat(q_nondefault)

    assert "flag=" not in default_repr
    assert "flag=True" in nondefault_repr


def test_repr_hides_non_singleton_defaults() -> None:
    """Non-singleton default values (float, str) should be omitted when equal."""
    # Both fields have default values (not same object, but equal)
    q_default = QuantityWithNonSingletonDefaults(
        [1, 2, 3], "m", scale=1.0, label="default"
    )
    # One field has a non-default value
    q_custom_scale = QuantityWithNonSingletonDefaults([1, 2, 3], "m", scale=2.0)
    q_custom_label = QuantityWithNonSingletonDefaults([1, 2, 3], "m", label="custom")

    default_repr = wl.pformat(q_default)
    custom_scale_repr = wl.pformat(q_custom_scale)
    custom_label_repr = wl.pformat(q_custom_label)

    # Default values should be omitted (equality check, not identity)
    assert "scale=" not in default_repr
    assert "label=" not in default_repr

    # Non-default values should appear
    assert "scale=2.0" in custom_scale_repr
    assert "label=" not in custom_scale_repr  # label is still default

    assert "label='custom'" in custom_label_repr
    assert "scale=" not in custom_label_repr  # scale is still default


class TestShortName:
    """Test the short_name feature for wadler-lindig printing."""

    def test_quantity_has_short_name(self):
        """Test that the default ``Quantity`` (``u.Q``) has short_name 'Q'."""
        assert hasattr(u.Q, "short_name")
        assert u.Q.short_name == "Q"

    def test_parametricquantity_short_name(self):
        """Test that ``ParametricQuantity`` (``u.PQ``) has short_name 'PQ'."""
        assert hasattr(u.PQ, "short_name")
        assert u.PQ.short_name == "PQ"

    def test_use_short_name_default_false(self):
        """Test that use_short_name defaults to False."""
        q = u.Q([1, 2, 3], "m")
        result = wl.pformat(q)
        assert result.startswith("Quantity")
        assert not result.startswith("Q(")

        # The parametric class uses its full name by default too.
        pq = u.PQ([1, 2, 3], "m")
        pq_result = wl.pformat(pq)
        assert pq_result.startswith("ParametricQuantity")
        assert not pq_result.startswith("PQ(")

    def test_use_short_name_true(self):
        """Test that use_short_name=True uses the short name."""
        q = u.Q([1, 2, 3], "m")
        result = wl.pformat(q, use_short_name=True)
        assert result.startswith("Q(")
        assert "unit='m'" in result

    def test_parametric_use_short_name_true(self):
        """Test that ``ParametricQuantity`` uses its short name 'PQ'."""
        pq = u.PQ([1, 2, 3], "m")
        result = wl.pformat(pq, use_short_name=True)
        assert result.startswith("PQ(")
        assert "unit='m'" in result

    def test_use_short_name_with_include_params(self):
        """Test that use_short_name works with include_params."""
        # The bare default ``Quantity`` has no type parameter, so
        # include_params adds nothing to the short name.
        q = u.Q([1, 2, 3], "m")
        result = wl.pformat(q, use_short_name=True, include_params=True)
        assert result.startswith("Q(")

        # The parametric class shows its dimension parameter.
        pq = u.PQ([1, 2, 3], "m")
        pq_result = wl.pformat(pq, use_short_name=True, include_params=True)
        assert pq_result.startswith("PQ['length']")

    def test_use_short_name_with_named_unit_false(self):
        """Test that use_short_name works with named_unit=False."""
        q = u.Q([1, 2, 3], "m")
        result = wl.pformat(q, use_short_name=True, named_unit=False)
        assert result.startswith("Q(")
        # Should have unit as positional arg not named
        assert "'m')" in result or ", 'm')" in result

    def test_use_short_name_with_short_arrays(self):
        """Test that use_short_name works with short_arrays."""
        q = u.Q([1, 2, 3], "m")

        # Default short_arrays=True
        result = wl.pformat(q, use_short_name=True, short_arrays=True)
        assert result.startswith("Q(")
        assert "i32[3]" in result

        # short_arrays=False
        result = wl.pformat(q, use_short_name=True, short_arrays=False)
        assert result.startswith("Q(")
        assert "Array(" in result

    def test_use_short_name_with_short_arrays_compact(self):
        """Test that use_short_name works with short_arrays='compact'."""
        q = u.Q([1, 2, 3], "m")
        result = wl.pformat(q, use_short_name=True, short_arrays="compact")
        assert result.startswith("Q(")
        assert "[1, 2, 3]" in result

    def test_bare_quantity_use_short_name(self):
        """Test that the bare default ``Quantity`` uses its short name 'Q'."""
        q = u.quantity.Quantity([1, 2, 3], "m")
        result = wl.pformat(q, use_short_name=True)
        assert result.startswith("Q(")

        # Without use_short_name it uses the full class name.
        assert wl.pformat(q).startswith("Quantity")

    def test_pprint(self):
        """Test that pprint works with use_short_name."""
        q = u.Q([1, 2, 3], "m")
        # This should not raise an error
        wl.pprint(q, use_short_name=True)

    def test_pdoc_method_directly(self):
        """Test calling __pdoc__ directly with use_short_name."""
        q = u.Q([1, 2, 3], "m")

        doc = q.__pdoc__(use_short_name=False)
        formatted = wl.pformat(doc)
        assert formatted.startswith("Quantity")

        doc = q.__pdoc__(use_short_name=True)
        formatted = wl.pformat(doc)
        assert formatted.startswith("Q(")


class TestStringConversionWithJIT:
    """Test str() on ParametricQuantity and Angle inside JAX JIT with tracers."""

    def test_str_quantity_in_jit(self):
        """Test that str(ParametricQuantity) works inside jax.jit with tracers.

        When values are tracers inside JIT, str() should work without raising an
        error.  We verify this by calling str() during JIT tracing and returning
        a derived value.
        """

        @jax.jit
        def process_with_str(q: u.Q) -> u.Q:
            # Call str() on the tracer to verify it doesn't raise
            _ = str(q)
            # Return the quantity multiplied by 2
            return q * 2

        q = u.Q([1.0, 2.0, 3.0], "m")
        result = process_with_str(q)
        assert result.unit == q.unit
        assert jnp.allclose(result.value, q.value * 2)

    def test_str_angle_in_jit(self):
        """Test that str(Angle) works inside jax.jit with tracers.

        When values are tracers inside JIT, str() should work without raising an error.
        """

        @jax.jit
        def process_with_str(angle: u.Angle) -> u.Angle:
            # Call str() on the tracer to verify it doesn't raise
            _ = str(angle)
            # Return the angle multiplied by 2
            return angle * 2

        angle = u.Angle([0.5, 1.0, 1.5], "rad")
        result = process_with_str(angle)
        assert result.unit == angle.unit
        assert jnp.allclose(result.value, angle.value * 2)

    def test_str_quantity_multiple_calls_in_jit(self):
        """Test that str(ParametricQuantity) works reliably in multiple JIT calls."""

        @jax.jit
        def process_and_stringify(q: u.Q) -> u.Q:
            # Multiple str() calls shouldn't affect the computation
            _ = str(q)
            q_doubled = q * 2
            _ = str(q_doubled)
            return q_doubled

        q = u.Q(5.0, "kg")
        result = process_and_stringify(q)
        assert result.unit == q.unit

        assert jnp.allclose(result.value, q.value * 2)
