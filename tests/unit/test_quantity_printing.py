"""Tests for Quantity printing with wadler-lindig."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
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


def test_pdoc_chains_caller_custom_hook() -> None:
    """Regression: __pdoc__ used to drop the caller's custom= hook."""
    q = u.Q([1.0, 2, 3], "m")

    def custom(obj: object) -> wl.AbstractDoc | None:
        return wl.TextDoc("ARRAY_HOOK_FIRED") if isinstance(obj, jax.Array) else None

    assert wl.pformat(q, custom=custom) == "Quantity(ARRAY_HOOK_FIRED, unit='m')"


class TestShortName:
    """Test the short_name feature for wadler-lindig printing."""

    def test_quantity_has_short_name(self):
        """Test that the default ``Quantity`` (``u.Q``) has short_name 'Q'."""
        assert hasattr(u.Q, "short_name")
        assert u.Q.short_name == "Q"

    def test_use_short_name_default_false(self):
        """Test that use_short_name defaults to False."""
        q = u.Q([1, 2, 3], "m")
        result = wl.pformat(q)
        assert result.startswith("Quantity")
        assert not result.startswith("Q(")

    def test_use_short_name_true(self):
        """Test that use_short_name=True uses the short name."""
        q = u.Q([1, 2, 3], "m")
        result = wl.pformat(q, use_short_name=True)
        assert result.startswith("Q(")
        assert "unit='m'" in result

    def test_use_short_name_with_include_params(self):
        """Test that use_short_name works with include_params."""
        # The bare default ``Quantity`` has no type parameter, so
        # include_params adds nothing to the short name.
        q = u.Q([1, 2, 3], "m")
        result = wl.pformat(q, use_short_name=True, include_params=True)
        assert result.startswith("Q(")

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
    """Test str() on Quantity and Angle inside JAX JIT with tracers."""

    def test_str_quantity_in_jit(self):
        """Test that str(Quantity) works inside jax.jit with tracers.

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
        """Test that str(Quantity) works reliably in multiple JIT calls."""

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


def test_format_spec_applies_to_value_and_appends_unit() -> None:
    """``f"{q:.2f}"`` formats the value per the spec and appends the unit."""
    q = u.Q(3.14159, "m")
    assert f"{q:.2f}" == "3.14 m"
    assert format(q, ".3e") == "3.142e+00 m"


def test_format_dimensionless_has_no_trailing_unit() -> None:
    """A dimensionless quantity formats to just the value (no trailing space)."""
    q = u.Q(3.14159, "")
    assert f"{q:.2f}" == "3.14"


def test_format_empty_spec_matches_str() -> None:
    """An empty format spec preserves the existing ``str`` representation."""
    q = u.Q(3.14159, "m")
    assert f"{q}" == str(q)
    assert format(q, "") == str(q)


def test_format_static_quantity() -> None:
    """A ``StaticQuantity`` (value is a ``StaticValue``) formats like a scalar."""
    q = u.StaticQuantity(3.14159, "m")
    assert f"{q:.2f}" == "3.14 m"
    assert f"{u.StaticQuantity(3.14159, ''):.2f}" == "3.14"
    assert f"{q}" == str(q)


def test_format_non_scalar_raises() -> None:
    """A non-empty spec on a non-scalar quantity raises (NumPy semantics)."""
    for q in (u.Q([1.5, 2.5], "m"), u.StaticQuantity(np.array([1.5, 2.5]), "m")):
        with pytest.raises(TypeError, match="unsupported format string"):
            format(q, ".2f")


def test_format_presets_are_reachable_from_an_f_string() -> None:
    """`unxt.fmt.FORMAT_PRESETS` names work as format specs."""
    q = u.Q([1.0, 2, 3], "m")
    assert f"{q:mul}" == "[1., 2., 3.] * m"
    assert f"{q:bare}" == "[1., 2., 3.] m"
    assert f"{q:short}" == "f32[3] * m"
    assert f"{q:compact}" == "Q([1., 2., 3.], unit='m')"


def test_format_preset_beats_the_value_spec_under_jit() -> None:
    """The preset lookup must run before the value-spec branch.

    A non-empty spec handed straight to a tracer raises, so a preset checked
    second would be unreachable inside ``jax.jit``.
    """
    seen: list[str] = []

    @jax.jit
    def f(q: u.Q) -> u.Q:
        seen.append(f"{q:mul}")
        return q

    f(u.Q([1.0, 2, 3], "m"))
    assert seen == ["f32[3] * m"]


def test_format_colon_fill_character_is_preserved() -> None:
    """``:`` is legal as a fill character; no preset may shadow it."""
    assert f"{u.Q(3.14159, 'm'):>12.2f}" == "        3.14 m"
    assert format(u.Q(3.14159, "m"), ":>12.2f") == "::::::::3.14 m"


def test_format_unknown_spec_names_the_type_and_presets() -> None:
    with pytest.raises(ValueError, match=r"invalid format spec 'nonsense'"):
        format(u.Q(3.14, "m"), "nonsense")
