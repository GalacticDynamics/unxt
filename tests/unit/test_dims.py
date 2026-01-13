"""Unit tests for dimension construction and parsing."""

import pytest
from hypothesis import example, given, strategies as st

import unxt as u
import unxt_hypothesis as ust


class TestDimensionSimple:
    """Test simple dimension construction without mathematical operations."""

    @given(dim_name=st.sampled_from(ust.DIMENSION_NAMES))
    @example(dim_name="length")
    @example(dim_name="time")
    @example(dim_name="mass")
    @example(dim_name="angle")
    def test_dimension_from_string(self, dim_name):
        """Test creating dimension from simple string."""
        dim = u.dimension(dim_name)
        assert dim_name in str(dim)

    @given(dim=ust.named_dimensions())
    def test_dimension_from_dimension(self, dim) -> None:
        """Test dimension identity."""
        assert u.dimension(dim) is dim


class TestDimensionMathematicalParsing:
    """Test dimension construction from mathematical expressions."""

    @given(
        dim_name1=st.sampled_from(ust.DIMENSION_NAMES),
        dim_name2=st.sampled_from(ust.DIMENSION_NAMES),
    )
    def test_division(self, dim_name1, dim_name2):
        """Test division operator."""
        parsed = u.dimension(f"({dim_name1}) / ({dim_name2})")
        direct = u.dimension(dim_name1) / u.dimension(dim_name2)
        assert parsed == direct

    @given(
        dim_name1=st.sampled_from(ust.DIMENSION_NAMES),
        dim_name2=st.sampled_from(ust.DIMENSION_NAMES),
    )
    def test_multiplication(self, dim_name1, dim_name2):
        """Test multiplication operator."""
        parsed = u.dimension(f"({dim_name1}) * ({dim_name2})")
        direct = u.dimension(dim_name1) * u.dimension(dim_name2)
        assert parsed == direct

    @given(dim_name=st.sampled_from(ust.DIMENSION_NAMES), exponent=st.integers(-3, 3))
    def test_power(self, dim_name, exponent):
        """Test power operator."""
        parsed = u.dimension(f"({dim_name})**{exponent}")
        direct = u.dimension(dim_name) ** exponent
        assert parsed == direct

    def test_acceleration(self):
        """Test compound expression for acceleration."""
        dim = u.dimension("length / time**2")
        expected = u.dimension("length") / u.dimension("time") ** 2
        assert dim == expected

    def test_force(self):
        """Test compound expression for force."""
        dim = u.dimension("mass * length / time**2")
        expected = (
            u.dimension("mass") * u.dimension("length") / u.dimension("time") ** 2
        )
        assert dim == expected

    def test_with_spaces(self):
        """Test that whitespace is handled correctly."""
        dim = u.dimension("length / time ** 2")
        expected = u.dimension("length") / u.dimension("time") ** 2
        assert dim == expected

    def test_with_parentheses(self):
        """Test expressions with parentheses."""
        dim = u.dimension("mass * (length / time**2)")
        expected = (
            u.dimension("mass") * u.dimension("length") / u.dimension("time") ** 2
        )
        assert dim == expected

    def test_complex_expression(self):
        """Test more complex expression."""
        dim = u.dimension("length**2 / time")
        expected = u.dimension("length") ** 2 / u.dimension("time")
        assert dim == expected

    def test_double_asterisk_not_confused(self):
        """Test that ** is not confused with single *."""
        # This should be power, not multiplication
        dim = u.dimension("length**3")
        expected = u.dimension("length") ** 3
        assert dim == expected

        # This should be multiplication
        dim2 = u.dimension("length * length * length")
        assert dim2 == expected


class TestDimensionErrors:
    """Test error handling in dimension parsing."""

    def test_invalid_syntax(self):
        """Test that invalid syntax raises ValueError."""
        with pytest.raises(ValueError, match="Invalid dimension expression"):
            u.dimension("length / / time")

    def test_unsupported_operator_add(self):
        """Test that + is not treated as an operator - it's part of dimension name."""
        # This should try to parse as a dimension name, not as an expression
        # It will fail because astropy doesn't recognize it, but won't trigger parser
        with pytest.raises(ValueError, match="not a known physical type"):
            u.dimension("length + time")

    def test_unsupported_operator_in_expression(self):
        """Test that unsupported operators in expressions raise ValueError."""
        # Use parentheses to force parsing, then use unsupported operator
        with pytest.raises(ValueError, match="Unsupported operator"):
            u.dimension("(length) + (time)")

    def test_non_numeric_exponent(self):
        """Test that non-numeric exponents raise TypeError."""
        with pytest.raises(TypeError, match="Power exponent must be a number"):
            u.dimension("length**time")


class TestDimensionOperatorDetection:
    """Test that operator detection works correctly."""

    def test_no_operators_uses_astropy(self):
        """Test that strings without operators use astropy directly."""
        # This should not raise even for custom dimension names
        # that astropy might not recognize
        dim = u.dimension("length")
        assert dim == u.dimension("length")

    def test_asterisk_triggers_parser(self):
        """Test that * triggers the parser."""
        dim = u.dimension("length*time")
        expected = u.dimension("length") * u.dimension("time")
        assert dim == expected

    def test_slash_triggers_parser(self):
        """Test that / triggers the parser."""
        dim = u.dimension("length/time")
        expected = u.dimension("length") / u.dimension("time")
        assert dim == expected

    def test_power_triggers_parser(self):
        """Test that ** triggers the parser."""
        dim = u.dimension("length**2")
        expected = u.dimension("length") ** 2
        assert dim == expected

    @given(dim_name=st.sampled_from(ust.DIMENSION_NAMES))
    def test_parenthesized_with_whitespace(self, dim_name):
        """Test that whitespace in parentheses is stripped correctly.

        Hypothesis generates dimension names and tests that wrapping them in
        parentheses with varying whitespace doesn't change the result.
        """
        # Without spaces
        dim_no_space = u.dimension(f"({dim_name})")
        # With spaces
        dim_with_space = u.dimension(f"( {dim_name} )")
        # With multiple spaces
        dim_multi_space = u.dimension(f"(  {dim_name}  )")

        expected = u.dimension(dim_name)
        assert dim_no_space == expected
        assert dim_with_space == expected
        assert dim_multi_space == expected

    @given(
        dim_name1=st.sampled_from(ust.DIMENSION_NAMES),
        dim_name2=st.sampled_from(ust.DIMENSION_NAMES),
    )
    def test_division_expression_with_whitespace(self, dim_name1, dim_name2):
        """Test division expressions with whitespace in parentheses."""
        dim_no_space = u.dimension(f"({dim_name1}) / ({dim_name2})")
        dim_with_space = u.dimension(f"( {dim_name1} ) / ( {dim_name2} )")

        expected = u.dimension(dim_name1) / u.dimension(dim_name2)
        assert dim_no_space == expected
        assert dim_with_space == expected
