"""Unit tests for dimension construction and parsing."""

import pytest

import unxt as u


class TestDimensionSimple:
    """Test simple dimension construction without mathematical operations."""

    def test_dimension_from_string(self):
        """Test creating dimension from simple string."""
        dim = u.dimension("length")
        assert str(dim) == "length"

    def test_dimension_from_dimension(self):
        """Test dimension identity."""
        dim = u.dimension("length")
        assert u.dimension(dim) is dim

    def test_dimension_various_types(self):
        """Test various dimension types."""
        assert str(u.dimension("time")) == "time"
        assert str(u.dimension("mass")) == "mass"
        assert str(u.dimension("angle")) == "angle"


class TestDimensionMathematicalParsing:
    """Test dimension construction from mathematical expressions."""

    def test_division(self):
        """Test division operator."""
        dim = u.dimension("length / time")
        # Should be velocity/speed dimension
        expected = u.dimension("length") / u.dimension("time")
        assert dim == expected

    def test_multiplication(self):
        """Test multiplication operator."""
        dim = u.dimension("length * length")
        # Should be area
        expected = u.dimension("length") ** 2
        assert dim == expected

    def test_power(self):
        """Test power operator."""
        dim = u.dimension("length**2")
        expected = u.dimension("length") ** 2
        assert dim == expected

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

    def test_negative_exponents(self):
        """Test negative exponents in dimension expressions."""
        # Single negative exponent
        dim = u.dimension("length**-1")
        expected = u.dimension("length") ** -1
        assert dim == expected

        # Multiple dimensions with negative exponent
        dim2 = u.dimension("length**-2")
        expected2 = u.dimension("length") ** -2
        assert dim2 == expected2

    def test_negative_exponents_with_other_ops(self):
        """Test negative exponents combined with other operations."""
        # Negative exponent in compound expression
        dim = u.dimension("mass * length**-1")
        expected = u.dimension("mass") * u.dimension("length") ** -1
        assert dim == expected

        # Division followed by negative exponent
        dim2 = u.dimension("length / time**-2")
        expected2 = u.dimension("length") / u.dimension("time") ** -2
        assert dim2 == expected2

    def test_whitespace_robustness_in_parentheses(self):
        """Test whitespace inside parens dimension names is handled correctly.

        This ensures the parser is robust to user formatting variations like
        adding spaces around multi-word dimension names in parentheses.
        """
        # Standard format without extra whitespace
        dim1 = u.dimension("(amount of substance) / time")
        expected = u.dimension("catalytic activity")
        assert dim1 == expected

        # With spaces inside parentheses
        dim2 = u.dimension("( amount of substance ) / time")
        assert dim2 == expected

        # With multiple spaces
        dim3 = u.dimension("(  amount of substance  ) / time")
        assert dim3 == expected

        # Multiple parenthesized names with whitespace
        dim4 = u.dimension("( length ) / ( time )")
        expected_speed = u.dimension("speed")
        assert dim4 == expected_speed


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
