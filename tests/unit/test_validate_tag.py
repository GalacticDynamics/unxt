"""Unit tests for scripts/validate_tag.py.

Tests cover the core release/tagging validation rules:
- Package tag required (not bare vX.Y.Z tags)
- .0 releases require coordinator tag
- Legacy (<1.11) behavior exceptions
- subprocess-based coordinator lookup with error handling
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add scripts to path so we can import
scripts_path = Path(__file__).parent.parent.parent / "scripts"
sys.path.insert(0, str(scripts_path))

from validate_tag import (  # noqa: E402
    check_coordinator_tag_exists,
    parse_version_tag,
    validate_tag_for_package,
)


class TestParseVersionTag:
    """Test tag parsing functionality."""

    def test_valid_package_specific_tag(self):
        """Parse package-specific tags correctly."""
        assert parse_version_tag("unxt-v1.11.0") == ("unxt", 1, 11, 0)
        assert parse_version_tag("unxt-api-v2.3.4") == ("unxt-api", 2, 3, 4)
        assert parse_version_tag("unxt-hypothesis-v0.1.2") == (
            "unxt-hypothesis",
            0,
            1,
            2,
        )

    def test_valid_bare_tag(self):
        """Parse bare coordinator tags correctly."""
        assert parse_version_tag("v1.11.0") == ("", 1, 11, 0)
        assert parse_version_tag("v2.3.4") == ("", 2, 3, 4)

    def test_invalid_formats(self):
        """Return None for invalid tag formats."""
        assert parse_version_tag("1.11.0") is None  # Missing 'v' prefix
        assert parse_version_tag("va.b.c") is None  # Non-numeric version
        assert parse_version_tag("unxt-1.11.0") is None  # Missing 'v'
        assert parse_version_tag("unxt_v1.11.0") is None  # Underscore instead of dash
        assert parse_version_tag("v1.11") is None  # Missing patch version
        assert parse_version_tag("random-tag") is None  # Not a version tag


class TestCheckCoordinatorTagExists:
    """Test coordinator tag existence checking."""

    def test_tag_exists(self):
        """Return True when coordinator tag exists."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "v1.11.0\n"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            assert check_coordinator_tag_exists("1.11.0") is True

    def test_tag_does_not_exist(self):
        """Return False when coordinator tag doesn't exist."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "\n"  # Empty result
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            assert check_coordinator_tag_exists("1.11.0") is False

    def test_git_command_failure_raises_error(self):
        """Raise RuntimeError when git command fails."""
        mock_result = MagicMock()
        mock_result.returncode = 128  # Git error
        mock_result.stdout = ""
        mock_result.stderr = "fatal: not a git repository"

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError) as exc_info:
                check_coordinator_tag_exists("1.11.0")

            error_msg = str(exc_info.value)
            assert "git tag -l failed" in error_msg
            assert "fetch-depth" in error_msg

    def test_git_command_failure_without_stderr(self):
        """Raise RuntimeError when git fails without stderr output."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError) as exc_info:
                check_coordinator_tag_exists("1.11.0")

            assert "git tag -l failed" in str(exc_info.value)


class TestValidateTagForPackage:
    """Test tag validation logic for packages."""

    # Legacy behavior tests

    def test_legacy_tags_always_valid(self):
        """Tags for version 1.10.x and lower are always valid."""
        # Bare tags are acceptable for legacy
        is_valid, error = validate_tag_for_package("v1.10.0", "unxt")
        assert is_valid is True
        assert error == ""

        is_valid, error = validate_tag_for_package("v1.9.5", "unxt")
        assert is_valid is True

        is_valid, error = validate_tag_for_package("v0.5.0", "unxt-api")
        assert is_valid is True

    # Invalid format tests

    def test_invalid_tag_format_rejected(self):
        """Invalid tag formats are rejected."""
        is_valid, error = validate_tag_for_package("invalid-tag", "unxt")
        assert is_valid is False
        assert "Invalid tag format" in error

        is_valid, error = validate_tag_for_package("1.11.0", "unxt")
        assert is_valid is False

    # Rule 1: Package-specific tag required (post-1.10)

    def test_bare_tag_rejected_for_new_versions(self):
        """Bare vX.Y.Z tags are rejected for versions >= 1.11."""
        is_valid, error = validate_tag_for_package("v1.11.0", "unxt")
        assert is_valid is False
        assert "package-specific tags" in error
        assert "coordinator tags" in error

        is_valid, error = validate_tag_for_package("v2.0.0", "unxt-api")
        assert is_valid is False

    # Rule 2: Package must match

    def test_package_specific_tag_matches_package(self):
        """Package-specific tags must match the specified package."""
        # Valid: tag matches package
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "v1.11.0\n"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            is_valid, error = validate_tag_for_package("unxt-v1.11.0", "unxt")
            assert is_valid is True
            assert error == ""

    def test_package_specific_tag_wrong_package(self):
        """Reject tag when package doesn't match."""
        is_valid, error = validate_tag_for_package("unxt-api-v1.11.0", "unxt")
        assert is_valid is False
        assert "This tag is for package 'unxt-api'" in error
        assert "but this workflow is for package 'unxt'" in error

        is_valid, error = validate_tag_for_package("unxt-v1.11.0", "unxt-api")
        assert is_valid is False
        assert "This tag is for package 'unxt'" in error

    # Rule 3: .0 releases require coordinator tag

    def test_dot_zero_release_with_coordinator_tag(self):
        """Accept .0 release when coordinator tag exists."""
        # Test for unxt-v1.11.0 with v1.11.0 coordinator
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "v1.11.0\n"  # Coordinator tag exists
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            is_valid, error = validate_tag_for_package("unxt-v1.11.0", "unxt")
            assert is_valid is True
            assert error == ""

        # Test for unxt-api-v2.0.0 with v2.0.0 coordinator
        mock_result_2 = MagicMock()
        mock_result_2.returncode = 0
        mock_result_2.stdout = "v2.0.0\n"  # Different coordinator tag
        mock_result_2.stderr = ""

        with patch("subprocess.run", return_value=mock_result_2):
            is_valid, error = validate_tag_for_package("unxt-api-v2.0.0", "unxt-api")
            assert is_valid is True

    def test_dot_zero_release_without_coordinator_tag(self):
        """Reject .0 release when coordinator tag doesn't exist."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "\n"  # No coordinator tag
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            is_valid, error = validate_tag_for_package("unxt-v1.11.0", "unxt")
            assert is_valid is False
            assert "must have a corresponding coordinator tag" in error
            assert "v1.11.0" in error
            assert "synchronized releases" in error

    # Bug-fix releases (patch > 0)

    def test_bugfix_release_no_coordinator_required(self):
        """Bug-fix releases (X.Y.Z where Z > 0) don't require coordinator tag."""
        # No subprocess.run call should happen for patch > 0
        is_valid, error = validate_tag_for_package("unxt-v1.11.1", "unxt")
        assert is_valid is True
        assert error == ""

        is_valid, error = validate_tag_for_package("unxt-api-v2.3.5", "unxt-api")
        assert is_valid is True

        is_valid, error = validate_tag_for_package(
            "unxt-hypothesis-v1.12.99", "unxt-hypothesis"
        )
        assert is_valid is True

    # Package name normalization

    def test_none_package_defaults_to_unxt(self):
        """package=None should default to 'unxt'."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "v1.11.0\n"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            # Should validate as if package='unxt'
            is_valid, error = validate_tag_for_package("unxt-v1.11.0", None)
            assert is_valid is True

            # Should fail for wrong package
            is_valid, error = validate_tag_for_package("unxt-api-v1.11.0", None)
            assert is_valid is False
            assert "This tag is for package 'unxt-api'" in error
            assert "but this workflow is for package 'unxt'" in error


class TestIntegration:
    """Integration tests combining multiple validation rules."""

    def test_complete_validation_flow_dot_zero_release(self):
        """Test complete flow for .0 release validation."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "v1.11.0\n"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result) as mock_subprocess:
            # Valid .0 release
            is_valid, _ = validate_tag_for_package("unxt-v1.11.0", "unxt")
            assert is_valid is True

            # Verify git was called to check coordinator tag
            mock_subprocess.assert_called_once()
            call_args = mock_subprocess.call_args[0][0]
            assert call_args == ["git", "tag", "-l", "v1.11.0"]

    def test_complete_validation_flow_bugfix_release(self):
        """Test complete flow for bug-fix release (no coordinator check)."""
        with patch("subprocess.run") as mock_subprocess:
            # Valid bug-fix release
            is_valid, _ = validate_tag_for_package("unxt-v1.11.1", "unxt")
            assert is_valid is True

            # Git should NOT be called for bug-fix releases
            mock_subprocess.assert_not_called()

    def test_validation_with_git_failure(self):
        """Test validation handles git failures gracefully."""
        mock_result = MagicMock()
        mock_result.returncode = 128
        mock_result.stdout = ""
        mock_result.stderr = "fatal: not a git repository"

        with patch("subprocess.run", return_value=mock_result):
            # Should raise RuntimeError from check_coordinator_tag_exists
            with pytest.raises(RuntimeError) as exc_info:
                validate_tag_for_package("unxt-v1.11.0", "unxt")

            assert "git tag -l failed" in str(exc_info.value)
