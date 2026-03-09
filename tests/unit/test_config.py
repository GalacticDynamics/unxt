"""Tests for unxt config loading and context behavior."""

import json
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

import unxt as u


def _run_isolated_import(cwd: Path, extra_code: str = "") -> str:
    """Import unxt in a subprocess and print config values as JSON."""
    code = textwrap.dedent(
        f"""
        import json
        import unxt as u

        {extra_code}

        payload = {{
            "short_arrays": u.config.quantity_repr.short_arrays,
            "use_short_name": u.config.quantity_repr.use_short_name,
            "named_unit": u.config.quantity_repr.named_unit,
            "include_params": u.config.quantity_repr.include_params,
        }}
        print(json.dumps(payload))
        """
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_auto_load_toml_from_pyproject_on_import(tmp_path: Path) -> None:
    """Auto-load TOML config from nearest pyproject.toml when importing unxt."""
    project = tmp_path / "demo_project"
    project.mkdir()
    _ = (project / "pyproject.toml").write_text(
        """\
[project]
name = "demo-project"
version = "0.1.0"

[tool.unxt.quantity.repr]
short_arrays = "compact"
use_short_name = true
named_unit = false
include_params = true
""",
        encoding="utf-8",
    )

    payload = json.loads(_run_isolated_import(project))
    assert payload["short_arrays"] == "compact"
    assert payload["use_short_name"] is True
    assert payload["named_unit"] is False
    assert payload["include_params"] is True


def test_auto_load_uses_nearest_pyproject(tmp_path: Path) -> None:
    """Auto-loading should prefer the closest pyproject.toml in parent search."""
    root = tmp_path / "workspace"
    pkg = root / "packages" / "pkg_a"
    pkg.mkdir(parents=True)

    _ = (root / "pyproject.toml").write_text(
        """\
[tool.unxt.quantity.repr]
short_arrays = false
use_short_name = false
""",
        encoding="utf-8",
    )
    _ = (pkg / "pyproject.toml").write_text(
        """\
[tool.unxt.quantity.repr]
short_arrays = "compact"
use_short_name = true
""",
        encoding="utf-8",
    )

    payload = json.loads(_run_isolated_import(pkg))
    assert payload["short_arrays"] == "compact"
    assert payload["use_short_name"] is True


def test_auto_load_ignores_pyproject_without_tool_unxt(tmp_path: Path) -> None:
    """Import should keep defaults if pyproject.toml has no [tool.unxt] section."""
    project = tmp_path / "no_unxt_config"
    project.mkdir()
    _ = (project / "pyproject.toml").write_text(
        """\
[project]
name = "demo-project"
version = "0.1.0"

[tool.other]
enabled = true
""",
        encoding="utf-8",
    )

    payload = json.loads(_run_isolated_import(project))
    assert payload["short_arrays"] is False
    assert payload["use_short_name"] is False
    assert payload["named_unit"] is True
    assert payload["include_params"] is False


def test_auto_load_ignores_invalid_toml_entries(tmp_path: Path) -> None:
    """Malformed TOML entries should not fail import and valid keys still apply."""
    project = tmp_path / "invalid_unxt_config"
    project.mkdir()
    _ = (project / "pyproject.toml").write_text(
        """\
[project]
name = "demo-project"
version = "0.1.0"

[tool.unxt.quantity.repr]
use_short_name = true
indent = "not-an-int"
typo_option = true

[tool.unxt.quantity.str]
indent = "not-an-int"
""",
        encoding="utf-8",
    )

    payload = json.loads(_run_isolated_import(project))
    # Valid key should be applied.
    assert payload["use_short_name"] is True
    # Invalid values/keys should be ignored without breaking import.
    assert payload["short_arrays"] is False
    assert payload["named_unit"] is True
    assert payload["include_params"] is False


# =============================================================================
# Context Manager Tests


def test_config_context_manager_basic() -> None:
    """Test basic context manager functionality."""
    # Get original values
    original_short_arrays = u.config.quantity_repr.short_arrays
    original_use_short_name = u.config.quantity_repr.use_short_name

    try:
        # Inside context, values should be overridden
        with u.config.override(
            quantity_repr__short_arrays="compact", quantity_repr__use_short_name=True
        ):
            assert u.config.quantity_repr.short_arrays == "compact"
            assert u.config.quantity_repr.use_short_name is True

        # After exiting context, values should be restored
        assert u.config.quantity_repr.short_arrays == original_short_arrays
        assert u.config.quantity_repr.use_short_name == original_use_short_name
    finally:
        # Ensure cleanup even if test fails
        u.config.quantity_repr.short_arrays = original_short_arrays
        u.config.quantity_repr.use_short_name = original_use_short_name


def test_config_context_manager_nested() -> None:
    """Test nested context managers."""
    original_short_arrays = u.config.quantity_repr.short_arrays
    original_use_short_name = u.config.quantity_repr.use_short_name

    try:
        # First level
        with u.config.override(quantity_repr__short_arrays="compact"):
            assert u.config.quantity_repr.short_arrays == "compact"
            assert u.config.quantity_repr.use_short_name == original_use_short_name

            # Second level (nested)
            with u.config.override(quantity_repr__use_short_name=True):
                assert u.config.quantity_repr.short_arrays == "compact"
                assert u.config.quantity_repr.use_short_name is True

                # Third level (override previous)
                with u.config.override(quantity_repr__short_arrays=True):
                    assert u.config.quantity_repr.short_arrays is True
                    assert u.config.quantity_repr.use_short_name is True

                # Back to second level
                assert u.config.quantity_repr.short_arrays == "compact"
                assert u.config.quantity_repr.use_short_name is True

            # Back to first level
            assert u.config.quantity_repr.short_arrays == "compact"
            assert u.config.quantity_repr.use_short_name == original_use_short_name

        # Back to original
        assert u.config.quantity_repr.short_arrays == original_short_arrays
        assert u.config.quantity_repr.use_short_name == original_use_short_name
    finally:
        u.config.quantity_repr.short_arrays = original_short_arrays
        u.config.quantity_repr.use_short_name = original_use_short_name


def test_config_context_manager_thread_local() -> None:
    """Test that nested config override() is thread-safe."""
    # The nested config override() uses thread-local storage for thread safety
    original_short_arrays = u.config.quantity_repr.short_arrays
    results = {}

    def thread_function(name: str) -> None:
        """Function to run in a separate thread."""
        # Check initial value
        results[f"{name}_before"] = u.config.quantity_repr.short_arrays

        # Set thread-local override using nested config's override method
        with u.config.quantity_repr.override(
            short_arrays="compact" if name == "thread1" else True
        ):
            results[f"{name}_inside"] = u.config.quantity_repr.short_arrays
            time.sleep(0.01)
            results[f"{name}_still_inside"] = u.config.quantity_repr.short_arrays

        # After exiting
        results[f"{name}_after"] = u.config.quantity_repr.short_arrays

    try:
        # Start two threads with different overrides
        thread1 = threading.Thread(target=thread_function, args=("thread1",))
        thread2 = threading.Thread(target=thread_function, args=("thread2",))

        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()

        # Verify thread1 results - should be thread-local
        assert results["thread1_before"] == original_short_arrays
        assert results["thread1_inside"] == "compact"
        assert results["thread1_still_inside"] == "compact"
        assert results["thread1_after"] == original_short_arrays

        # Verify thread2 results - should be thread-local
        assert results["thread2_before"] == original_short_arrays
        assert results["thread2_inside"] is True
        assert results["thread2_still_inside"] is True
        assert results["thread2_after"] == original_short_arrays

        # Main thread should be unchanged
        assert u.config.quantity_repr.short_arrays == original_short_arrays
    finally:
        u.config.quantity_repr.short_arrays = original_short_arrays


def test_root_config_context_manager_thread_local() -> None:
    """Test that root config override() delegates thread-local behavior."""
    original_repr_short_arrays = u.config.quantity_repr.short_arrays
    original_str_named_unit = u.config.quantity_str.named_unit
    results = {}

    def thread_function(name: str) -> None:
        """Function to run in a separate thread."""
        # Check initial values
        results[f"{name}_repr_before"] = u.config.quantity_repr.short_arrays
        results[f"{name}_str_before"] = u.config.quantity_str.named_unit

        # Set thread-local override using root config override method
        with u.config.override(
            quantity_repr__short_arrays=("compact" if name == "thread1" else True),
            quantity_str__named_unit=(name == "thread1"),
        ):
            results[f"{name}_repr_inside"] = u.config.quantity_repr.short_arrays
            results[f"{name}_str_inside"] = u.config.quantity_str.named_unit
            time.sleep(0.01)
            results[f"{name}_repr_still_inside"] = u.config.quantity_repr.short_arrays
            results[f"{name}_str_still_inside"] = u.config.quantity_str.named_unit

        # After exiting
        results[f"{name}_repr_after"] = u.config.quantity_repr.short_arrays
        results[f"{name}_str_after"] = u.config.quantity_str.named_unit

    try:
        # Start two threads with different overrides
        thread1 = threading.Thread(target=thread_function, args=("thread1",))
        thread2 = threading.Thread(target=thread_function, args=("thread2",))

        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()

        # Verify thread1 results - should be thread-local
        assert results["thread1_repr_before"] == original_repr_short_arrays
        assert results["thread1_repr_inside"] == "compact"
        assert results["thread1_repr_still_inside"] == "compact"
        assert results["thread1_repr_after"] == original_repr_short_arrays

        assert results["thread1_str_before"] == original_str_named_unit
        assert results["thread1_str_inside"] is True
        assert results["thread1_str_still_inside"] is True
        assert results["thread1_str_after"] == original_str_named_unit

        # Verify thread2 results - should be thread-local
        assert results["thread2_repr_before"] == original_repr_short_arrays
        assert results["thread2_repr_inside"] is True
        assert results["thread2_repr_still_inside"] is True
        assert results["thread2_repr_after"] == original_repr_short_arrays

        assert results["thread2_str_before"] == original_str_named_unit
        assert results["thread2_str_inside"] is False
        assert results["thread2_str_still_inside"] is False
        assert results["thread2_str_after"] == original_str_named_unit

        # Main thread should be unchanged
        assert u.config.quantity_repr.short_arrays == original_repr_short_arrays
        assert u.config.quantity_str.named_unit == original_str_named_unit
    finally:
        u.config.quantity_repr.short_arrays = original_repr_short_arrays
        u.config.quantity_str.named_unit = original_str_named_unit


def test_config_context_manager_partial_override() -> None:
    """Test that only specified values are overridden."""
    original_short_arrays = u.config.quantity_repr.short_arrays
    original_use_short_name = u.config.quantity_repr.use_short_name
    original_named_unit = u.config.quantity_repr.named_unit

    try:
        # Override only short_arrays
        with u.config.override(quantity_repr__short_arrays="compact"):
            assert u.config.quantity_repr.short_arrays == "compact"
            # Other values should remain unchanged
            assert u.config.quantity_repr.use_short_name == original_use_short_name
            assert u.config.quantity_repr.named_unit == original_named_unit

        # All values restored
        assert u.config.quantity_repr.short_arrays == original_short_arrays
        assert u.config.quantity_repr.use_short_name == original_use_short_name
        assert u.config.quantity_repr.named_unit == original_named_unit
    finally:
        u.config.quantity_repr.short_arrays = original_short_arrays
        u.config.quantity_repr.use_short_name = original_use_short_name
        u.config.quantity_repr.named_unit = original_named_unit


def test_config_context_manager_with_quantity() -> None:
    """Test context manager actually affects Quantity repr."""
    original_short_arrays = u.config.quantity_repr.short_arrays
    original_use_short_name = u.config.quantity_repr.use_short_name

    try:
        # Explicitly set config to known non-compact state
        u.config.quantity_repr.short_arrays = False
        u.config.quantity_repr.use_short_name = False

        q = u.Quantity([1, 2, 3], "m")

        # Default repr (with short_arrays=False, use_short_name=False)
        default_repr = repr(q)
        assert "Quantity(" in default_repr  # Full name, not short
        assert "Array" in default_repr  # Not compact

        # With compact arrays and short name
        with u.config.override(
            quantity_repr__short_arrays="compact", quantity_repr__use_short_name=True
        ):
            compact_repr = repr(q)
            assert "Q(" in compact_repr  # Short name
            assert "[1, 2, 3]" in compact_repr  # Compact array
            assert "Array" not in compact_repr  # Not the Array wrapper
            assert compact_repr != default_repr

        # Back to default
        restored_repr = repr(q)
        assert restored_repr == default_repr
    finally:
        u.config.quantity_repr.short_arrays = original_short_arrays
        u.config.quantity_repr.use_short_name = original_use_short_name


def test_nested_config_override() -> None:
    """Test that nested config objects support override() method directly."""
    original_short_arrays = u.config.quantity_repr.short_arrays
    original_use_short_name = u.config.quantity_repr.use_short_name

    try:
        # Test override on nested config directly
        with u.config.quantity_repr.override(
            short_arrays="compact", use_short_name=True
        ):
            assert u.config.quantity_repr.short_arrays == "compact"
            assert u.config.quantity_repr.use_short_name is True

        # After exiting context, values should be restored
        assert u.config.quantity_repr.short_arrays == original_short_arrays
        assert u.config.quantity_repr.use_short_name == original_use_short_name
    finally:
        u.config.quantity_repr.short_arrays = original_short_arrays
        u.config.quantity_repr.use_short_name = original_use_short_name


def test_nested_str_config_override() -> None:
    """Test that QuantityStrConfig override() is applied and restored."""
    original_short_arrays = u.config.quantity_str.short_arrays
    original_named_unit = u.config.quantity_str.named_unit

    try:
        with u.config.quantity_str.override(short_arrays=True, named_unit=False):
            assert u.config.quantity_str.short_arrays is True
            assert u.config.quantity_str.named_unit is False

        assert u.config.quantity_str.short_arrays == original_short_arrays
        assert u.config.quantity_str.named_unit == original_named_unit
    finally:
        u.config.quantity_str.short_arrays = original_short_arrays
        u.config.quantity_str.named_unit = original_named_unit


def test_nested_repr_config_override_invalid_key() -> None:
    """QuantityReprConfig.override() rejects unknown keyword options."""
    with pytest.raises(ValueError, match="Unknown QuantityReprConfig override option"):  # noqa: SIM117
        with u.config.quantity_repr.override(short_arrayz="compact"):
            pass


def test_nested_str_config_override_invalid_key() -> None:
    """QuantityStrConfig.override() rejects unknown keyword options."""
    with pytest.raises(ValueError, match="Unknown QuantityStrConfig override option"):  # noqa: SIM117
        with u.config.quantity_str.override(short_arrayz=True):
            pass


def test_root_override_invalid_config_section() -> None:
    """UnxtConfig.override() rejects unknown config sections."""
    with pytest.raises(ValueError, match="Unknown config section"):  # noqa: SIM117
        with u.config.override(quantity_repp__short_arrays="compact"):
            pass


def test_root_override_invalid_config_option() -> None:
    """UnxtConfig.override() rejects unknown options in known sections."""
    with pytest.raises(ValueError, match="Unknown option 'short_arrayz'"):  # noqa: SIM117
        with u.config.override(quantity_repr__short_arrayz="compact"):
            pass


def test_config_not_callable_for_context_override() -> None:
    """Config overrides must use the explicit override() method."""
    with pytest.raises(TypeError):
        _ = u.config(short_arrays="compact")


# =============================================================================
# Integration Test: Mock Package


def test_auto_load_config_from_package_pyproject_toml(tmp_path: Path) -> None:
    """Test auto-loading config from a package's pyproject.toml (integration test).

    This simulates a real-world scenario where a uv-based package has its
    unxt configuration in pyproject.toml alongside project metadata.
    """
    # Create a mock package structure
    package_root = tmp_path / "my_physics_project"
    package_root.mkdir()

    src_dir = package_root / "src" / "my_physics_project"
    src_dir.mkdir(parents=True)

    # Create a minimal package with __init__.py
    (src_dir / "__init__.py").write_text(
        '"""My physics analysis package."""\n\n__version__ = "0.1.0"\n',
        encoding="utf-8",
    )

    # Create a pyproject.toml with both project metadata and unxt config
    pyproject_content = """\
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "my-physics-project"
version = "0.1.0"
description = "A physics analysis package using unxt"
requires-python = ">=3.11"
dependencies = [
    "unxt",
    "jax",
]

[tool.uv]
dev-dependencies = [
    "pytest>=8.0",
]

[tool.unxt.quantity.repr]
# Configure unxt for compact notebook output
short_arrays = "compact"
use_short_name = true
named_unit = false
include_params = true
"""
    pyproject_file = package_root / "pyproject.toml"
    pyproject_file.write_text(pyproject_content, encoding="utf-8")

    # Create a README
    readme = package_root / "README.md"
    readme.write_text(
        "# My Physics Project\n\nUsing unxt with custom configuration.\n",
        encoding="utf-8",
    )

    # Auto-loading should happen during import in the package root directory
    payload = json.loads(_run_isolated_import(package_root))
    assert payload["short_arrays"] == "compact"
    assert payload["use_short_name"] is True
    assert payload["named_unit"] is False
    assert payload["include_params"] is True

    # Demonstrate the package structure was created correctly
    assert package_root.exists()
    assert (package_root / "pyproject.toml").exists()
    assert (package_root / "src" / "my_physics_project" / "__init__.py").exists()
    assert (package_root / "README.md").exists()
