"""Tests for doctest configuration helpers."""

from doctest import ELLIPSIS
import importlib.util
from pathlib import Path


def _load_root_conftest():
    """Load the repository-root ``conftest.py`` for direct unit testing."""
    current = Path(__file__).resolve()
    root = next(
        ((parent / "conftest.py") for parent in current.parents if (parent / "conftest.py").exists()),
        None,
    )
    if root is None:
        msg = "Could not locate repository-root conftest.py"
        raise RuntimeError(msg)
    spec = importlib.util.spec_from_file_location("root_conftest", root)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ROOT_CONFTEST = _load_root_conftest()


def test_jax_output_checker_ignores_weak_type_repr_detail() -> None:
    """Doctest output checker should ignore unstable weak_type repr detail."""
    checker = ROOT_CONFTEST.JaxAwareOutputChecker()

    assert checker.check_output(
        "Quantity(Array(2., dtype=float32), unit='m')\n",
        "Quantity(Array(2., dtype=float32, weak_type=True), unit='m')\n",
        ELLIPSIS,
    )


def test_jax_output_checker_still_rejects_real_output_changes() -> None:
    """Doctest output checker should still catch substantive mismatches."""
    checker = ROOT_CONFTEST.JaxAwareOutputChecker()

    assert not checker.check_output(
        "Quantity(Array(2., dtype=float32), unit='m')\n",
        "Quantity(Array(2., dtype=float32, weak_type=True), unit='cm')\n",
        ELLIPSIS,
    )


def test_jax_output_checker_accepts_optional_scalar_repr_metadata() -> None:
    """Doctest output checker should allow scalar dtype repr metadata to vary."""
    checker = ROOT_CONFTEST.JaxAwareOutputChecker()

    assert checker.check_output(
        "Quantity(Array(100., dtype=float32, ...), unit='cm')\n",
        "Quantity(Array(100., dtype=float32), unit='cm')\n",
        ELLIPSIS,
    )


def test_jax_output_checker_accepts_added_scalar_repr_metadata() -> None:
    """Doctest output checker should allow additional scalar repr metadata."""
    checker = ROOT_CONFTEST.JaxAwareOutputChecker()

    assert checker.check_output(
        "Quantity(Array(2.7182817, dtype=float32), unit='')\n",
        "Quantity(Array(2.7182817, dtype=float32, weak_type=True), unit='')\n",
        ELLIPSIS,
    )
