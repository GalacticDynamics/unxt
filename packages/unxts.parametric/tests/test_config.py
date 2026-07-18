"""Tests for ``unxts.parametric.config`` (the ``include_params`` setting)."""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import unxts.parametric as up
from traitlets.config import Config

import unxt as u


def test_defaults() -> None:
    """Repr omits the type parameter by default; str includes it."""
    assert up.config.quantity_repr.include_params is False
    assert up.config.quantity_str.include_params is True


def test_repr_and_str_reflect_defaults() -> None:
    """The default repr/str of a ParametricQuantity follow the config."""
    q = up.PQ([1, 2, 3], "m")
    assert repr(q).startswith("ParametricQuantity(")  # no ['length']
    assert str(q).startswith("ParametricQuantity['length']")


def test_top_level_override() -> None:
    """`config.override(quantity_repr__include_params=...)` is thread-local."""
    q = up.PQ([1, 2, 3], "m")
    with up.config.override(quantity_repr__include_params=True):
        assert repr(q).startswith("ParametricQuantity['length']")
    # restored
    assert repr(q).startswith("ParametricQuantity(")


def test_nested_override() -> None:
    """`config.quantity_str.override(include_params=False)` is thread-local."""
    q = up.PQ([1, 2, 3], "m")
    with up.config.quantity_str.override(include_params=False):
        assert str(q).startswith("ParametricQuantity(")  # no ['length']
    # restored
    assert str(q).startswith("ParametricQuantity['length']")


def test_override_with_config_preserves_unmentioned_traits() -> None:
    """``override(cfg)`` must not reset traits the Config does not mention.

    Regression: the ``cfg`` branch overrode every trait, reading the class
    default for traits absent from the Config, so entering with an empty (or
    unrelated) Config silently discarded a globally-set value.
    """
    original = up.config.quantity_repr.include_params
    try:
        up.config.quantity_repr.include_params = True
        with up.config.quantity_repr.override(Config()):
            # Empty Config mentions nothing -> the global value must survive.
            assert up.config.quantity_repr.include_params is True
        assert up.config.quantity_repr.include_params is True
    finally:
        up.config.quantity_repr.include_params = original


def test_override_rejects_unknown_option() -> None:
    with pytest.raises(ValueError, match="Unknown option 'bogus'"):
        up.config.override(quantity_repr__bogus=True)


def test_override_requires_double_underscore() -> None:
    with pytest.raises(ValueError, match="double-underscore"):
        up.config.override(include_params=True)


def test_include_params_removed_from_unxt_config() -> None:
    """`include_params` no longer lives on unxt.config (moved here)."""
    assert "include_params" not in u.config.quantity_repr.trait_names()
    assert "include_params" not in u.config.quantity_str.trait_names()
    with pytest.raises(ValueError, match="Unknown option 'include_params'"):
        u.config.override(quantity_repr__include_params=True)


def _run_isolated_import(cwd: Path) -> str:
    """Import unxts.parametric in a subprocess and dump the config values."""
    code = "\n".join(
        [
            "import json",
            "import unxts.parametric as up",
            textwrap.dedent(
                """
                payload = {
                    "repr": up.config.quantity_repr.include_params,
                    "str": up.config.quantity_str.include_params,
                }
                print(json.dumps(payload))
                """
            ).strip(),
        ]
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_auto_load_toml_from_pyproject(tmp_path: Path) -> None:
    """Auto-load ``[tool.unxts.parametric...]`` from the nearest pyproject.toml."""
    project = tmp_path / "demo_project"
    project.mkdir()
    _ = (project / "pyproject.toml").write_text(
        """\
[project]
name = "demo-project"
version = "0.1.0"

[tool.unxts.parametric.quantity.repr]
include_params = true

[tool.unxts.parametric.quantity.str]
include_params = false
""",
        encoding="utf-8",
    )

    payload = json.loads(_run_isolated_import(project))
    assert payload["repr"] is True
    assert payload["str"] is False


def test_auto_load_defaults_without_config(tmp_path: Path) -> None:
    """Without a ``[tool.unxts.parametric]`` section, defaults are kept."""
    project = tmp_path / "demo_project"
    project.mkdir()
    _ = (project / "pyproject.toml").write_text(
        '[project]\nname = "demo-project"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )

    payload = json.loads(_run_isolated_import(project))
    assert payload["repr"] is False
    assert payload["str"] is True
