"""Configuration for ``unxts.parametric`` using traitlets.

This mirrors :mod:`unxt.config` for the parametric-only ``include_params``
display setting, reusing unxt's thread-local override and TOML-loading
machinery. TOML configuration is discovered automatically from the nearest
``pyproject.toml`` at import time, from the ``[tool.unxts.parametric]`` section.
"""
# The config classes mirror the structure of ``unxt._src.config`` (and the
# public ``config`` re-export) by design; silence the duplicate-code check.
# pylint: disable=duplicate-code

__all__ = (
    "ParametricConfig",
    "ParametricQuantityReprConfig",
    "ParametricQuantityStrConfig",
    "config",
)

import tomllib
from pathlib import Path
from typing import Any, ClassVar, Final

from traitlets import Bool, TraitError
from traitlets.config import SingletonConfigurable

from unxt._src.config import (
    AbstractUnxtConfig,
    LocalConfigurable,
    _find_pyproject,
    _load_toml_config_from_pyproject,
    _override_keys,
)


class ParametricQuantityReprConfig(LocalConfigurable):
    """``include_params`` for ``ParametricQuantity.__repr__`` (default ``False``).

    ``__getattribute__`` (thread-local override lookup) and ``override()`` (the
    context manager) are inherited from
    :class:`~unxt._src.config.LocalConfigurable`; this class only declares its
    overridable trait.
    """

    include_params: ClassVar[object] = Bool(
        default_value=False,
        help="Include type parameters in repr for parametric quantities",
    ).tag(config=True)


class ParametricQuantityStrConfig(LocalConfigurable):
    """``include_params`` for ``ParametricQuantity.__str__`` (default ``True``).

    See :class:`ParametricQuantityReprConfig` — the override machinery is
    inherited from :class:`~unxt._src.config.LocalConfigurable`.
    """

    include_params: ClassVar[object] = Bool(
        default_value=True,
        help="Include type parameters in str for parametric quantities",
    ).tag(config=True)


class ParametricConfig(AbstractUnxtConfig, SingletonConfigurable):
    """Configuration for ``unxts.parametric`` display options.

    Singleton config controlling whether a ``ParametricQuantity`` renders its
    dimension type parameter (e.g. ``['length']``) in ``repr()`` / ``str()``.

    - ``quantity_repr.include_params``: default ``False``
    - ``quantity_str.include_params``: default ``True``

    Examples
    --------
    >>> import unxts.parametric as up

    >>> up.config.quantity_repr.include_params
    False

    >>> with up.config.override(quantity_repr__include_params=True):
    ...     print(repr(up.PQ([1, 2, 3], "m")))
    ParametricQuantity['length'](Array([1, 2, 3], dtype=int32), unit='m')

    >>> print(repr(up.PQ([1, 2, 3], "m")))
    ParametricQuantity(Array([1, 2, 3], dtype=int32), unit='m')

    """

    classes: ClassVar[list[type]] = [
        ParametricQuantityReprConfig,
        ParametricQuantityStrConfig,
    ]
    _override_sections: ClassVar[tuple[str, ...]] = ("quantity_repr", "quantity_str")

    def __init__(self, **kwargs: Any) -> None:
        """Initialize ParametricConfig with nested config instances."""
        super().__init__(**kwargs)
        self.quantity_repr = ParametricQuantityReprConfig(
            config=self.config, parent=self
        )
        self.quantity_str = ParametricQuantityStrConfig(config=self.config, parent=self)

    # ``override()`` (the top-level context manager) is inherited from
    # ``AbstractUnxtConfig`` and returns unxt's ``_ConfigContext``.


# Mapping from TOML sub-path to Config class name (under [tool.unxts.parametric])
_TOML_PATH_TO_CONFIG_CLASS: Final = {
    ("quantity", "repr"): "ParametricQuantityReprConfig",
    ("quantity", "str"): "ParametricQuantityStrConfig",
}

# Mapping from Config class name to config instance
_CONFIG_CLASS_TO_INSTANCE: Final[dict[str, Any]] = {}


def _initialize_config_mapping(cfg: ParametricConfig) -> None:
    """Populate the class-name -> instance mapping (call after singleton init)."""
    for name in cfg._override_sections:  # noqa: SLF001
        instance = getattr(cfg, name)
        _CONFIG_CLASS_TO_INSTANCE[type(instance).__name__] = instance


def _auto_load_project_toml_config(cfg: ParametricConfig, /) -> None:
    """Auto-load ``[tool.unxts.parametric]`` config without import-time errors."""
    pyproject = _find_pyproject(Path.cwd())
    if pyproject is None:
        return

    try:
        loaded = _load_toml_config_from_pyproject(
            pyproject,
            tool_path=("unxts", "parametric"),
            path_to_class=_TOML_PATH_TO_CONFIG_CLASS,
        )
    except (OSError, tomllib.TOMLDecodeError, TypeError, KeyError):
        return

    if not loaded:
        return

    for class_name, class_config in loaded.items():
        if class_name not in _CONFIG_CLASS_TO_INSTANCE:
            continue
        config_instance = _CONFIG_CLASS_TO_INSTANCE[class_name]
        valid_keys = _override_keys(type(config_instance))
        for key, value in class_config.items():
            if key not in valid_keys:
                continue
            try:
                setattr(config_instance, key, value)
            except (TraitError, AttributeError):
                continue


# Create the global singleton instance
config = ParametricConfig.instance()
_initialize_config_mapping(config)
_auto_load_project_toml_config(config)
