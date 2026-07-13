"""Configuration for ``unxts.parametric`` using traitlets.

This mirrors :mod:`unxt.config` for the parametric-only ``include_params``
display setting, reusing unxt's thread-local override and TOML-loading
machinery. TOML configuration is discovered automatically from the nearest
``pyproject.toml`` at import time, from the ``[tool.unxts.parametric]`` section.
"""

__all__ = (
    "ParametricConfig",
    "ParametricQuantityReprConfig",
    "ParametricQuantityStrConfig",
    "config",
)

import contextlib
import threading
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, ClassVar, Final

from traitlets import Bool, TraitError
from traitlets.config import Config, SingletonConfigurable

from unxt._src.config import (
    LocalConfigurable,
    _find_pyproject,
    _load_toml_config_from_pyproject,
    _NestedConfigContext,
)

PARAMETRIC_REPR_CONFIG_KEYS: Final = frozenset({"include_params"})
PARAMETRIC_STR_CONFIG_KEYS: Final = frozenset({"include_params"})

PARAMETRIC_OVERRIDE_CONFIG_KEYS: Final = {
    "quantity_repr": PARAMETRIC_REPR_CONFIG_KEYS,
    "quantity_str": PARAMETRIC_STR_CONFIG_KEYS,
}


class _ParametricLocalConfig(LocalConfigurable):
    """Base for the parametric repr/str configs with thread-local overrides.

    Subclasses set ``_config_keys`` (their overridable trait names) and their
    own ``_local`` thread-local storage; this base provides the generic
    ``__getattribute__`` override lookup and ``override()`` context manager.
    """

    _config_keys: ClassVar[frozenset[str]] = frozenset()

    def __getattribute__(self, name: str) -> Any:
        """Get attribute, checking thread-local overrides first."""
        keys = object.__getattribute__(self, "_config_keys")
        if name in keys:
            with contextlib.suppress(AttributeError):
                local = object.__getattribute__(self, "_local")
                if hasattr(local, "stack") and local.stack:
                    for overrides in reversed(local.stack):
                        if name in overrides:
                            return overrides[name]

        return object.__getattribute__(self, name)

    def override(
        self, cfg: Config | None = None, /, **kwargs: Any
    ) -> _NestedConfigContext:
        """Create a context manager for temporary config changes.

        Examples
        --------
        >>> import unxts.parametric as up
        >>> with up.config.quantity_repr.override(include_params=True):
        ...     print(up.config.quantity_repr.include_params)
        True

        """
        keys = self._config_keys
        cls_name = type(self).__name__

        if cfg is not None and kwargs:
            msg = "Cannot specify both cfg and keyword arguments to override()"
            raise ValueError(msg)

        if kwargs:
            unknown_keys = set(kwargs) - keys
            if unknown_keys:
                valid_keys = ", ".join(sorted(keys))
                unknown = ", ".join(sorted(unknown_keys))
                msg = (
                    f"Unknown {cls_name} override option(s): {unknown}. "
                    f"Valid options are: {valid_keys}"
                )
                raise ValueError(msg)

            # Validate/resolve through traitlets immediately for fast, clear errors.
            temp_instance = self.__class__()
            validated_kwargs: dict[str, Any] = {}
            for key, value in kwargs.items():
                try:
                    setattr(temp_instance, key, value)
                except TraitError as e:
                    msg = (
                        f"Invalid value for {cls_name} override option "
                        f"'{key}': {value!r}"
                    )
                    raise ValueError(msg) from e
                validated_kwargs[key] = object.__getattribute__(temp_instance, key)
            kwargs = validated_kwargs

        if cfg is not None:
            temp_instance = self.__class__(config=cfg)
            overrides = {}
            for trait_name in self.trait_names():
                overrides[trait_name] = object.__getattribute__(
                    temp_instance, trait_name
                )
            kwargs = overrides

        return _NestedConfigContext(self, kwargs)


class ParametricQuantityReprConfig(_ParametricLocalConfig):
    """``include_params`` for ``ParametricQuantity.__repr__`` (default ``False``)."""

    _local: threading.local = threading.local()
    _config_keys: ClassVar[frozenset[str]] = PARAMETRIC_REPR_CONFIG_KEYS

    include_params: ClassVar[object] = Bool(
        default_value=False,
        help="Include type parameters in repr for parametric quantities",
    ).tag(config=True)


class ParametricQuantityStrConfig(_ParametricLocalConfig):
    """``include_params`` for ``ParametricQuantity.__str__`` (default ``True``)."""

    _local: threading.local = threading.local()
    _config_keys: ClassVar[frozenset[str]] = PARAMETRIC_STR_CONFIG_KEYS

    include_params: ClassVar[object] = Bool(
        default_value=True,
        help="Include type parameters in str for parametric quantities",
    ).tag(config=True)


class ParametricConfig(SingletonConfigurable):
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

    def __init__(self, **kwargs: Any) -> None:
        """Initialize ParametricConfig with nested config instances."""
        super().__init__(**kwargs)
        self.quantity_repr = ParametricQuantityReprConfig(
            config=self.config, parent=self
        )
        self.quantity_str = ParametricQuantityStrConfig(config=self.config, parent=self)

    def override(self, **kwargs: Any) -> "_ParametricConfigContext":
        """Create a context manager for temporary config changes.

        Use double-underscore notation for nested configs, e.g.
        ``quantity_repr__include_params=True``.
        """
        parsed_overrides: dict[str, dict[str, Any]] = {}
        for key, value in kwargs.items():
            if "__" not in key:
                msg = (
                    f"Config override '{key}' must use double-underscore notation "
                    "(e.g., 'quantity_repr__include_params')"
                )
                raise ValueError(msg)

            config_name, attr_name = key.split("__", 1)
            if config_name not in PARAMETRIC_OVERRIDE_CONFIG_KEYS:
                valid_configs = ", ".join(sorted(PARAMETRIC_OVERRIDE_CONFIG_KEYS))
                msg = (
                    f"Unknown config section '{config_name}' in override key "
                    f"'{key}'. Valid sections are: {valid_configs}"
                )
                raise ValueError(msg)

            valid_attrs = PARAMETRIC_OVERRIDE_CONFIG_KEYS[config_name]
            if attr_name not in valid_attrs:
                valid_options = ", ".join(sorted(valid_attrs))
                msg = (
                    f"Unknown option '{attr_name}' for config section "
                    f"'{config_name}' in override key '{key}'. "
                    f"Valid options are: {valid_options}"
                )
                raise ValueError(msg)

            parsed_overrides.setdefault(config_name, {})[attr_name] = value

        return _ParametricConfigContext(self, parsed_overrides)


@dataclass(slots=True)
class _ParametricConfigContext:
    """Context manager for temporary ``ParametricConfig`` changes.

    Not instantiated directly -- use ``config.override(**kwargs)``.
    """

    config: ParametricConfig
    overrides: dict[str, dict[str, Any]]  # {config_name: {attr: value}}
    _stack: contextlib.ExitStack = field(init=False, repr=False)

    def __enter__(self) -> ParametricConfig:
        """Enter context, applying thread-local config overrides."""
        self._stack = contextlib.ExitStack()
        for config_name, attrs in self.overrides.items():
            nested_config = getattr(self.config, config_name)
            self._stack.enter_context(nested_config.override(**attrs))
        return self.config

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context, restoring previous config values."""
        self._stack.__exit__(exc_type, exc_val, exc_tb)


# Mapping from TOML sub-path to Config class name (under [tool.unxts.parametric])
_TOML_PATH_TO_CONFIG_CLASS: Final = {
    ("quantity", "repr"): "ParametricQuantityReprConfig",
    ("quantity", "str"): "ParametricQuantityStrConfig",
}

# Mapping from Config class name to (config instance, valid keys)
_CONFIG_CLASS_TO_INSTANCE: Final[dict[str, tuple[Any, frozenset[str]]]] = {}


def _initialize_config_mapping(cfg: ParametricConfig) -> None:
    """Populate the class-name -> instance mapping (call after singleton init)."""
    _CONFIG_CLASS_TO_INSTANCE["ParametricQuantityReprConfig"] = (
        cfg.quantity_repr,
        PARAMETRIC_REPR_CONFIG_KEYS,
    )
    _CONFIG_CLASS_TO_INSTANCE["ParametricQuantityStrConfig"] = (
        cfg.quantity_str,
        PARAMETRIC_STR_CONFIG_KEYS,
    )


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
        config_instance, valid_keys = _CONFIG_CLASS_TO_INSTANCE[class_name]
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
