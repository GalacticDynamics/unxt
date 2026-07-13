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

import contextlib
import threading
import tomllib
from pathlib import Path
from typing import Any, ClassVar, Final

from traitlets import Bool, TraitError
from traitlets.config import Config, SingletonConfigurable

from unxt._src.config import (
    AbstractUnxtConfig,
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
    _override_config_keys: ClassVar[dict[str, frozenset[str]]] = (
        PARAMETRIC_OVERRIDE_CONFIG_KEYS
    )

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
