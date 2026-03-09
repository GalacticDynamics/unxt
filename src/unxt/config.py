"""Configuration for unxt using traitlets.

TOML configuration is discovered automatically from ``pyproject.toml`` at import
time.
"""

__all__ = ("config", "UnxtConfig", "QuantityReprConfig", "QuantityStrConfig")

import threading
import tomllib
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, ClassVar, Final

from traitlets import Bool, Enum, TraitError, Union
from traitlets.config import Config, Configurable, SingletonConfigurable
from traitlets.traitlets import Int


class LocalConfigurable(Configurable):
    """Base class for config objects that support thread-local overrides.

    This class provides a mechanism for temporary, thread-local configuration
    changes that can be used in nested contexts (e.g., within Quantity.__repr__()).
    """

    # Thread-local storage for context manager overrides
    _local: threading.local


@dataclass(frozen=True, slots=True)
class _NestedConfigContext:
    """Context manager for temporary config changes on nested config objects.

    This class should not be instantiated directly. Use the override() method
    on nested config objects (e.g., config.quantity_repr.override(...)).
    """

    config: LocalConfigurable  # The nested config object (e.g., QuantityReprConfig)
    overrides: dict[str, Any]  # {attr: value}

    def __enter__(self) -> LocalConfigurable:
        """Enter context, pushing overrides onto thread-local stack."""
        # Initialize thread-local stack if needed
        if not hasattr(self.config._local, "stack"):  # noqa: SLF001
            self.config._local.stack = []  # noqa: SLF001

        # Push overrides onto the stack
        self.config._local.stack.append(self.overrides)  # noqa: SLF001
        return self.config

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context, popping overrides from thread-local stack."""
        # Pop overrides from the stack
        if hasattr(self.config._local, "stack") and self.config._local.stack:  # noqa: SLF001
            self.config._local.stack.pop()  # noqa: SLF001


# ============================================================================
# Quantity `__repr__`

QUANTITY_REPR_CONFIG_KEYS: Final = frozenset(
    {"short_arrays", "use_short_name", "named_unit", "include_params", "indent"}
)


class QuantityReprConfig(LocalConfigurable):
    """Configuration for Quantity.__repr__() display options.

    This controls how Quantity objects are displayed in ``repr()``.

    Attributes
    ----------
    short_arrays : bool | Literal["compact"]
        Controls how arrays are displayed in repr. Options:
        - "compact": Show array values without Array wrapper
        - `True`: Show short array summary (shape/dtype)
        - `False`: Show full array representation
        Default: `False`
    use_short_name : bool
        If True and a class has a `short_name` attribute, use the short
        name instead of the full class name in repr. Default: False
    named_unit : bool
        If `True`, display unit as a named argument `unit='m'`.
        If `False`, display unit as a positional argument `'m'`.
        Default: `True`
    include_params : bool
        If `True`, include type parameters in repr for parametric quantities.
        If `False`, omit type parameters from repr. Default: `False`

    Examples
    --------
    >>> import unxt as u
    >>> with u.config.quantity_repr.override(
    ...     short_arrays="compact", use_short_name=True
    ... ):
    ...     q = u.Quantity([1, 2, 3], "m")
    ...     print(repr(q))
    Q([1, 2, 3], unit='m')

    """

    # Thread-local storage for context manager overrides
    _local: threading.local = threading.local()

    short_arrays: ClassVar[object] = Union(
        [Bool(), Enum(("compact",))],
        default_value=False,
        help=(
            "Controls array display in repr. "
            "Options: 'compact' (values only), "
            "True (shape/dtype), False (full repr)"
        ),
    ).tag(config=True)

    use_short_name: ClassVar[object] = Bool(
        default_value=False,
        help="Use short class name if available (e.g., Q instead of Quantity)",
    ).tag(config=True)

    named_unit: ClassVar[object] = Bool(
        default_value=True,
        help="Display unit as named argument (unit='m') vs positional ('m')",
    ).tag(config=True)

    include_params: ClassVar[object] = Bool(
        default_value=False,
        help="Include type parameters in repr for parametric quantities",
    ).tag(config=True)

    indent: ClassVar[object] = Int(
        default_value=4, help="Indentation level for nested structures in repr"
    ).tag(config=True)

    def __getattribute__(self, name: str) -> Any:
        """Get attribute, checking thread-local overrides first."""
        if name in QUANTITY_REPR_CONFIG_KEYS:
            try:
                local = object.__getattribute__(self, "_local")
                # Return the most recent override for this attribute
                if hasattr(local, "stack") and local.stack:
                    for overrides in reversed(local.stack):
                        if name in overrides:
                            return overrides[name]
            except AttributeError:
                pass

        return object.__getattribute__(self, name)

    def override(
        self, cfg: Config | None = None, /, **kwargs: Any
    ) -> "_NestedConfigContext":
        """Create a context manager for temporary config changes.

        Parameters
        ----------
        cfg : traitlets.config.Config, optional
            A traitlets Config object with settings for this config class.
            Cannot be used together with keyword arguments.
        **kwargs
            Configuration options to set temporarily (e.g.,
            short_arrays="compact").  Cannot be used together with cfg
            parameter.

        Returns
        -------
        _NestedConfigContext
            A context manager that applies the config changes on entry
            and restores previous values on exit.

        Raises
        ------
        ValueError
            If both cfg and keyword arguments are provided.

        Examples
        --------
        Using keyword arguments:

        >>> import unxt as u
        >>> with u.config.quantity_repr.override(
        ...     short_arrays="compact", use_short_name=True
        ... ):
        ...     q = u.Quantity([1, 2, 3], "m")
        ...     print(repr(q))
        Q([1, 2, 3], unit='m')

        Using a Config object:

        >>> from traitlets.config import Config
        >>> cfg = Config()
        >>> cfg.QuantityReprConfig.short_arrays = "compact"
        >>> cfg.QuantityReprConfig.use_short_name = True
        >>> with u.config.quantity_repr.override(cfg):
        ...     q = u.Quantity([1, 2, 3], "m")
        ...     print(repr(q))
        Q([1, 2, 3], unit='m')

        >>> print(str(q))
        Quantity['length']([1, 2, 3], unit='m')

        """
        if cfg is not None and kwargs:
            msg = "Cannot specify both cfg and keyword arguments to override()"
            raise ValueError(msg)

        if kwargs:
            unknown_keys = set(kwargs) - QUANTITY_REPR_CONFIG_KEYS
            if unknown_keys:
                valid_keys = ", ".join(sorted(QUANTITY_REPR_CONFIG_KEYS))
                unknown = ", ".join(sorted(unknown_keys))
                msg = (
                    f"Unknown QuantityReprConfig override option(s): {unknown}. "
                    f"Valid options are: {valid_keys}"
                )
                raise ValueError(msg)

        if cfg is not None:
            # Create a temporary instance, apply the config to it, and read the
            # resolved trait values. This ensures LazyConfigValue objects are
            # properly resolved to their actual values.
            temp_instance = self.__class__(config=cfg)
            overrides = {}
            for trait_name in self.trait_names():
                overrides[trait_name] = getattr(temp_instance, trait_name)
            kwargs = overrides

        return _NestedConfigContext(self, kwargs)


# ============================================================================
# Quantity `__str__`

QUANTITY_STR_CONFIG_KEYS: Final = frozenset(
    {"short_arrays", "use_short_name", "named_unit", "indent", "include_params"}
)

UNXT_OVERRIDE_CONFIG_KEYS: Final = {
    "quantity_repr": QUANTITY_REPR_CONFIG_KEYS,
    "quantity_str": QUANTITY_STR_CONFIG_KEYS,
}


class QuantityStrConfig(LocalConfigurable):
    """Configuration for Quantity.__str__() display options.

    This controls how Quantity objects are displayed in ``str()``.

    Attributes
    ----------
    short_arrays : bool | Literal["compact"]
        Controls how arrays are displayed in str. Options:
        - "compact": Show array values without Array wrapper
        - `True`: Show short array summary (shape/dtype)
        - `False`: Show full array representation
        Default: "compact"
    use_short_name : bool
        If True and a class has a `short_name` attribute, use the short
        name instead of the full class name in str. Default: False
    named_unit : bool
        If True, display unit as a named argument `unit='m'`.
        If False, display unit as a positional argument `'m'`.
        Default: True
    include_params : bool
        If True, include type parameters in str for parametric quantities.
        If False, omit type parameters from str. Default: `True`
    indent : int
        Indentation width for nested structures in str representation.
        Default: 4

    Examples
    --------
    >>> import unxt as u
    >>> with u.config.quantity_str.override(
    ...     short_arrays=True, use_short_name=True, include_params=False
    ... ):
    ...     q = u.Quantity([1, 2, 3], "m")
    ...     print(str(q))
    Q(i32[3], unit='m')

    >>> print(str(q))
    Quantity['length']([1, 2, 3], unit='m')

    """

    # Thread-local storage for context manager overrides
    _local: threading.local = threading.local()

    short_arrays: ClassVar[object] = Union(
        [Bool(), Enum(("compact",))],
        default_value="compact",
        help=(
            "Controls array display in str. "
            "Options: 'compact' (values only), "
            "True (shape/dtype), False (full str)"
        ),
    ).tag(config=True)

    use_short_name: ClassVar[object] = Bool(
        default_value=False,
        help="Use short class name if available (e.g., Q instead of Quantity)",
    ).tag(config=True)

    named_unit: ClassVar[object] = Bool(
        default_value=True,
        help="Display unit as named argument (unit='m') vs positional ('m')",
    ).tag(config=True)

    include_params: ClassVar[object] = Bool(
        default_value=True,
        help="Include type parameters in str for parametric quantities",
    ).tag(config=True)

    indent: ClassVar[object] = Int(
        default_value=4, help="Indentation level for nested structures in str"
    ).tag(config=True)

    def __getattribute__(self, name: str) -> Any:
        """Get attribute, checking thread-local overrides first."""
        if name in QUANTITY_STR_CONFIG_KEYS:
            try:
                local = object.__getattribute__(self, "_local")
                if hasattr(local, "stack") and local.stack:
                    for overrides in reversed(local.stack):
                        if name in overrides:
                            return overrides[name]
            except AttributeError:
                pass

        return object.__getattribute__(self, name)

    def override(
        self, cfg: Config | None = None, /, **kwargs: Any
    ) -> "_NestedConfigContext":
        """Create a context manager for temporary config changes.

        Parameters
        ----------
        cfg : traitlets.config.Config, optional
            A traitlets Config object with settings for this config class.
            Cannot be used together with keyword arguments.
        **kwargs
            Configuration options to set temporarily.
            Cannot be used together with cfg parameter.

        Returns
        -------
        _NestedConfigContext
            A context manager that applies the config changes on entry
            and restores previous values on exit.

        Raises
        ------
        ValueError
            If both cfg and keyword arguments are provided.

        """
        if cfg is not None and kwargs:
            msg = "Cannot specify both cfg and keyword arguments to override()"
            raise ValueError(msg)

        if kwargs:
            unknown_keys = set(kwargs) - QUANTITY_STR_CONFIG_KEYS
            if unknown_keys:
                valid_keys = ", ".join(sorted(QUANTITY_STR_CONFIG_KEYS))
                unknown = ", ".join(sorted(unknown_keys))
                msg = (
                    f"Unknown QuantityStrConfig override option(s): {unknown}. "
                    f"Valid options are: {valid_keys}"
                )
                raise ValueError(msg)

        if cfg is not None:
            # Create a temporary instance, apply the config to it, and read the
            # resolved trait values. This ensures LazyConfigValue objects are
            # properly resolved to their actual values.
            temp_instance = self.__class__(config=cfg)
            overrides = {}
            for trait_name in self.trait_names():
                overrides[trait_name] = getattr(temp_instance, trait_name)
            kwargs = overrides

        return _NestedConfigContext(self, kwargs)


# ============================================================================
# Unxt configuration


class UnxtConfig(SingletonConfigurable):
    """Configuration for unxt display and printing options.

    This is a singleton configuration object that controls how quantities
    and other objects are displayed.

    The config uses a hierarchical structure with separate config objects
    for different components:

    - ``quantity_repr``: Configuration for Quantity.__repr__()
    - ``quantity_str``: Configuration for Quantity.__str__() (future)

    The config can be used as a context manager for temporary, thread-local
    configuration changes that are automatically restored on exit.

    Examples
    --------
    >>> import unxt as u

    Access nested config

    >>> u.config.quantity_repr.short_arrays
    False

    Modify config globally

    >>> u.config.quantity_repr.short_arrays = "compact"  # doctest: +SKIP
    >>> u.config.quantity_repr.use_short_name = True  # doctest: +SKIP

    Use as context manager for temporary changes (thread-local)

    >>> with u.config.override(
    ...     quantity_repr__short_arrays="compact", quantity_repr__use_short_name=True
    ... ):
    ...     q = u.Quantity([1, 2, 3], "m")
    ...     print(repr(q))
    Q([1, 2, 3], unit='m')

    Config restored after exiting context

    >>> print(repr(q))
    Quantity(Array([1, 2, 3], dtype=int32), unit='m')

    """

    # Configurable classes that are part of this config hierarchy
    classes: ClassVar[list[type]] = [QuantityReprConfig, QuantityStrConfig]

    def __init__(self, **kwargs: Any) -> None:
        """Initialize UnxtConfig with nested config instances."""
        super().__init__(**kwargs)
        # Initialize child configs with parent config for inheritance
        self.quantity_repr = QuantityReprConfig(config=self.config, parent=self)
        self.quantity_str = QuantityStrConfig(config=self.config, parent=self)

    def override(self, **kwargs: Any) -> "_ConfigContext":
        """Create a context manager for temporary config changes.

        Parameters
        ----------
        **kwargs
            Configuration options to set temporarily. Use double-underscore
            notation for nested configs, e.g., quantity_repr__short_arrays="compact".

        Returns
        -------
        _ConfigContext
            A context manager that applies the config changes on entry
            and restores previous values on exit.

        Raises
        ------
        ValueError
            If an unknown configuration option is provided.

        Examples
        --------
        >>> import unxt as u
        >>> with u.config.override(quantity_repr__short_arrays=True):
        ...     q = u.Quantity([1.0, 2.0, 3.0], "m")
        ...     print(repr(q))  # Uses compact display
        Quantity(f32[3], unit='m')

        Config restored to previous value

        >>> print(repr(q))
        Quantity(Array([1., 2., 3.], dtype=float32), unit='m')

        """
        # Parse nested config overrides (e.g., quantity_repr__short_arrays)
        parsed_overrides: dict[str, dict[str, Any]] = {}
        for key, value in kwargs.items():
            if "__" in key:
                config_name, attr_name = key.split("__", 1)
                if config_name not in UNXT_OVERRIDE_CONFIG_KEYS:
                    valid_configs = ", ".join(sorted(UNXT_OVERRIDE_CONFIG_KEYS))
                    msg = (
                        "Unknown config section "
                        f"'{config_name}' in override key '{key}'. "
                        f"Valid sections are: {valid_configs}"
                    )
                    raise ValueError(msg)

                valid_attrs = UNXT_OVERRIDE_CONFIG_KEYS[config_name]
                if attr_name not in valid_attrs:
                    valid_options = ", ".join(sorted(valid_attrs))
                    msg = (
                        "Unknown option "
                        f"'{attr_name}' for config section '{config_name}' "
                        f"in override key '{key}'. Valid options are: {valid_options}"
                    )
                    raise ValueError(msg)

                if config_name not in parsed_overrides:
                    parsed_overrides[config_name] = {}
                parsed_overrides[config_name][attr_name] = value
            else:
                msg = (
                    f"Config override '{key}' must use double-underscore notation "
                    "(e.g., 'quantity_repr__short_arrays')"
                )
                raise ValueError(msg)

        return _ConfigContext(self, parsed_overrides)


@dataclass(slots=True)
class _ConfigContext:
    """Context manager for temporary config changes.

    This class should not be instantiated directly. Use
    `config.override(**kwargs)` to create a context manager.
    """

    config: UnxtConfig
    overrides: dict[str, dict[str, Any]]  # {config_name: {attr: value}}
    _stack: ExitStack = field(init=False, repr=False)

    def __enter__(self) -> UnxtConfig:
        """Enter context, applying thread-local config overrides."""
        self._stack = ExitStack()
        for config_name, attrs in self.overrides.items():
            # Get the nested config object
            nested_config = getattr(self.config, config_name)
            # Delegate to nested config contexts to preserve thread-local behavior.
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


def _find_pyproject(start: Path, /) -> Path | None:
    """Find nearest ``pyproject.toml`` from ``start`` up to filesystem root."""
    for directory in (start, *start.parents):
        candidate = directory / "pyproject.toml"
        if candidate.is_file():
            return candidate
    return None


def _load_toml_config_from_pyproject(path: Path, /) -> Config:
    """Load ``[tool.unxt]`` section from a ``pyproject.toml`` file.

    Supports nested sections like:
        [tool.unxt.quantity.repr]
        short_arrays = "compact"
        use_short_name = true
    """
    with path.open("rb") as f:
        data = tomllib.load(f)

    tool = data.get("tool")
    if not isinstance(tool, dict):
        return Config()

    unxt_cfg = tool.get("unxt")
    if not isinstance(unxt_cfg, dict):
        return Config()

    # Build config from nested sections
    config = Config()  # pylint: disable=redefined-outer-name

    # Handle [tool.unxt.quantity.repr] -> QuantityReprConfig
    if "quantity" in unxt_cfg and isinstance(unxt_cfg["quantity"], dict):
        quantity_cfg = unxt_cfg["quantity"]

        if "repr" in quantity_cfg and isinstance(quantity_cfg["repr"], dict):
            config["QuantityReprConfig"] = Config(quantity_cfg["repr"])

        if "str" in quantity_cfg and isinstance(quantity_cfg["str"], dict):
            config["QuantityStrConfig"] = Config(quantity_cfg["str"])

    return config


def _auto_load_project_toml_config(cfg: UnxtConfig, /) -> None:  # noqa: C901
    """Auto-load nearest project TOML config without raising import-time errors."""
    pyproject = _find_pyproject(Path.cwd())
    if pyproject is None:
        return

    try:
        loaded = _load_toml_config_from_pyproject(pyproject)
    except (OSError, tomllib.TOMLDecodeError, TypeError):
        # Never fail import because project config is missing or malformed.
        return

    if not loaded:
        return

    # Apply nested config values directly to the nested config instances
    if "QuantityReprConfig" in loaded:
        for k, v in loaded["QuantityReprConfig"].items():
            if k not in QUANTITY_REPR_CONFIG_KEYS:
                continue
            try:
                setattr(cfg.quantity_repr, k, v)
            except TraitError:
                continue

    if "QuantityStrConfig" in loaded:
        for k, v in loaded["QuantityStrConfig"].items():
            if k not in QUANTITY_STR_CONFIG_KEYS:
                continue
            try:
                setattr(cfg.quantity_str, k, v)
            except TraitError:
                continue


# Create the global singleton instance
config = UnxtConfig.instance()
_auto_load_project_toml_config(config)
