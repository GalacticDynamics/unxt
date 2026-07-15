"""Configuration for unxt using traitlets.

TOML configuration is discovered automatically from ``pyproject.toml`` at import
time.
"""

__all__ = (
    "AbstractUnxtConfig",
    "QuantityReprConfig",
    "QuantityStrConfig",
    "UnxtConfig",
    "config",
)

import contextlib
import threading
import tomllib
import warnings
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
    changes that can be used in nested contexts (e.g., within a quantity's
    ``__repr__()``).
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
    {"short_arrays", "use_short_name", "named_unit", "indent"}
)


class QuantityReprConfig(LocalConfigurable):
    """Configuration for quantity ``__repr__()`` display options.

    This controls how quantity objects are displayed in ``repr()``. It is
    consumed by ``AbstractQuantity.__repr__`` and so applies to every quantity
    class, including the default ``Quantity`` (and ``ParametricQuantity`` from
    the ``unxts.parametric`` package).

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

    Examples
    --------
    >>> import unxt as u
    >>> with u.config.quantity_repr.override(
    ...     short_arrays="compact", use_short_name=True
    ... ):
    ...     q = u.Q([1, 2, 3], "m")
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
        help=(
            "Use a class's short name if available (e.g. Quantity -> Q, "
            "ParametricQuantity -> PQ)"
        ),
    ).tag(config=True)

    named_unit: ClassVar[object] = Bool(
        default_value=True,
        help="Display unit as named argument (unit='m') vs positional ('m')",
    ).tag(config=True)

    indent: ClassVar[object] = Int(
        default_value=4, help="Indentation level for nested structures in repr"
    ).tag(config=True)

    def __getattribute__(self, name: str) -> Any:
        """Get attribute, checking thread-local overrides first."""
        if name in QUANTITY_REPR_CONFIG_KEYS:
            with contextlib.suppress(AttributeError):
                local = object.__getattribute__(self, "_local")
                # Return the most recent override for this attribute
                if hasattr(local, "stack") and local.stack:
                    for overrides in reversed(local.stack):
                        if name in overrides:
                            return overrides[name]

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
        ...     q = u.Q([1, 2, 3], "m")
        ...     print(repr(q))
        Q([1, 2, 3], unit='m')

        Using a Config object:

        >>> from traitlets.config import Config
        >>> cfg = Config()
        >>> cfg.QuantityReprConfig.short_arrays = "compact"
        >>> cfg.QuantityReprConfig.use_short_name = True
        >>> with u.config.quantity_repr.override(cfg):
        ...     q = u.Q([1, 2, 3], "m")
        ...     print(repr(q))
        Q([1, 2, 3], unit='m')

        >>> print(str(q))
        Quantity([1, 2, 3], unit='m')

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

            # Validate and resolve values through traitlets immediately so
            # override() fails fast with clear errors.
            temp_instance = self.__class__()
            validated_kwargs: dict[str, Any] = {}
            for key, value in kwargs.items():
                try:
                    setattr(temp_instance, key, value)
                except TraitError as e:
                    msg = (
                        "Invalid value for QuantityReprConfig override option "
                        f"'{key}': {value!r}"
                    )
                    raise ValueError(msg) from e
                # Bypass thread-local override lookup when reading back the
                # resolved trait value from the temporary instance.
                validated_kwargs[key] = object.__getattribute__(temp_instance, key)
            kwargs = validated_kwargs

        if cfg is not None:
            # Create a temporary instance, apply the config to it, and read the
            # resolved trait values. This ensures LazyConfigValue objects are
            # properly resolved to their actual values.
            temp_instance = self.__class__(config=cfg)
            overrides = {}
            for trait_name in self.trait_names():
                overrides[trait_name] = object.__getattribute__(
                    temp_instance, trait_name
                )
            kwargs = overrides

        return _NestedConfigContext(self, kwargs)


# ============================================================================
# Quantity `__str__`

QUANTITY_STR_CONFIG_KEYS: Final = frozenset(
    {"short_arrays", "use_short_name", "named_unit", "indent"}
)

UNXT_OVERRIDE_CONFIG_KEYS: Final = {
    "quantity_repr": QUANTITY_REPR_CONFIG_KEYS,
    "quantity_str": QUANTITY_STR_CONFIG_KEYS,
}


class QuantityStrConfig(LocalConfigurable):
    """Configuration for quantity ``__str__()`` display options.

    This controls how quantity objects are displayed in ``str()``. It is
    consumed by ``AbstractQuantity.__str__`` and so applies to every quantity
    class, including the default ``Quantity`` (and ``ParametricQuantity`` from
    the ``unxts.parametric`` package).

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
    indent : int
        Indentation width for nested structures in str representation.
        Default: 4

    Examples
    --------
    >>> import unxt as u
    >>> with u.config.quantity_str.override(short_arrays=True, use_short_name=True):
    ...     q = u.Q([1, 2, 3], "m")
    ...     print(str(q))
    Q(i32[3], unit='m')

    >>> print(str(q))
    Quantity([1, 2, 3], unit='m')

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
        help=(
            "Use a class's short name if available (e.g. Quantity -> Q, "
            "ParametricQuantity -> PQ)"
        ),
    ).tag(config=True)

    named_unit: ClassVar[object] = Bool(
        default_value=True,
        help="Display unit as named argument (unit='m') vs positional ('m')",
    ).tag(config=True)

    indent: ClassVar[object] = Int(
        default_value=4, help="Indentation level for nested structures in str"
    ).tag(config=True)

    def __getattribute__(self, name: str) -> Any:
        """Get attribute, checking thread-local overrides first."""
        if name in QUANTITY_STR_CONFIG_KEYS:
            with contextlib.suppress(AttributeError):
                local = object.__getattribute__(self, "_local")
                if hasattr(local, "stack") and local.stack:
                    for overrides in reversed(local.stack):
                        if name in overrides:
                            return overrides[name]

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

            # Validate and resolve values through traitlets immediately so
            # override() fails fast with clear errors.
            temp_instance = self.__class__()
            validated_kwargs: dict[str, Any] = {}
            for key, value in kwargs.items():
                try:
                    setattr(temp_instance, key, value)
                except TraitError as e:
                    msg = (
                        "Invalid value for QuantityStrConfig override option "
                        f"'{key}': {value!r}"
                    )
                    raise ValueError(msg) from e
                # Bypass thread-local override lookup when reading back the
                # resolved trait value from the temporary instance.
                validated_kwargs[key] = object.__getattribute__(temp_instance, key)
            kwargs = validated_kwargs

        if cfg is not None:
            # Create a temporary instance, apply the config to it, and read the
            # resolved trait values. This ensures LazyConfigValue objects are
            # properly resolved to their actual values.
            temp_instance = self.__class__(config=cfg)
            overrides = {}
            for trait_name in self.trait_names():
                overrides[trait_name] = object.__getattribute__(
                    temp_instance, trait_name
                )
            kwargs = overrides

        return _NestedConfigContext(self, kwargs)


# ============================================================================
# Unxt configuration


class AbstractUnxtConfig:
    """Mixin for hierarchical, unxt-style config singletons.

    A concrete config inherits this alongside
    :class:`~traitlets.config.SingletonConfigurable`, declares the sections it
    accepts via ``_override_config_keys`` (``{section name: valid option
    names}``), and creates its nested config objects in ``__init__``. This mixin
    provides the shared top-level ``override()`` context manager, so packages
    that mirror this config (e.g. ``unxts.parametric``) can reuse the same
    machinery -- their singletons are accepted by :class:`_ConfigContext`
    without reimplementation.

    It is a plain mixin (not itself a ``SingletonConfigurable``) so that each
    concrete config keeps its own independent singleton instance.
    """

    # {section_name: frozenset of valid option names}; set by subclasses.
    _override_config_keys: ClassVar[dict[str, frozenset[str]]] = {}

    def override(self, **kwargs: Any) -> "_ConfigContext":
        """Create a context manager for temporary config changes.

        Use double-underscore notation for nested configs, e.g.
        ``quantity_repr__short_arrays="compact"``.

        Raises
        ------
        ValueError
            If an unknown configuration option is provided.

        """
        # Parse nested config overrides (e.g., quantity_repr__short_arrays)
        parsed_overrides: dict[str, dict[str, Any]] = {}
        for key, value in kwargs.items():
            if "__" in key:
                config_name, attr_name = key.split("__", 1)
                if config_name not in self._override_config_keys:
                    valid_configs = ", ".join(sorted(self._override_config_keys))
                    msg = (
                        "Unknown config section "
                        f"'{config_name}' in override key '{key}'. "
                        f"Valid sections are: {valid_configs}"
                    )
                    raise ValueError(msg)

                valid_attrs = self._override_config_keys[config_name]
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


class UnxtConfig(AbstractUnxtConfig, SingletonConfigurable):
    """Configuration for unxt display and printing options.

    This is a singleton configuration object that controls how quantities
    and other objects are displayed.

    The config uses a hierarchical structure with separate config objects
    for different components:

    - ``quantity_repr``: Configuration for Quantity.__repr__()
    - ``quantity_str``: Configuration for Quantity.__str__()

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
    ...     q = u.Q([1, 2, 3], "m")
    ...     print(repr(q))
    Q([1, 2, 3], unit='m')

    Config restored after exiting context

    >>> print(repr(q))
    Quantity(Array([1, 2, 3], dtype=int32), unit='m')

    """

    # Configurable classes that are part of this config hierarchy
    classes: ClassVar[list[type]] = [QuantityReprConfig, QuantityStrConfig]
    _override_config_keys: ClassVar[dict[str, frozenset[str]]] = (
        UNXT_OVERRIDE_CONFIG_KEYS
    )

    def __init__(self, **kwargs: Any) -> None:
        """Initialize UnxtConfig with nested config instances."""
        super().__init__(**kwargs)
        # Initialize child configs with parent config for inheritance
        self.quantity_repr = QuantityReprConfig(config=self.config, parent=self)
        self.quantity_str = QuantityStrConfig(config=self.config, parent=self)

    def reload(self) -> None:
        """Re-read configuration from the nearest project ``pyproject.toml``.

        The ``[tool.unxts.unxt]`` section is auto-loaded once at import, using
        the current working directory at that moment. If ``unxt`` was imported
        before the process changed into the project directory (common in
        notebooks and test runners), that section is missed; call this to pick
        it up. Values already set on the config are overwritten by the file;
        unset ones are left as-is.

        Examples
        --------
        >>> import unxt as u
        >>> u.config.reload()  # re-read pyproject.toml from the current cwd

        """
        _auto_load_project_toml_config(self)


@dataclass(slots=True)
class _ConfigContext:
    """Context manager for temporary config changes.

    This class should not be instantiated directly. Use
    `config.override(**kwargs)` to create a context manager.
    """

    config: AbstractUnxtConfig
    overrides: dict[str, dict[str, Any]]  # {config_name: {attr: value}}
    _stack: contextlib.ExitStack = field(init=False, repr=False)

    def __enter__(self) -> AbstractUnxtConfig:
        """Enter context, applying thread-local config overrides."""
        self._stack = contextlib.ExitStack()
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


# Mapping from TOML path to Config class name
_TOML_PATH_TO_CONFIG_CLASS: Final = {
    ("quantity", "repr"): "QuantityReprConfig",
    ("quantity", "str"): "QuantityStrConfig",
}


def _walk_toml_config(
    data: dict[str, Any],
    path: tuple[str, ...] = (),
    *,
    path_to_class: dict[tuple[str, ...], str] | None = None,
) -> dict[str, Config]:
    r"""Walk nested TOML dict and build Config objects for known paths.

    This function recursively walks the parsed TOML structure and creates
    Config objects for recognized configuration paths.

    Parameters
    ----------
    data : dict
        Nested dictionary from parsed TOML (e.g., tool.unxts.unxt section)
    path : tuple of str
        Current path in the nested structure (e.g., ("quantity", "repr"))
    path_to_class : dict
        Mapping from TOML sub-path to Config class name. Defaults to unxt's
        own mapping; other packages (e.g. ``unxts.parametric``) pass their own.

    Returns
    -------
    dict[str, Config]
        Mapping from Config class names to Config objects
        (e.g., {"QuantityReprConfig": Config({"short_arrays": "compact"})})

    Examples
    --------
    >>> import tomllib
    >>> from pathlib import Path
    >>> toml = '[tool.unxts.unxt.quantity.repr]\nshort_arrays = "compact"\n'
    >>> data = tomllib.loads(toml)["tool"]["unxts"]["unxt"]
    >>> configs = _walk_toml_config(data)
    >>> "QuantityReprConfig" in configs
    True

    """
    if path_to_class is None:
        path_to_class = _TOML_PATH_TO_CONFIG_CLASS

    result: dict[str, Config] = {}

    for key, value in data.items():
        current_path = (*path, key)

        # Check if this path maps to a known Config class
        if current_path in path_to_class:
            config_class_name = path_to_class[current_path]
            if isinstance(value, dict):
                result[config_class_name] = Config(value)
        elif isinstance(value, dict):
            # Continue walking nested dicts
            nested_result = _walk_toml_config(
                value, current_path, path_to_class=path_to_class
            )
            result.update(nested_result)

    return result


def _load_toml_config_from_pyproject(
    path: Path,
    /,
    *,
    tool_path: tuple[str, ...] = ("unxts", "unxt"),
    path_to_class: dict[tuple[str, ...], str] | None = None,
) -> Config:
    """Load a ``[tool.<...>]`` section from a ``pyproject.toml`` file.

    Supports both nested sections and dotted keys:

    Nested sections:
        [tool.unxts.unxt.quantity.repr]
        short_arrays = "compact"
        use_short_name = true

    Dotted keys:
        [tool.unxts.unxt]
        quantity.repr.short_arrays = "compact"
        quantity.repr.use_short_name = true

    Both syntaxes produce identical configuration.

    Parameters
    ----------
    path : Path
        Path to pyproject.toml file
    tool_path : tuple of str
        Keys under ``[tool]`` to navigate into (e.g. ``("unxts", "unxt")`` or
        ``("unxts", "parametric")``). Defaults to unxt's own section,
        ``[tool.unxts.unxt]``.
    path_to_class : dict
        Mapping from TOML sub-path to Config class name (passed through to
        :func:`_walk_toml_config`).

    Returns
    -------
    Config
        Traitlets Config object with nested configuration

    """
    if path_to_class is None:
        path_to_class = _TOML_PATH_TO_CONFIG_CLASS

    with path.open("rb") as f:
        data = tomllib.load(f)

    # Navigate to the tool.<tool_path> section
    section: Any = data.get("tool")
    for key in tool_path:
        if not isinstance(section, dict):
            return Config()
        section = section.get(key)
    if not isinstance(section, dict):
        return Config()

    # Walk the nested structure and build Config objects
    config_dict = _walk_toml_config(section, path_to_class=path_to_class)

    # Build final Config object
    config = Config()  # pylint: disable=redefined-outer-name
    for class_name, class_config in config_dict.items():
        config[class_name] = class_config

    return config


# Mapping from Config class name to (config object, valid keys)
_CONFIG_CLASS_TO_INSTANCE: Final[dict[str, tuple[Any, frozenset[str]]]] = {}


def _initialize_config_mapping(cfg: UnxtConfig) -> None:
    """Initialize the mapping from config class names to instances.

    This must be called after creating the UnxtConfig instance.
    """
    _CONFIG_CLASS_TO_INSTANCE["QuantityReprConfig"] = (
        cfg.quantity_repr,
        QUANTITY_REPR_CONFIG_KEYS,
    )
    _CONFIG_CLASS_TO_INSTANCE["QuantityStrConfig"] = (
        cfg.quantity_str,
        QUANTITY_STR_CONFIG_KEYS,
    )


def _warn_if_legacy_unxt_config(path: Path, /) -> None:
    """Warn if the deprecated ``[tool.unxt]`` pyproject section is present.

    Before the namespace rename (unxt#685), configuration lived under
    ``[tool.unxt]``; it is now read from ``[tool.unxts.unxt]``. A leftover
    ``[tool.unxt]`` section is silently ignored, so warn the user to migrate.
    """
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return

    tool = data.get("tool")
    if isinstance(tool, dict) and isinstance(tool.get("unxt"), dict):
        warnings.warn(
            "The '[tool.unxt]' pyproject.toml section is deprecated and ignored; "
            "unxt now reads its configuration from '[tool.unxts.unxt]'. Move your "
            "settings there.",
            DeprecationWarning,
            stacklevel=2,
        )


def _auto_load_project_toml_config(cfg: UnxtConfig, /) -> None:
    """Auto-load nearest project TOML config without raising import-time errors.

    This function:
    1. Searches for nearest pyproject.toml from cwd
    2. Loads [tool.unxts.unxt] configuration
    3. Applies valid settings to config instances
    4. Silently ignores invalid values or missing files

    """
    pyproject = _find_pyproject(Path.cwd())
    if pyproject is None:
        return

    _warn_if_legacy_unxt_config(pyproject)

    try:
        loaded = _load_toml_config_from_pyproject(pyproject)
    except (OSError, tomllib.TOMLDecodeError, TypeError, KeyError):
        # Never fail import because project config is missing or malformed.
        return

    if not loaded:
        return

    # Apply config values using the mapping
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
config = UnxtConfig.instance()
_initialize_config_mapping(config)
_auto_load_project_toml_config(config)
