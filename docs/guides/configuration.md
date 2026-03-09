# Configuration

`unxt` provides a hierarchical configuration system for customizing how quantities are displayed in `repr()` and `str()` representations. This is powered by [traitlets](https://traitlets.readthedocs.io/).

## Quick Start

<!-- invisible-code-block: python
import unxt as u
-->

The configuration system is organized hierarchically with separate configs for different display methods:

- `u.config.quantity_repr` - Options for `Quantity.__repr__()`
- `u.config.quantity_str` - Options for `Quantity.__str__()`

Modify configuration options directly:

<!-- skip: start -->

```{code-block} python
# Configure repr display
u.config.quantity_repr.short_arrays = "compact"
u.config.quantity_repr.use_short_name = True
u.config.quantity_repr.named_unit = False

q = u.Quantity([1, 2, 3], "m")
print(repr(q))  # Q([1, 2, 3], 'm')
```

<!-- skip: end -->

For temporary changes, use the context manager API with either the root config or nested configs:

```{code-block} python
>>> # Option 1: Use root config with double-underscore notation
>>> with u.config.override(quantity_repr__short_arrays="compact", quantity_repr__use_short_name=True):
...     q = u.Quantity([1.0, 2.0, 3.0], "m")
...     print(repr(q))
Q([1., 2., 3.], unit='m')
```

```{code-block} python
>>> # Option 2: Use nested config directly (cleaner for many options)
>>> with u.config.quantity_repr.override(short_arrays="compact", use_short_name=True):
...     q = u.Quantity([1.0, 2.0, 3.0], "m")
...     print(repr(q))
Q([1., 2., 3.], unit='m')
```

## Configuration Hierarchy

The `unxt.config` object has the following structure:

- **`config.quantity_repr`** (`QuantityReprConfig`) - Controls `repr()` display
- **`config.quantity_str`** (`QuantityStrConfig`) - Controls `str()` display

Each nested config has its own set of options and supports independent overrides.

## Quantity Repr Options

Options for controlling `Quantity.__repr__()` via `u.config.quantity_repr`.

### `short_arrays`

Controls how arrays are displayed in `repr()`.

- **Type**: `bool | Literal["compact"]`
- **Default**: `False`

Options:

- `False`: full array representation
- `True`: compact shape and dtype summary
- `"compact"`: values without the `Array(...)` wrapper

<!-- skip: start -->

```{code-block} python
>>> q = u.Quantity([1.0, 2.0, 3.0], "m")

>>> u.config.quantity_repr.short_arrays = False
>>> print(repr(q))  # Quantity(Array([1., 2., 3.], dtype=float32), unit='m')

>>> u.config.quantity_repr.short_arrays = True
>>> print(repr(q))  # Quantity(f32[3], unit='m')

>>> u.config.quantity_repr.short_arrays = "compact"
>>> print(repr(q))  # Quantity([1., 2., 3.], unit='m')
```

<!-- skip: end -->

### `use_short_name`

Use class short names where available (for example, `Quantity` -> `Q`).

- **Type**: `bool`
- **Default**: `False`

```{code-block} python
>>> q = u.Quantity(1.0, "m")

>>> # Default behavior (use_short_name=False)
>>> with u.config.quantity_repr.override(use_short_name=False):
...     print(repr(q))
Quantity(Array(1., dtype=float32...), unit='m')

>>> # With use_short_name=True
>>> with u.config.quantity_repr.override(use_short_name=True):
...     print(repr(q))
Q(Array(1., dtype=float32...), unit='m')
```

### `named_unit`

Display units as `unit='m'` instead of positional `'m'`.

- **Type**: `bool`
- **Default**: `True`

```{code-block} python
>>> q = u.Quantity(1.0, "m")

>>> # Default behavior (named_unit=True)
>>> with u.config.quantity_repr.override(named_unit=False):
...     print(repr(q))
Quantity(Array(1., dtype=float32, ...), 'm')

>>> # With named_unit=True
>>> with u.config.quantity_repr.override(named_unit=True):
...     print(repr(q))
Quantity(Array(1., dtype=float32, ...), unit='m')
```

### `include_params`

Include type parameters in `repr()` for parametric quantities.

- **Type**: `bool`
- **Default**: `False`

```{code-block} python
>>> q = u.Quantity["length"](1.0, "m")

>>> with u.config.quantity_repr.override(include_params=False):
...     print(repr(q))
Quantity(Array(1., dtype=float32, weak_type=True), unit='m')

>>> with u.config.quantity_repr.override(include_params=True):
...     print(repr(q))
Quantity['length'](Array(1., dtype=float32, weak_type=True), unit='m')
```

### `indent`

Indentation width for nested structures in `repr()`.

- **Type**: `int`
- **Default**: `4`

```{code-block} python
>>> q = u.Quantity([[1.0, 2.0], [3.0, 4.0]], "m")
>>> print(repr(q))  # Default indentation (4 spaces)
Quantity(Array([[1., 2.],
                [3., 4.]], dtype=float32), unit='m')
```

## Quantity Str Options

Options for controlling `Quantity.__str__()` via `u.config.quantity_str`.

### `short_arrays`

Controls how arrays are displayed in `str()`.

- **Type**: `bool | Literal["compact"]`
- **Default**: `"compact"`

```{code-block} python
>>> q = u.Quantity([1.0, 2.0, 3.0], "m")
>>> print(str(q))  # Default behavior (compact)
Quantity['length']([1., 2., 3.], unit='m')

>>> with u.config.quantity_str.override(short_arrays=False):
...     print(str(q))
Quantity['length'](Array([1., 2., 3.], dtype=float32), unit='m')

>>> with u.config.quantity_str.override(short_arrays=True):
...     print(str(q))
Quantity['length'](f32[3], unit='m')
```

### `named_unit`

Display units as named keyword in `str()`.

- **Type**: `bool`
- **Default**: `True`

```{code-block} python
>>> q = u.Quantity(1.0, "m")
>>> print(str(q))  # Default behavior (named)
Quantity['length'](1., unit='m')

>>> with u.config.quantity_str.override(named_unit=False):
...     print(str(q))
Quantity['length'](1., 'm')
```

### `use_short_name`

Use short class names in `str()` representation.

- **Type**: `bool`
- **Default**: `False`

```{code-block} python
>>> q = u.Quantity(1.0, "m")
>>> print(str(q))
Quantity['length'](1., unit='m')

>>> with u.config.quantity_str.override(use_short_name=True):
...     print(str(q))
Q['length'](1., unit='m')
```

### `indent`

Indentation width for nested structures in `str()`.

- **Type**: `int`
- **Default**: `4`

## File-Based Configuration

`unxt` loads configuration from the nearest `pyproject.toml` when imported (searching upward from `Path.cwd()`).

### `pyproject.toml` Format

The configuration uses a hierarchical format matching the nested structure:

```toml
[tool.unxt.quantity.repr]
short_arrays = "compact"
use_short_name = true
named_unit = false
include_params = true
indent = 4

[tool.unxt.quantity.str]
short_arrays = true
named_unit = false
use_short_name = false
indent = 4
```

You only need to set the options you want to override.

Minimal example:

```toml
[tool.unxt.quantity.repr]
short_arrays = "compact"
use_short_name = true
```

The loaded configuration is applied globally for the active Python process.

## Temporary Configuration with Context Manager

Use context managers for temporary changes that are automatically restored. There are two approaches:

1. **Root config with double-underscore notation**: `u.config.override(quantity_repr__option=value)`
2. **Nested config directly**: `u.config.quantity_repr.override(option=value)`

### Basic Usage - Root Config

```{code-block} python
print(f"Original: {u.config.quantity_repr.short_arrays}")

with u.config.override(quantity_repr__short_arrays="compact", quantity_repr__use_short_name=True):
    q = u.Quantity([1.0, 2.0, 3.0], "m")
    print(repr(q))  # Q([1., 2., 3.], unit='m')
    print(f"Inside context: {u.config.quantity_repr.short_arrays}")

print(f"After context: {u.config.quantity_repr.short_arrays}")
```

### Basic Usage - Nested Config

```{code-block} python
# Cleaner syntax when configuring multiple options for one display method
with u.config.quantity_repr.override(short_arrays="compact", use_short_name=True, named_unit=False):
    q = u.Quantity([1.0, 2.0, 3.0], "m")
    print(repr(q))  # Q([1., 2., 3.], 'm')
```

### Nested Contexts

Both override methods can be nested:

```{code-block} python
with u.config.quantity_repr.override(short_arrays="compact"):
    print(f"Outer: {u.config.quantity_repr.short_arrays}")  # compact

    with u.config.quantity_repr.override(short_arrays=True):
        print(f"Inner: {u.config.quantity_repr.short_arrays}")  # True

    print(f"Back to outer: {u.config.quantity_repr.short_arrays}")  # compact
```

### Multiple Config Scopes

You can override both repr and str configs simultaneously:

```{code-block} python
>>> # Using root config with double-underscore notation
>>> with u.config.override(
...     quantity_repr__short_arrays="compact",
...     quantity_str__short_arrays=True
... ):
...     q = u.Quantity([1.0, 2.0, 3.0], "m")
...     print(repr(q), str(q))
Quantity([1., 2., 3.], unit='m') Quantity['length'](f32[3], unit='m')
>>>
>>> # Or using separate nested overrides
>>> with (
...     u.config.quantity_repr.override(short_arrays="compact"),
...     u.config.quantity_str.override(short_arrays=True)
... ):
...         q = u.Quantity([1.0, 2.0, 3.0], "m")
...         print(repr(q), str(q))
Quantity([1., 2., 3.], unit='m') Quantity['length'](f32[3], unit='m')
```

### Thread Safety

Overrides are thread-local, meaning each thread has its own independent override stack:

<!-- skip: start -->

```{code-block} python
>>> import threading
>>> def worker(name: str, setting: str | bool) -> None:
...     with u.config.quantity_repr.override(short_arrays=setting):
...         q = u.Quantity([1, 2, 3], "m")
...         print(f"{name}: {repr(q)}")
>>> thread1 = threading.Thread(target=worker, args=("Thread 1", "compact"))
>>> thread2 = threading.Thread(target=worker, args=("Thread 2", True))
>>> thread1.start()
>>> thread2.start()
>>> thread1.join()
>>> thread2.join()

# Each thread sees its own config without interference
```

<!-- skip: end -->

## Programmatic Access

Access and modify configuration options directly via the nested config objects:

<!-- skip: start -->

```{code-block} python
# Read current settings
>>> print(f"Repr short arrays: {u.config.quantity_repr.short_arrays}")
>>> print(f"Repr use short name: {u.config.quantity_repr.use_short_name}")
>>> print(f"Str short arrays: {u.config.quantity_str.short_arrays}")

# Modify settings
>>> u.config.quantity_repr.short_arrays = "compact"
>>> u.config.quantity_repr.use_short_name = True
>>> u.config.quantity_repr.named_unit = False
>>> u.config.quantity_str.short_arrays = True
>>> u.config.quantity_str.named_unit = False
```

<!-- skip: end -->

## Advanced: Traitlets Integration

Since `unxt.config` and its nested configs are traitlets `Configurable` objects, you can apply traitlets `Config` objects directly.

```{code-block} python
>>> from traitlets.config import Config

>>> # Create a Config object and configure nested configs by class name
>>> cfg = Config()
>>> cfg.QuantityReprConfig.short_arrays = "compact"
>>> cfg.QuantityReprConfig.use_short_name = True
>>> cfg.QuantityStrConfig.short_arrays = True
```

Apply to the nested config instances:

<!-- skip: start -->

```{code-block} python
>>> u.config.quantity_repr.update_config(cfg)
>>> u.config.quantity_str.update_config(cfg)
```

<!-- skip: end -->

```{code-block} python
>>> q = u.Quantity([1.0, 2.0, 3.0], "m")
>>> with u.config.quantity_repr.override(cfg):
...     print(repr(q))
Q([1., 2., 3.], unit='m')
```

Or update the root config:

<!-- skip: start -->

```{code-block} python
from traitlets.config import Config

cfg = Config()
cfg.QuantityReprConfig.short_arrays = "compact"

u.config.update_config(cfg)
```

<!-- skip: end -->

## See Also

- [Traitlets documentation](https://traitlets.readthedocs.io/)
- {doc}`quantity` - Quantity user guide
- {doc}`type-checking` - Type checking guide
