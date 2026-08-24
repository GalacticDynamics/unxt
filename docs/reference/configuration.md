# Configuration

`unxt.config` controls how quantities render in `repr()` and `str()`. It is a [traitlets](https://traitlets.readthedocs.io/) `Configurable` with two nested sections:

| Object | Class | Controls |
| --- | --- | --- |
| `u.config.quantity_repr` | `QuantityReprConfig` | `repr()` of every quantity class |
| `u.config.quantity_str` | `QuantityStrConfig` | `str()` of every quantity class |

Each section holds the same four options with different defaults. For how to set them, see {doc}`../how-to/control-display`.

```{code-block} python
>>> import unxt as u
```

## Options

| Option | Type | `quantity_repr` default | `quantity_str` default |
| --- | --- | --- | --- |
| `short_arrays` | `bool \| Literal["compact"]` | `False` | `"compact"` |
| `use_short_name` | `bool` | `False` | `False` |
| `named_unit` | `bool` | `True` | `True` |
| `indent` | `int` | `4` | `4` |

### `short_arrays`

How the value is rendered.

| Value       | Rendering                                                |
| ----------- | -------------------------------------------------------- |
| `False`     | full array repr — `Array([1., 2., 3.], dtype=float32)`   |
| `True`      | shape and dtype summary — `f32[3]`                       |
| `"compact"` | values without the `Array(...)` wrapper — `[1., 2., 3.]` |

```{code-block} python
>>> q = u.Q([1.0, 2.0, 3.0], "m")

>>> with u.config.quantity_repr.override(short_arrays=False):
...     print(repr(q))
Quantity(Array([1., 2., 3.], dtype=float32), unit='m')

>>> with u.config.quantity_repr.override(short_arrays=True):
...     print(repr(q))
Quantity(f32[3], unit='m')

>>> with u.config.quantity_repr.override(short_arrays="compact"):
...     print(repr(q))
Quantity([1., 2., 3.], unit='m')
```

### `use_short_name`

When `True`, a class renders under its short name where it has one — `Quantity` becomes `Q`.

```{code-block} python
>>> q1 = u.Q(1.0, "m")

>>> with u.config.quantity_repr.override(use_short_name=False):
...     print(repr(q1))
Quantity(Array(1., dtype=float32...), unit='m')

>>> with u.config.quantity_repr.override(use_short_name=True):
...     print(repr(q1))
Q(Array(1., dtype=float32...), unit='m')
```

### `named_unit`

When `True`, the unit renders as the keyword `unit='m'`; when `False`, positionally as `'m'`.

```{code-block} python
>>> with u.config.quantity_repr.override(named_unit=False):
...     print(repr(q1))
Quantity(Array(1., dtype=float32...), 'm')

>>> with u.config.quantity_repr.override(named_unit=True):
...     print(repr(q1))
Quantity(Array(1., dtype=float32...), unit='m')
```

### `indent`

Indentation width, in spaces, for nested structures.

```{code-block} python
>>> print(repr(u.Q([[1.0, 2.0], [3.0, 4.0]], "m")))  # default: 4
Quantity(Array([[1., 2.],
                [3., 4.]], dtype=float32), unit='m')
```

## `str()` defaults

`quantity_str` differs from `quantity_repr` only in the default for `short_arrays`, which is `"compact"`:

```{code-block} python
>>> print(str(q))
Quantity([1., 2., 3.], unit='m')

>>> with u.config.quantity_str.override(short_arrays=False):
...     print(str(q))
Quantity(Array([1., 2., 3.], dtype=float32), unit='m')

>>> with u.config.quantity_str.override(short_arrays=True):
...     print(str(q))
Quantity(f32[3], unit='m')

>>> with u.config.quantity_str.override(named_unit=False):
...     print(str(q1))
Quantity(1., 'm')

>>> with u.config.quantity_str.override(use_short_name=True):
...     print(str(q1))
Q(1., unit='m')
```

## `pyproject.toml` keys

`unxt` reads the nearest `pyproject.toml` once, at import, searching upward from `Path.cwd()`. Only the keys present in the file are applied; every other setting keeps its default.

| Section                           | Applies to               |
| --------------------------------- | ------------------------ |
| `[tool.unxts.unxt.quantity.repr]` | `u.config.quantity_repr` |
| `[tool.unxts.unxt.quantity.str]`  | `u.config.quantity_str`  |

```toml
[tool.unxts.unxt.quantity.repr]
short_arrays = "compact"
use_short_name = true
named_unit = false
indent = 4

[tool.unxts.unxt.quantity.str]
short_arrays = true
named_unit = false
use_short_name = false
indent = 4
```

The loaded configuration applies globally for the active Python process.

:::{warning}

The section was named `[tool.unxt...]` before v2. A leftover `[tool.unxt]` section is **ignored**, and `unxt` emits a `DeprecationWarning` at import. See {doc}`../how-to/migrate-to-v2`.

:::

## API

| Member | Signature | Returns |
| --- | --- | --- |
| `config.override` | `(**opts)` — nested options use `section__option` | context manager |
| `config.<section>.override` | `(**opts)` or `(traitlets_config)` | context manager |
| `config.reload` | `()` | `bool` — `False` if no settings were found or applied |
| `config.update_config` | `(traitlets.config.Config)` | `None` |

Overrides are thread-local: each thread keeps its own independent override stack.

## See also

- {doc}`../how-to/control-display` — recipes for setting these.
- {doc}`quantity` — the classes whose display these control.
