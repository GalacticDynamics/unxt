# How to control how quantities are displayed

`unxt` renders quantities through [`wadler_lindig`](https://docs.kidger.site/wadler_lindig), and the four display options are documented in {doc}`../reference/configuration`. This guide covers how to _apply_ them — for one call, for a block, for a process, or for a project.

```{code-block} python
>>> import unxt as u
>>> q = u.Q([1, 2, 3], "m")
```

## For a single call

To change the rendering of one value without touching global state, print it with `wadler_lindig` directly and pass the options as keyword arguments:

```{code-block} python
>>> import wadler_lindig as wl

>>> wl.pprint(q)  # the default
Quantity(i32[3], unit='m')

>>> wl.pprint(q, short_arrays=False)
Quantity(Array([1, 2, 3], dtype=int32), unit='m')

>>> wl.pprint(q, short_arrays="compact")
Quantity([1, 2, 3], unit='m')

>>> wl.pprint(q, named_unit=False)
Quantity(i32[3], 'm')

>>> wl.pprint(q, use_short_name=True)
Q(i32[3], unit='m')

```

The options combine:

```{code-block} python
>>> wl.pprint(q, use_short_name=True, short_arrays="compact")
Q([1, 2, 3], unit='m')

```

## For a block of code

To change the setting for a region and have it restored automatically, use the `override` context manager. If you are setting options across both sections, use the root config with `section__option` names:

```{code-block} python
>>> with u.config.override(quantity_repr__short_arrays="compact",
...                        quantity_repr__use_short_name=True):
...     print(repr(u.Q([1.0, 2.0, 3.0], "m")))
Q([1., 2., 3.], unit='m')
```

If every option belongs to one section, override that section directly — it reads better and avoids the prefixes:

```{code-block} python
>>> with u.config.quantity_repr.override(short_arrays="compact", use_short_name=True):
...     print(repr(u.Q([1.0, 2.0, 3.0], "m")))
Q([1., 2., 3.], unit='m')
```

Both forms nest, and the inner scope wins for the options it names:

```{code-block} python
>>> with u.config.quantity_repr.override(short_arrays="compact"):
...     with u.config.quantity_repr.override(short_arrays=True):
...         print(u.config.quantity_repr.short_arrays)
...     print(u.config.quantity_repr.short_arrays)
True
compact
```

Overrides are thread-local, so a worker thread's block does not leak into its siblings.

## For the whole process

To set the display once at startup, assign the attributes:

<!-- skip: start -->

```{code-block} python
u.config.quantity_repr.short_arrays = "compact"
u.config.quantity_repr.use_short_name = True
u.config.quantity_str.named_unit = False
```

<!-- skip: end -->

## For a project

Put the settings in your `pyproject.toml` under `[tool.unxts.unxt.quantity.repr]` / `[...str]` — the key list is in {doc}`../reference/configuration`. `unxt` reads the file once, at import, searching upward from the current working directory.

That last detail bites in notebooks and test runners, where `unxt` is often imported _before_ you change into the project directory. If your settings appear not to have taken effect, re-read the file:

```{code-block} python
>>> loaded = u.config.reload()  # re-read pyproject.toml from the current cwd
```

`reload()` returns `False` when no `pyproject.toml` was found, it could not be read, or it held no `[tool.unxts.unxt]` settings — so you can detect a no-op rather than guessing.

## From a traitlets `Config`

If your application already carries a `traitlets.config.Config` — an IPython or Jupyter application, for instance — apply it directly. Configure by _class_ name:

```{code-block} python
>>> from traitlets.config import Config

>>> cfg = Config()
>>> cfg.QuantityReprConfig.short_arrays = "compact"
>>> cfg.QuantityStrConfig.short_arrays = True
```

`override` accepts one, for a scoped application:

```{code-block} python
>>> with u.config.quantity_repr.override(cfg):
...     print(repr(u.Q([1.0, 2.0, 3.0], "m")))
Quantity([1., 2., 3.], unit='m')
```

A nested section takes one directly, applying it permanently:

```{code-block} python
>>> before = u.config.quantity_str.short_arrays
>>> u.config.quantity_str.update_config(cfg)
>>> u.config.quantity_str.short_arrays
True
>>> u.config.quantity_str.short_arrays = before
```

`update_config` on the _root_ config does the same but forwards to whichever nested sections the `Config` names. Capture the current value first if you intend to restore it:

```{code-block} python
>>> baseline = u.config.quantity_repr.short_arrays

>>> root_cfg = Config()
>>> root_cfg.QuantityReprConfig.short_arrays = "compact"
>>> u.config.update_config(root_cfg)
>>> u.config.quantity_repr.short_arrays
'compact'

>>> reset_cfg = Config()
>>> reset_cfg.QuantityReprConfig.short_arrays = baseline
>>> u.config.update_config(reset_cfg)
>>> u.config.quantity_repr.short_arrays == baseline
True
```

## See also

- {doc}`../reference/configuration` — every option, type and default.
- {doc}`../reference/quantity` — the classes being displayed.
