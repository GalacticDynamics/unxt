# Configuration

`unxts.parametric.config` — the parametric counterpart to `unxt.config`. It controls whether the dimension type parameter (e.g. `['length']`) appears in a `ParametricQuantity`'s `repr()` / `str()`. The other display settings (`short_arrays`, `use_short_name`, `named_unit`, `indent`) are shared with all quantities and remain in `unxt.config`; see the [unxt Configuration guide](../../guides/configuration).

```{code-block} python
>>> import unxt as u
>>> import unxts.parametric as up
```

## `include_params`

By default the dimension parameter is hidden in `repr()` and shown in `str()`:

```{code-block} python
>>> up.config.quantity_repr.include_params
False
>>> up.config.quantity_str.include_params
True

>>> q = up.PQ([1, 2, 3], "m")
>>> repr(q)
"ParametricQuantity(Array([1, 2, 3], dtype=int32), unit='m')"
>>> print(q)
ParametricQuantity['length']([1, 2, 3], unit='m')
```

Override it temporarily (thread-local) with the top-level `override` — using double-underscore notation for the nested config — or on a sub-config directly:

```{code-block} python
>>> with up.config.override(quantity_repr__include_params=True):
...     print(repr(q))
ParametricQuantity['length'](Array([1, 2, 3], dtype=int32), unit='m')

>>> print(repr(q))  # restored on exit
ParametricQuantity(Array([1, 2, 3], dtype=int32), unit='m')
```

## `pyproject.toml`

It can also be set in the nearest `pyproject.toml`, under `[tool.unxts.parametric...]` (auto-loaded at import):

```toml
[tool.unxts.parametric.quantity.repr]
include_params = true

[tool.unxts.parametric.quantity.str]
include_params = false
```
