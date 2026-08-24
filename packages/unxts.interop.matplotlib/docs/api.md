# API

`unxts.interop.matplotlib` exposes the converter and a function to toggle it. Importing the package enables the converter automatically, so most users never need to call either directly.

```python
from unxts.interop.matplotlib import (
    UnxtConverter,
    setup_matplotlib_support_for_unxt,
)
```

## `setup_matplotlib_support_for_unxt(*, enable=True)`

Register (or unregister) the `unxt` quantity converter with `matplotlib`. It is called with `enable=True` when the package is imported.

- `enable` (bool, keyword-only, default `True`): if `True`, register `UnxtConverter` for `unxt.quantity.AbstractQuantity` (which covers `Quantity` and the other quantity types) so quantities can be plotted; if `False`, remove the registration.

```{code-block} python

from unxts.interop.matplotlib import setup_matplotlib_support_for_unxt

# Stop matplotlib from converting Quantity objects
setup_matplotlib_support_for_unxt(enable=False)

# Re-enable (the default on import)
setup_matplotlib_support_for_unxt(enable=True)
```

## `UnxtConverter`

A `matplotlib.units.ConversionInterface` subclass that teaches `matplotlib` how to turn an `unxt` quantity (any `AbstractQuantity`) into plottable magnitudes, and to label axes with the unit. It is registered — for `AbstractQuantity` — by `setup_matplotlib_support_for_unxt`; you rarely instantiate it yourself.

| Field | Type | Default | Purpose |
| --- | --- | --- | --- |
| `unit_format` | `str` | `"latex_inline"` | Format spec passed to `Unit.to_string` when building the axis label. |
| `axisinfo_kw` | `dict \| None` | `None` | **Deprecated.** See below. |

```{code-block} python
>>> import unxt as u
>>> from unxts.interop.matplotlib import UnxtConverter

>>> UnxtConverter().unit_format
'latex_inline'

>>> UnxtConverter(unit_format="latex").unit_format
'latex'
```

The format controls how the unit is rendered on the axis:

```{code-block} python
>>> print(UnxtConverter().axisinfo(u.unit("km"), None).label)
$\mathrm{km}$
```

:::{deprecated} 2.0.0

`axisinfo_kw` is deprecated in favour of `unit_format` and will be removed in a future release. Passing it emits a `DeprecationWarning` and its `"format"` key is copied into `unit_format`:

```{code-block} python
>>> import warnings

>>> with warnings.catch_warnings(record=True) as caught:
...     warnings.simplefilter("always")
...     converter = UnxtConverter(axisinfo_kw={"format": "latex"})
>>> caught[0].category.__name__
'DeprecationWarning'
>>> converter.unit_format
'latex'
```

Use `UnxtConverter(unit_format="latex")` instead.

:::
