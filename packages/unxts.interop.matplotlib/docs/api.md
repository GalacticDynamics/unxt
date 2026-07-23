# `unxts.interop.matplotlib` API

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

A `matplotlib.units.ConversionInterface` subclass that teaches `matplotlib` how to turn an `unxt` quantity (any `AbstractQuantity`) into plottable magnitudes (and to label axes with the unit). It is registered — for `AbstractQuantity` — by `setup_matplotlib_support_for_unxt`; you rarely instantiate it yourself.
