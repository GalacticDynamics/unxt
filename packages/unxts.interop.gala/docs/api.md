# `unxts.interop.gala` API

`unxts.interop.gala` exposes two conversion functions. Both are registered as [`plum`](https://beartype.github.io/plum/) conversion methods when the package is imported, so the idiomatic way to use them is through `plum.convert` (shown below). The functions are also importable directly.

```python
from unxts.interop.gala import (
    convert_gala_unitsystem_to_unxt_unitsystem,
    convert_unxt_unitsystem_to_gala_unitsystem,
)
```

## `convert_gala_unitsystem_to_unxt_unitsystem(usys)`

Convert a `gala.units.UnitSystem` to a [`unxt.unitsystems.AbstractUnitSystem`][unxt-AbstractUnitSystem].

```{code-block} python

>>> import gala.units as gu
>>> from unxts.interop.gala import convert_gala_unitsystem_to_unxt_unitsystem

>>> convert_gala_unitsystem_to_unxt_unitsystem(gu.galactic)
unitsystem(kpc, Myr, solMass, rad)

```

## `convert_unxt_unitsystem_to_gala_unitsystem(usys)`

Convert a `unxt.unitsystems.AbstractUnitSystem` to a `gala.units.UnitSystem`.

```{code-block} python

>>> import unxt
>>> from unxts.interop.gala import convert_unxt_unitsystem_to_gala_unitsystem

>>> usys = unxt.unitsystem("galactic")
>>> convert_unxt_unitsystem_to_gala_unitsystem(usys)
<UnitSystem (kpc, Myr, solMass, rad)>

```

## Using `plum.convert` (preferred)

The two calls above are equivalent to `plum.convert` with the target type, which is the recommended interface:

```{code-block} python

>>> from plum import convert

>>> convert(gu.galactic, unxt.AbstractUnitSystem)
unitsystem(kpc, Myr, solMass, rad)

>>> convert(usys, gu.UnitSystem)
<UnitSystem (kpc, Myr, solMass, rad)>

```

[unxt-AbstractUnitSystem]: https://unxt.readthedocs.io/en/latest/api/unitsystems/#unxt.unitsystems.AbstractUnitSystem
