# unxts.interop.gala

[gala](https://gala.adrian.pw/) integration for [unxt](https://github.com/GalacticDynamics/unxt).

This is the canonical package (`unxts.interop.gala`). It provides conversions between `gala.units.UnitSystem` and `unxt.unitsystems.AbstractUnitSystem`.

## Install

```bash
pip install unxts.interop.gala
```

## Usage

```python
import unxts.interop.gala
```

## Public API

`unxts.interop.gala` exposes two conversion functions:

- `convert_gala_unitsystem_to_unxt_unitsystem`
- `convert_unxt_unitsystem_to_gala_unitsystem`

These are `plum.conversion_method`s, registered with `plum`'s dispatch table as a side effect of importing `unxts.interop.gala`. Prefer `plum.convert(usys, ...)` over calling either function directly, e.g.:

```python
from plum import convert
import unxt as u
import gala.units as gu

convert(gu.galactic, u.AbstractUnitSystem)
convert(u.unitsystem("galactic"), gu.UnitSystem)
```
