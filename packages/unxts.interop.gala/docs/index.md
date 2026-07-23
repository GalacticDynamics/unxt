# `unxts.interop.gala`

```{toctree}
:maxdepth: 1
:hidden:

guide
api
```

The [`gala`][gala-link] package provides tools for Galactic dynamics. It is built on top of [`astropy`][astropy-link] and adds a units tool, the [`gala.units.UnitSystem`][gala-UnitSystem] class, for working with different unit systems.

`unxts.interop.gala` is the canonical location for `gala` integration. It provides conversions between `gala.units.UnitSystem` and [`unxt.unitsystems.AbstractUnitSystem`][unxt-AbstractUnitSystem] objects. Importing the package — directly, or transitively via `unxt`, which imports it when both the package and `gala` are importable — registers the conversions as a side effect. (`unxt` guards on `gala` too, so on platforms where `gala` is absent the conversions are not registered.)

## Installation

The recommended install adds `unxts.interop.gala` alongside `unxt` via the `interop-gala` [extra](https://peps.python.org/pep-0508/#extras), so it, `unxt`, and `gala` are resolved together as a compatible set:

::::{tab-set}

:::{tab-item} uv

```bash
uv add "unxt[interop-gala]"
```

:::

:::{tab-item} pip

```bash
pip install "unxt[interop-gala]"
```

:::

::::

Or install the package directly:

::::{tab-set}

:::{tab-item} uv

```bash
uv add unxts.interop.gala
```

:::

:::{tab-item} pip

```bash
pip install unxts.interop.gala
```

:::

::::

## Quick example

```{code-block} python

>>> import unxt
>>> import gala.units as gu

>>> gu.galactic  # a gala unit system
<UnitSystem (kpc, Myr, solMass, rad)>

>>> unxt.unitsystem(gu.galactic)  # as a unxt unit system
unitsystem(kpc, Myr, solMass, rad)

```

See the [guide](guide) for two-way conversion, and the [API reference](api) for the exposed functions.

[gala-link]: https://gala.adrian.pw/en/stable/
[astropy-link]: https://www.astropy.org/
[gala-UnitSystem]: https://gala.adrian.pw/en/stable/api/gala.units.UnitSystem.html
[unxt-AbstractUnitSystem]: https://unxt.readthedocs.io/en/latest/api/unitsystems/#unxt.unitsystems.AbstractUnitSystem
