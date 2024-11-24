# 🌌 Gala

The [`gala`][gala-link] package provides tools for Galactic dynamics. It is
built on top of the [`astropy`][astropy-link] package and adds an additional
units tool, the [`gala.units.UnitSystem`][gala-UnitSystem] class that can be
used to convert between different unit systems.

`unxt` supports `gala` unit systems as inputs, converting them to
[`unxt.unitsystems.AbstractUnitSystem`][unxt-AbstractUnitSystem] objects. This
conversion is automatically enabled if `gala` is installed. `unxt` is compatible
with most versions of `gala`, but to ensure that compatible versions of `gala`
and `unxt` are installed, the following installation
[extra](https://peps.python.org/pep-0508/#extras) is provided:

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

::::

The following example demonstrates how to convert a `gala` unit system to a
`unxt` unit system:

```{code-block} python

>>> import unxt
>>> import gala.units as gu

>>> gu.galactic  # gala unit system
<UnitSystem (kpc, Myr, solMass, rad)>

>>> unxt.unitsystem(gu.galactic)  # unxt unit system
unitsystem(kpc, Myr, solMass, rad)

```

Alternatively, the multiple-dispatch library on which `unxt` is built enables
2-way conversion.

```{code-block} python
>>> from plum import convert

>>> usys = convert(gu.galactic, unxt.AbstractUnitSystem)
>>> usys
unitsystem(kpc, Myr, solMass, rad)

>>> convert(usys, gu.UnitSystem)
<UnitSystem (kpc, Myr, solMass, rad)>

```

[gala-link]: https://gala.adrian.pw/en/stable/
[astropy-link]: https://www.astropy.org/
[gala-UnitSystem]:
  https://gala.adrian.pw/en/stable/api/gala.units.UnitSystem.html
[unxt-AbstractUnitSystem]:
  https://unxt.readthedocs.io/en/latest/api/unitsystems/#unxt.unitsystems.AbstractUnitSystem
