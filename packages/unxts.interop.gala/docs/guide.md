# `gala` Interoperability Guide

This guide shows how to convert between [`gala`][gala-link]'s `gala.units.UnitSystem` and `unxt`'s [`unxt.unitsystems.AbstractUnitSystem`][unxt-AbstractUnitSystem].

## Setup

Importing `unxts.interop.gala` registers the conversions with [`plum`](https://beartype.github.io/plum/) as a side effect. `unxt` imports the package automatically when it is installed, so in practice you usually only need to import `unxt` and `gala`:

```{code-block} python

>>> import unxt
>>> import gala.units as gu

```

## `gala` → `unxt`

The most direct route is {func}`unxt.unitsystem`, which accepts a `gala.units.UnitSystem`:

```{code-block} python

>>> gu.galactic
<UnitSystem (kpc, Myr, solMass, rad)>

>>> unxt.unitsystem(gu.galactic)
unitsystem(kpc, Myr, solMass, rad)

```

Because the conversions are registered with `plum`, you can equivalently use `plum.convert` with the target type:

```{code-block} python

>>> from plum import convert

>>> usys = convert(gu.galactic, unxt.AbstractUnitSystem)
>>> usys
unitsystem(kpc, Myr, solMass, rad)

```

## `unxt` → `gala`

The reverse conversion goes through `plum.convert` with `gala.units.UnitSystem` as the target:

```{code-block} python

>>> convert(usys, gu.UnitSystem)
<UnitSystem (kpc, Myr, solMass, rad)>

```

## Round trip

Converting a unit system to the other library and back yields an equivalent unit system:

```{code-block} python

>>> back = convert(convert(gu.galactic, unxt.AbstractUnitSystem), gu.UnitSystem)
>>> back
<UnitSystem (kpc, Myr, solMass, rad)>

```

## See Also

- [API reference](api) — the exposed conversion functions
- [gala documentation](https://gala.adrian.pw/en/stable/)
- [unxt unit systems](https://unxt.readthedocs.io/)

[gala-link]: https://gala.adrian.pw/en/stable/
[unxt-AbstractUnitSystem]: https://unxt.readthedocs.io/en/latest/api/unitsystems/#unxt.unitsystems.AbstractUnitSystem
