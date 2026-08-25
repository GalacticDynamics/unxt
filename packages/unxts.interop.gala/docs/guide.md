# How to convert between gala and unxt unit systems

This guide shows you how to convert between [`gala`][gala-link]'s `gala.units.UnitSystem` and `unxt`'s [`unxt.unitsystems.AbstractUnitSystem`][unxt-AbstractUnitSystem].

## Setup

Importing `unxts.interop.gala` registers the conversions with [`plum`](https://beartype.github.io/plum/) as a side effect. `unxt` imports the package automatically when both it and `gala` are importable (`gala` can be absent on some platforms, e.g. Windows, in which case the conversions are not registered), so in practice you usually only need to import `unxt` and `gala`:

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

Converting a unit system to the other library and back gives you the same **base units**, and the repr is identical:

```{code-block} python

>>> back = convert(convert(gu.galactic, unxt.AbstractUnitSystem), gu.UnitSystem)
>>> back
<UnitSystem (kpc, Myr, solMass, rad)>

```

:::{warning}

The round trip is **not** lossless, and the recovered system does not compare equal to the original:

```{code-block} python

>>> back == gu.galactic
False

```

A `gala.units.UnitSystem` can register a preferred unit for a _derived_ dimension alongside its base units, and `gala.galactic` does exactly that — it prefers `km / s` for velocity, even though its base units would give `kpc / Myr`:

```{code-block} python

>>> gu.galactic["speed"]
Unit("km / s")

```

An `unxt` unit system has no slot for that: it holds one unit per _base_ dimension and composes everything else, so `usys["velocity"]` is always `kpc / Myr`. The preference is dropped on the way in and cannot be reconstructed on the way back.

If you need the original object, keep a reference to it rather than round-tripping. If you only need the base units, the round trip is fine.

:::

## See also

- [API](api) — the exposed conversion functions.
- [unxt unit systems reference](https://unxt.readthedocs.io/en/latest/reference/unitsystems.html)
- [gala documentation](https://gala.adrian.pw/en/stable/)

[gala-link]: https://gala.adrian.pw/en/stable/
[unxt-AbstractUnitSystem]: https://unxt.readthedocs.io/en/latest/reference/api/unitsystems.html#unxt.unitsystems.AbstractUnitSystem
